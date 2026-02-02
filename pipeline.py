"""
监控视频人形检测增量处理 Pipeline

基于生产者-消费者模式的流式处理架构，使用独立的 Detector Worker 优化批处理效率：

架构图：
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Producer (主进程)                            │
    │  scan → group → push to task_queue                              │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Task Queue        │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  I/O Worker 1 │   │  I/O Worker 2 │   │  I/O Worker 3 │
    │  ffmpeg 解码  │   │  ffmpeg 解码  │   │  ffmpeg 解码  │
    │  (多线程)     │   │  (多线程)     │   │  (多线程)     │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Detection Queue     │
                    │   (批量帧数据)         │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Detector Workers    │
                    │   (多进程并行推理)    │
                    │   批量推理 → 返回结果 │
                    └───────────────────────┘

性能优化（针对 FFmpeg 解码瓶颈）：
- 两阶段检测 + 关键帧 seek：粗筛阶段使用关键帧跳转，快速排除无人视频
- fps 滤镜跳帧：精筛阶段比 select 滤镜快 ~10%，基于时间戳选帧更高效
- 减少 I/O Worker：避免多个 ffmpeg 进程争用 CPU
- 增加 decode_threads：每个 ffmpeg 使用更多线程，充分利用多核
- 多 Detector Worker：多个 YOLO 模型实例并行推理
- 异步预取（深度 8）：I/O Worker 可同时发送多个 batch，不必等待结果
- 自动配置：根据 CPU 核心数自动分配最佳参数

两阶段检测原理：
1. 粗筛阶段：使用 -ss input seek 直接跳转到关键帧（I帧），避免解码 P/B 帧
   - H.264/H.265 关键帧间隔通常 1-2 秒，seek 比完整解码快 5-10 倍
   - 并行读取多个时间点的帧，充分利用 I/O 等待时间
   - 无人视频直接跳过，节省 80%+ 解码时间
2. 精筛阶段：使用 fps 滤镜精确定位人物时间段

支持多种监控品牌的文件命名规则（通过 timestamp_parser 模块扩展）：
- 小米 (xiaomi): 文件夹 YYYYMMDDHH + 文件名 MMmSSs_TIMESTAMP.mp4
- 海康威视 (hikvision): 文件名 YYYYMMDDHHMMSS.mp4
- 大华 (dahua): 文件名 YYYY-MM-DD HH-MM-SS.mp4
- 通用 (generic): 自动检测多种常见格式
"""

import json
import shutil
import tempfile
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
import logging
import os
import numpy as np

from file_time_utils import get_file_times, set_file_times
from timestamp_parser import (
    get_parser, 
    BaseTimestampParser, 
    SUPPORTED_BRANDS,
    list_parsers
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class SourceFileInfo:
    """源文件信息"""
    rel_path: str
    abs_path: str
    created: str  # ISO 格式字符串（从文件名解析）
    created_timestamp: float  # Unix 时间戳（从文件名解析）
    duration: float = 0.0


# ============================================================
# 自动配置函数
# ============================================================

def get_cpu_count() -> int:
    """获取系统 CPU 核心数"""
    try:
        # 优先使用 os.cpu_count()
        cpu_count = os.cpu_count()
        if cpu_count and cpu_count > 0:
            return cpu_count
    except Exception:
        pass
    
    # 回退到 multiprocessing
    try:
        return mp.cpu_count()
    except Exception:
        pass
    
    # 默认值
    return 4


def get_auto_config(cpu_count: int | None = None) -> dict:
    """
    根据 CPU 核心数自动计算最佳配置
    
    策略（针对 FFmpeg 解码瓶颈优化）：
    - I/O Workers: 减少数量，避免多个 ffmpeg 进程争用 CPU
    - Detectors: 保持适量（YOLO 推理不是瓶颈）
    - Batch Size: 适中即可
    - Decode Threads: 增加每个 ffmpeg 的解码线程，充分利用多核
    
    优化原理：
    - Benchmark 显示 FFmpeg 解码占 90% 耗时，YOLO 推理仅 10%
    - 高分辨率视频(2304x1296)解码需要大量 CPU 资源
    - 减少并发 ffmpeg 进程，增加每个进程的线程数，可提升整体吞吐量
    
    优化配置表（针对解码瓶颈）：
    | CPU 核心数 | I/O Workers | Detectors | Batch Size | Decode Threads |
    |------------|-------------|-----------|------------|----------------|
    | 1-2 核     | 1           | 1         | 16         | 2              |
    | 3-4 核     | 2           | 2         | 32         | 2              |
    | 5-8 核     | 2           | 4         | 48         | 4              |
    | 9-16 核    | 3           | 6         | 64         | 4              |
    | 17+ 核     | 4           | 8         | 64         | 6              |
    
    Args:
        cpu_count: CPU 核心数（如果为 None，自动获取）
        
    Returns:
        dict: {'num_workers': int, 'num_detectors': int, 'batch_size': int, 'decode_threads': int}
    """
    if cpu_count is None:
        cpu_count = get_cpu_count()
    
    if cpu_count <= 2:
        return {
            'num_workers': 1,
            'num_detectors': 1,
            'batch_size': 16,
            'decode_threads': 2
        }
    elif cpu_count <= 4:
        return {
            'num_workers': 2,
            'num_detectors': 2,
            'batch_size': 32,
            'decode_threads': 2
        }
    elif cpu_count <= 8:
        return {
            'num_workers': 2,
            'num_detectors': 4,
            'batch_size': 48,
            'decode_threads': 4
        }
    elif cpu_count <= 16:
        return {
            'num_workers': 3,
            'num_detectors': 6,
            'batch_size': 64,
            'decode_threads': 4
        }
    else:
        return {
            'num_workers': 4,
            'num_detectors': 8,
            'batch_size': 64,
            'decode_threads': 6
        }


# ============================================================
# 工具函数
# ============================================================

def format_timestamp(timestamp: float) -> str:
    """将时间戳格式化为 ISO 字符串"""
    try:
        return datetime.fromtimestamp(timestamp).isoformat()
    except Exception:
        return ""


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    
    return 0.0


def probe_video_format(video_path: str) -> dict | None:
    """获取视频格式信息"""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        
        info = {
            'duration': float(data.get('format', {}).get('duration', 0)),
            'format_name': data.get('format', {}).get('format_name', 'unknown'),
            'video_codec': None,
            'audio_codec': None,
        }
        
        for stream in data.get('streams', []):
            codec_type = stream.get('codec_type')
            if codec_type == 'video' and info['video_codec'] is None:
                info['video_codec'] = stream.get('codec_name', 'unknown')
            elif codec_type == 'audio' and info['audio_codec'] is None:
                info['audio_codec'] = stream.get('codec_name', 'unknown')
        
        return info
        
    except Exception:
        return None


def source_file_to_dict(sf: SourceFileInfo) -> dict:
    """将 SourceFileInfo 转为可序列化的 dict"""
    return {
        "rel_path": sf.rel_path,
        "abs_path": sf.abs_path,
        "created": sf.created,
        "created_timestamp": sf.created_timestamp,
        "duration": sf.duration
    }


def dict_to_source_file(d: dict) -> SourceFileInfo:
    """从 dict 恢复 SourceFileInfo"""
    return SourceFileInfo(
        rel_path=d["rel_path"],
        abs_path=d["abs_path"],
        created=d["created"],
        created_timestamp=d["created_timestamp"],
        duration=d["duration"]
    )


# ============================================================
# 日志管理（进程安全）
# ============================================================

class ProcessingLog:
    """
    处理日志管理器（进程安全）
    
    日志记录格式：
    - status: "has_output" (有人物，生成了输出文件) 或 "no_human" (无人物，跳过)
    - output_relative_filepath: 输出文件相对路径（仅 has_output 状态有）
    - created: 组内第一个文件的创建时间
    - src_files: 源文件列表
    
    功能：
    - 加载现有日志，验证输出文件是否存在（仅对 has_output 状态验证）
    - 移除失效的日志记录（has_output 状态但输出文件不存在）
    - 追加新的处理结果（加锁）
    """
    
    def __init__(self, log_file: Path, output_base_dir: Path, lock: mp.Lock):
        self.log_file = log_file
        self.output_base_dir = output_base_dir
        self._lock = lock
        self._processed_sources: set[str] = set()  # rel_path set
        self._valid_entries: list[dict] = []  # 有效的日志条目
        self._invalid_count = 0
        self._load_and_validate_log()
    
    def _load_and_validate_log(self):
        """加载并验证现有的处理日志，移除失效记录"""
        if not self.log_file.exists():
            return
        
        with self._lock:
            try:
                entries = []
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue
                
                # 验证每条记录
                valid_entries = []
                for entry in entries:
                    status = entry.get('status', 'has_output')  # 兼容旧格式
                    
                    if status == 'no_human':
                        # 无人物的记录，直接有效（不需要验证输出文件）
                        valid_entries.append(entry)
                        for src in entry.get('src_files', []):
                            rel_path = src.get('rel_path')
                            if rel_path:
                                self._processed_sources.add(rel_path)
                    else:
                        # has_output 状态，需要验证输出文件是否存在
                        output_rel_path = entry.get('output_relative_filepath', '')
                        output_path = self.output_base_dir / output_rel_path
                        
                        if output_path.exists():
                            # 输出文件存在，记录有效
                            valid_entries.append(entry)
                            for src in entry.get('src_files', []):
                                rel_path = src.get('rel_path')
                                if rel_path:
                                    self._processed_sources.add(rel_path)
                        else:
                            # 输出文件不存在，记录失效
                            self._invalid_count += 1
                            logger.info(f"移除失效日志记录: {output_rel_path} (文件不存在)")
                
                self._valid_entries = valid_entries
                
                # 如果有失效记录，重写日志文件
                if self._invalid_count > 0:
                    self._rewrite_log()
                    
            except Exception as e:
                logger.warning(f"加载处理日志失败: {e}")
    
    def _rewrite_log(self):
        """重写日志文件（仅保留有效记录）"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                for entry in self._valid_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"已清理 {self._invalid_count} 条失效日志记录")
        except Exception as e:
            logger.error(f"重写日志文件失败: {e}")
    
    def is_processed(self, rel_path: str) -> bool:
        """检查源文件是否已处理"""
        return rel_path in self._processed_sources
    
    def get_processed_sources(self) -> set[str]:
        """获取已处理的源文件集合"""
        return self._processed_sources.copy()
    
    def append_result(self, result_dict: dict):
        """追加处理结果到日志（进程安全）"""
        with self._lock:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                
                # 更新内存中的记录
                for src in result_dict.get('src_files', []):
                    rel_path = src.get('rel_path')
                    if rel_path:
                        self._processed_sources.add(rel_path)
                    
            except Exception as e:
                logger.error(f"写入处理日志失败: {e}")
    
    def get_stats(self) -> dict:
        """获取日志统计信息"""
        return {
            "total_processed_sources": len(self._processed_sources),
            "invalid_entries_removed": self._invalid_count,
            "log_file": str(self.log_file)
        }


# ============================================================
# 文件扫描器
# ============================================================

class FileScanner:
    """
    文件扫描器
    
    使用指定品牌的时间戳解析器从文件名解析时间，并行获取文件时长
    """
    
    SUPPORTED_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
        '.webm', '.m4v', '.3gp', '.ts'
    }
    
    def __init__(
        self, 
        input_dir: Path, 
        processed_sources: set[str],
        scan_workers: int = 8,
        timestamp_parser: BaseTimestampParser = None,
        camera_brand: str = "xiaomi"
    ):
        """
        初始化文件扫描器
        
        Args:
            input_dir: 输入视频目录
            processed_sources: 已处理的源文件集合
            scan_workers: 并行扫描线程数
            timestamp_parser: 时间戳解析器实例（优先使用）
            camera_brand: 监控品牌名称（当 timestamp_parser 为 None 时使用）
        """
        self.input_dir = input_dir
        self.processed_sources = processed_sources
        self.scan_workers = scan_workers
        
        # 初始化时间戳解析器
        if timestamp_parser is not None:
            self._parser = timestamp_parser
        else:
            self._parser = get_parser(camera_brand)
        
        logger.info(f"使用时间戳解析器: {self._parser.brand} ({self._parser.description})")
    
    def _get_file_info(self, file_path: Path) -> SourceFileInfo | None:
        """获取单个文件的信息"""
        try:
            # 使用解析器从文件名解析时间戳
            created_timestamp = self._parser.parse(file_path)
            if created_timestamp is None:
                # 如果无法从文件名解析，回退到文件系统时间
                times = get_file_times(str(file_path))
                if times:
                    created_timestamp = times.get('created_time', 0)
                else:
                    logger.warning(f"无法获取文件时间: {file_path}")
                    return None
            
            # 获取视频时长
            duration = get_video_duration(str(file_path))
            rel_path = str(file_path.relative_to(self.input_dir))
            
            return SourceFileInfo(
                rel_path=rel_path,
                abs_path=str(file_path),
                created=format_timestamp(created_timestamp),
                created_timestamp=created_timestamp,
                duration=duration
            )
        except Exception as e:
            logger.warning(f"获取文件信息失败 {file_path}: {e}")
            return None
    
    def scan(self) -> list[SourceFileInfo]:
        """
        并行扫描所有未处理的文件
        
        Returns:
            按创建时间排序的源文件列表
        """
        # 1. 收集所有未处理的文件路径
        unprocessed_files = []
        skipped_count = 0
        
        for file_path in self.input_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            rel_path = str(file_path.relative_to(self.input_dir))
            if rel_path in self.processed_sources:
                skipped_count += 1
                continue
            
            unprocessed_files.append(file_path)
        
        if skipped_count > 0:
            logger.info(f"跳过 {skipped_count} 个已处理的文件")
        
        if not unprocessed_files:
            return []
        
        # 2. 并行获取文件信息
        source_files = []
        total_files = len(unprocessed_files)
        processed = 0
        
        with ThreadPoolExecutor(max_workers=self.scan_workers) as executor:
            futures = {executor.submit(self._get_file_info, f): f for f in unprocessed_files}
            
            for future in futures:
                result = future.result()
                if result:
                    source_files.append(result)
                processed += 1
                # 每 100 个文件输出一次进度
                if processed % 100 == 0 or processed == total_files:
                    print(f"   扫描进度: {processed}/{total_files} ({processed*100//total_files}%)")
        
        # 3. 按创建时间排序
        source_files.sort(key=lambda x: x.created_timestamp)
        
        return source_files


# ============================================================
# 连续性分组器（流式）
# ============================================================

class ConsecutiveGrouper:
    """
    连续性分组器（流式）
    
    遍历排序后的文件列表，实时判断连续性并产出分组
    """
    
    def __init__(self, time_threshold: float = 300.0):
        self.time_threshold = time_threshold
    
    def is_consecutive(self, prev: SourceFileInfo, curr: SourceFileInfo) -> bool:
        """判断两个文件是否连续"""
        time_diff = curr.created_timestamp - prev.created_timestamp
        
        return (
            prev.duration > 0 and
            -5.0 <= time_diff <= self.time_threshold and
            abs(time_diff - prev.duration) < self.time_threshold
        )
    
    def group_streaming(self, source_files: list[SourceFileInfo]):
        """
        流式分组生成器
        
        Args:
            source_files: 已按创建时间排序的文件列表
            
        Yields:
            每次产出一个分组
        """
        if not source_files:
            return
        
        current_group = [source_files[0]]
        
        for i in range(1, len(source_files)):
            prev = source_files[i - 1]
            curr = source_files[i]
            
            if self.is_consecutive(prev, curr):
                current_group.append(curr)
            else:
                # 当前组结束，产出
                yield current_group
                current_group = [curr]
        
        # 产出最后一组
        if current_group:
            yield current_group


# ============================================================
# Detector Worker（独立进程，负责 YOLO 推理）
# ============================================================

# 不兼容的音频编码
INCOMPATIBLE_AUDIO_CODECS = {
    'pcm_alaw', 'pcm_mulaw', 'pcm_s16le', 'pcm_s16be',
    'pcm_s24le', 'pcm_s24be', 'pcm_s32le', 'pcm_s32be',
    'pcm_f32le', 'pcm_f32be', 'pcm_u8', 'adpcm_ms',
}


def detector_worker(
    detection_queue: mp.Queue,
    response_queues: dict,  # {worker_id: Queue}
    model_name: str,
    conf_threshold: float,
    stop_event: mp.Event,
    detector_id: int = 1
):
    """
    Detector Worker - 独立进程，负责 YOLO 模型推理
    
    可以启动多个实例并行处理，充分利用 CPU
    """
    from ultralytics import YOLO
    
    logger.info(f"Detector {detector_id}: 启动中，加载模型 {model_name}...")
    
    # 加载模型
    model = YOLO(model_name)
    target_class = 0  # person
    
    logger.info(f"Detector {detector_id}: 模型加载完成")
    
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            # 从队列获取请求
            try:
                request = detection_queue.get(timeout=0.1)
            except Empty:
                continue
            
            if request is None:
                # 把 None 放回队列，让其他 Detector 也能收到终止信号
                detection_queue.put(None)
                logger.info(f"Detector {detector_id}: 收到终止信号，已处理 {processed_count} 个请求，退出")
                break
            
            request_id, worker_id, frames, frame_times = request
            
            if not frames:
                # 空请求，返回空结果
                if worker_id in response_queues:
                    response_queues[worker_id].put((request_id, []))
                continue
            
            # 直接批量推理
            try:
                results = model(
                    frames,
                    verbose=False,
                    conf=conf_threshold,
                    classes=[target_class]
                )
                
                # 解析结果
                frame_results = []
                for i, result in enumerate(results):
                    human_count = len(result.boxes)
                    frame_results.append((frame_times[i], human_count))
                
                # 发送响应
                if worker_id in response_queues:
                    response_queues[worker_id].put((request_id, frame_results))
                
                processed_count += 1
                    
            except Exception as e:
                logger.error(f"Detector {detector_id}: 推理失败: {e}")
                if worker_id in response_queues:
                    response_queues[worker_id].put((request_id, []))
                
        except Exception as e:
            logger.error(f"Detector {detector_id}: 处理出错: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Detector {detector_id}: 退出")


# ============================================================
# I/O Worker（独立进程，负责读帧、切片、合并）
# ============================================================

def io_worker(
    task_queue: mp.Queue,
    worker_id: int,
    output_dir: str,
    log_file: str,
    log_lock: mp.Lock,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    sample_interval: float,
    batch_size: int,
    result_counter: mp.Value,
    counter_lock: mp.Lock,
    decode_threads: int = 2,
    coarse_interval: float | None = None,  # 粗筛间隔
    use_keyframe_seek: bool = True  # 精筛是否使用关键帧 seek
):
    """
    I/O Worker - 负责视频读取、帧提取、切片、合并
    
    检测任务委托给 Detector Worker
    支持两阶段检测（粗筛 + 精筛）+ 关键帧 seek 优化
    """
    logger.info(f"I/O Worker {worker_id}: 启动")
    
    output_path = Path(output_dir)
    log_path = Path(log_file)
    
    processed_count = 0
    success_count = 0
    
    while True:
        try:
            task = task_queue.get()
            
            if task is None:
                logger.info(f"I/O Worker {worker_id}: 收到终止信号，已处理 {processed_count} 组，成功 {success_count} 组")
                break
            
            group_idx, group_dicts = task
            group = [dict_to_source_file(d) for d in group_dicts]
            
            logger.info(f"I/O Worker {worker_id}: 开始处理组 {group_idx} ({len(group)} 个文件)")
            
            # 处理任务（支持两阶段检测 + 关键帧 seek）
            result = process_group_streaming(
                group=group,
                output_dir=output_path,
                detection_queue=detection_queue,
                response_queue=response_queue,
                worker_id=worker_id,
                group_idx=group_idx,
                sample_interval=sample_interval,
                batch_size=batch_size,
                decode_threads=decode_threads,
                coarse_interval=coarse_interval,
                use_keyframe_seek=use_keyframe_seek
            )
            
            processed_count += 1
            
            if result:
                # 写入日志（加锁）
                with log_lock:
                    try:
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    except Exception as e:
                        logger.error(f"I/O Worker {worker_id}: 写入日志失败: {e}")
                
                # 根据状态输出不同的日志
                if result.get('status') == 'has_output':
                    success_count += 1
                    logger.info(f"I/O Worker {worker_id}: ✅ 组 {group_idx} 完成 → {result['output_relative_filepath']}")
                else:
                    logger.info(f"I/O Worker {worker_id}: ⏭️  组 {group_idx} 无人物，已记录跳过")
            else:
                logger.warning(f"I/O Worker {worker_id}: ⚠️  组 {group_idx} 处理返回空结果")
            
            # 更新计数器
            with counter_lock:
                result_counter.value += 1
                
        except Exception as e:
            logger.error(f"I/O Worker {worker_id}: 处理失败: {e}")
            import traceback
            traceback.print_exc()
            with counter_lock:
                result_counter.value += 1
            continue
    
    logger.info(f"I/O Worker {worker_id}: 退出")


def _build_no_human_result(group: list[SourceFileInfo], first_src: SourceFileInfo) -> dict:
    """构建无人物状态的结果记录"""
    src_results = [{
        "rel_path": src_file.rel_path,
        "created": src_file.created,
        "max_human_count": 0,
        "segments": []
    } for src_file in group]
    
    return {
        "status": "no_human",
        "created": first_src.created,
        "src_files": src_results
    }


def process_group_streaming(
    group: list[SourceFileInfo],
    output_dir: Path,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    worker_id: int,
    group_idx: int,
    sample_interval: float,
    batch_size: int,
    progress_interval: int = 5,
    decode_threads: int = 2,
    coarse_interval: float | None = None,  # 粗筛间隔
    use_keyframe_seek: bool = True  # 精筛是否使用关键帧 seek
) -> dict | None:
    """
    处理一组文件，流式检测人物片段
    
    返回格式：
    - status: "has_output" (有人物，生成了输出文件) 或 "no_human" (无人物，跳过)
    - output_relative_filepath: 输出文件相对路径（仅 has_output 状态有）
    - created: 组内第一个文件的创建时间
    - src_files: 源文件列表
    
    两阶段检测优化（关键帧 seek）：
    1. 粗筛阶段：使用较大间隔 + 关键帧 seek 快速判断视频是否有人
       - 无人 → 直接跳过，节省 80%+ 的解码时间
       - 有人 → 进入精筛阶段
    2. 精筛阶段：使用关键帧 seek（默认）精确定位人物时间段
       - 直接 seek 到每个 I 帧，速度比 fps 滤镜快 5-10 倍
    
    其他优化：
    - 并行读取多个关键帧，充分利用 I/O 等待时间
    - segment 边界留出采样间隔的 buffer
    """
    if not group:
        return None
    
    total_files = len(group)
    log_prefix = f"Worker {worker_id} 组 {group_idx}"
    first_src = group[0]
    
    # 收集所有视频的人物片段
    all_segments: list[tuple[SourceFileInfo, list[dict]]] = []
    max_person_count = 0
    files_with_human = 0
    files_skipped_by_coarse = 0  # 被粗筛跳过的文件数
    
    for idx, src_file in enumerate(group, 1):
        video_path = src_file.abs_path
        
        # 处理单个视频，流式检测（支持两阶段 + 关键帧 seek）
        segments, video_max_count = process_video_streaming(
            video_path=video_path,
            detection_queue=detection_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            group_idx=group_idx,
            video_idx=idx,
            sample_interval=sample_interval,
            batch_size=batch_size,
            decode_threads=decode_threads,
            coarse_interval=coarse_interval,
            use_keyframe_seek=use_keyframe_seek
        )
        
        if segments:
            all_segments.append((src_file, segments))
            max_person_count = max(max_person_count, video_max_count)
            files_with_human += 1
        
        # 输出进度
        if total_files > progress_interval and idx % progress_interval == 0:
            logger.info(f"{log_prefix}: 处理进度 {idx}/{total_files} ({idx*100//total_files}%)")
    
    if total_files > progress_interval:
        logger.info(f"{log_prefix}: 检测完成, {files_with_human}/{total_files} 个文件包含人物")
    
    # 如果没有人物，返回 no_human 状态的记录
    if not all_segments:
        return _build_no_human_result(group, first_src)
    
    # 切片和合并
    total_seg_count = sum(len(segs) for _, segs in all_segments)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sliced_files = []
        slice_count = 0
        
        for src_file, segments in all_segments:
            for seg in segments:
                slice_name = f"slice_{len(sliced_files):04d}.mp4"
                slice_path = temp_path / slice_name
                
                success = slice_video(
                    src_file.abs_path,
                    str(slice_path),
                    seg["start"],
                    seg["end"]
                )
                
                if success:
                    sliced_files.append(str(slice_path))
                
                slice_count += 1
                if total_seg_count > progress_interval and slice_count % progress_interval == 0:
                    logger.info(f"{log_prefix}: 切片进度 {slice_count}/{total_seg_count}")
        
        # 切片失败，返回 no_human 记录
        if not sliced_files:
            return _build_no_human_result(group, first_src)
        
        # 生成输出文件名
        try:
            created_dt = datetime.fromtimestamp(first_src.created_timestamp)
            time_prefix = created_dt.strftime("%Y%m%d_%H%M%S")
        except Exception:
            time_prefix = "unknown"
        
        if len(group) > 1 or len(sliced_files) > 1:
            output_filename = f"{time_prefix}_p_{max_person_count}_merged.mp4"
        else:
            output_filename = f"{time_prefix}_p_{max_person_count}.mp4"
        
        output_file_path = output_dir / output_filename
        
        # 合并
        if len(sliced_files) == 1:
            shutil.copy2(sliced_files[0], output_file_path)
            success = True
        else:
            success = merge_videos(sliced_files, str(output_file_path))
        
        # 合并失败，返回 no_human 记录
        if not success:
            return _build_no_human_result(group, first_src)
        
        # 设置文件时间
        try:
            set_file_times(str(output_file_path), {
                'created_time': first_src.created_timestamp,
                'modified_time': first_src.created_timestamp,
                'accessed_time': first_src.created_timestamp
            })
        except Exception as e:
            logger.warning(f"设置文件时间失败: {e}")
        
        # 构建结果 - has_output 状态
        src_results = []
        for src_file, segments in all_segments:
            file_max_count = max((s["max_human_count"] for s in segments), default=0)
            src_results.append({
                "rel_path": src_file.rel_path,
                "created": src_file.created,
                "max_human_count": file_max_count,
                "segments": segments
            })
        
        return {
            "status": "has_output",
            "output_relative_filepath": f"merged/{output_filename}",
            "created": first_src.created,
            "src_files": src_results
        }


def probe_video_info(video_path: str) -> dict | None:
    """
    探测视频信息（使用 ffprobe）
    
    返回:
        fps, width, height, frame_count, duration, readable
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        
        # 查找视频流
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        # 解析 fps（可能是 "30/1" 或 "29.97" 格式）
        fps_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 0
        else:
            fps = float(fps_str)
        
        # 获取帧数
        frame_count = int(video_stream.get('nb_frames', 0))
        
        # 获取时长
        duration = float(data.get('format', {}).get('duration', 0))
        
        # 如果帧数为 0，从时长和 fps 估算
        if frame_count == 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)
        
        info = {
            'fps': fps,
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'frame_count': frame_count,
            'duration': duration,
            'readable': fps > 0 and duration > 0
        }
        
        return info
        
    except Exception as e:
        logger.warning(f"探测视频信息失败: {video_path}, 错误: {e}")
        return None


class FFmpegFrameReader:
    """
    使用 ffmpeg 高效读取视频帧（支持多线程解码）
    
    优势：
    - 使用 ffmpeg 的多线程解码能力
    - 通过 fps 滤镜按时间间隔选帧，比 select 滤镜快 ~10%
    - 可选降分辨率输出，减少数据传输量
    - 管道传输，内存效率高
    """
    
    def __init__(
        self, 
        video_path: str, 
        frame_interval: int,
        total_frames: int,
        width: int,
        height: int,
        fps: float,
        decode_threads: int = 4,
        scale_width: int | None = None  # 可选：缩放到指定宽度
    ):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.total_frames = total_frames
        self.fps = fps
        self.decode_threads = decode_threads
        
        # 处理缩放
        if scale_width and scale_width < width:
            scale_ratio = scale_width / width
            self.width = scale_width
            self.height = int(height * scale_ratio)
            # 确保高度是偶数
            self.height = self.height - (self.height % 2)
            self._scale_filter = f",scale={self.width}:{self.height}"
        else:
            self.width = width
            self.height = height
            self._scale_filter = ""
        
        self._proc = None
        self._frame_size = self.width * self.height * 3  # RGB24
        self._current_frame_idx = 0
        self._sample_interval = frame_interval / fps  # 采样间隔（秒）
        self._current_time = 0.0
    
    def start(self):
        """启动 ffmpeg 进程"""
        # 使用 fps 滤镜按时间间隔选帧（比 select 滤镜快 ~10%）
        # fps=1/interval 表示每 interval 秒输出一帧
        output_fps = 1.0 / self._sample_interval
        vf_filter = f"fps={output_fps}{self._scale_filter}"
        
        cmd = [
            "ffmpeg",
            "-threads", str(self.decode_threads),  # 解码线程数
            "-i", self.video_path,
            "-vf", vf_filter,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-loglevel", "error",
            "-"
        ]
        
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_size * 4  # 缓冲 4 帧
        )
        self._current_frame_idx = 0
        self._current_time = 0.0
    
    def read_frame(self) -> tuple[np.ndarray | None, float]:
        """
        读取下一帧
        
        Returns:
            (frame, frame_time) 或 (None, -1) 表示结束
        """
        if self._proc is None or self._proc.poll() is not None:
            return None, -1
        
        try:
            raw_data = self._proc.stdout.read(self._frame_size)
            
            if len(raw_data) != self._frame_size:
                return None, -1
            
            # 转换为 numpy 数组
            frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            
            # 计算当前帧时间（基于采样间隔）
            frame_time = self._current_time
            self._current_time += self._sample_interval
            self._current_frame_idx += self.frame_interval
            
            return frame, frame_time
            
        except Exception:
            return None, -1
    
    def read_batch(self, batch_size: int) -> tuple[list[np.ndarray], list[float]]:
        """
        批量读取帧
        
        Returns:
            (frames_list, times_list)
        """
        frames = []
        times = []
        
        for _ in range(batch_size):
            frame, frame_time = self.read_frame()
            if frame is None:
                break
            frames.append(frame)
            times.append(frame_time)
        
        return frames, times
    
    def close(self):
        """关闭 ffmpeg 进程"""
        if self._proc:
            try:
                self._proc.stdout.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def process_video_streaming(
    video_path: str,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    worker_id: int,
    group_idx: int,
    video_idx: int,
    sample_interval: float,
    batch_size: int,
    max_pending_requests: int = 8,
    decode_threads: int = 4,
    coarse_interval: float | None = None,  # 粗筛间隔（None 表示禁用两阶段）
    use_keyframe_seek: bool = True  # 精筛是否使用关键帧 seek
) -> tuple[list[dict], int]:
    """
    流式处理单个视频（两阶段检测优化版）
    
    两阶段检测策略：
    1. 粗筛阶段：使用较大的采样间隔（如 10-15s）+ 关键帧 seek 快速判断视频是否有人
       - 无人 → 直接返回，跳过整个视频（大幅节省解码时间）
       - 有人 → 进入精筛阶段
    2. 精筛阶段：使用智能关键帧采样（默认）或固定间隔模式
       - 智能关键帧模式：根据关键帧间隔自动选择最优策略
         * 间隔 2.5-4s: 直接使用所有关键帧（最快）
         * 间隔 1-2.5s: 跳帧采样（如每隔2个关键帧取1个）
         * 间隔 >4s: 混合模式（关键帧 + 中间插值点）
       - 固定间隔模式：使用 fps 滤镜按 sample_interval 采样（兼容旧逻辑）
    
    优化原理：
    - H.264/H.265 关键帧间隔通常为 1-3 秒，与 sample_interval 相近
    - 关键帧 seek 只需解码 I 帧本身，无需解码 P/B 帧，速度快 5-10 倍
    - 并行读取多个关键帧，充分利用 I/O 等待时间
    - 智能策略根据实际关键帧间隔动态调整，平衡速度和覆盖率
    
    Args:
        sample_interval: 精筛采样间隔（秒），用于策略计算和 segment buffer
        coarse_interval: 粗筛采样间隔（秒），None 表示禁用两阶段检测
        use_keyframe_seek: 精筛是否使用关键帧 seek（默认 True）
        max_pending_requests: 最大同时等待的请求数（预取深度）
        decode_threads: ffmpeg 解码线程数（仅在固定间隔模式下使用）
    
    Returns:
        (segments, max_person_count)
    """
    video_info = probe_video_info(video_path)
    
    if not video_info or not video_info.get('readable', False):
        return [], 0
    
    fps = video_info['fps']
    duration = video_info['duration']
    total_frames = video_info['frame_count']
    width = video_info['width']
    height = video_info['height']
    
    if fps <= 0 or fps > 240:
        fps = 30.0
    
    if duration <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        return [], 0
    
    # ========================================
    # 阶段1: 粗筛（快速判断视频是否有人）
    # ========================================
    if coarse_interval and coarse_interval > sample_interval and duration >= coarse_interval * 2:
        has_human = _coarse_scan_video(
            video_path=video_path,
            detection_queue=detection_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            group_idx=group_idx,
            video_idx=video_idx,
            coarse_interval=coarse_interval,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            decode_threads=decode_threads,
            batch_size=batch_size
        )
        
        if not has_human:
            # 粗筛未发现人物，跳过整个视频
            return [], 0
    
    # ========================================
    # 阶段2: 精筛（精确定位人物时间段）
    # ========================================
    if use_keyframe_seek:
        # 使用关键帧 seek 模式（更快）
        return _fine_scan_with_keyframes(
            video_path=video_path,
            detection_queue=detection_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            group_idx=group_idx,
            video_idx=video_idx,
            duration=duration,
            width=width,
            height=height,
            batch_size=batch_size,
            sample_interval=sample_interval  # 用于 segment buffer
        )
    else:
        # 使用固定间隔模式（兼容旧逻辑）
        return _fine_scan_with_fps_filter(
            video_path=video_path,
            detection_queue=detection_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            group_idx=group_idx,
            video_idx=video_idx,
            sample_interval=sample_interval,
            batch_size=batch_size,
            max_pending_requests=max_pending_requests,
            decode_threads=decode_threads,
            fps=fps,
            duration=duration,
            total_frames=total_frames,
            width=width,
            height=height
        )


def _get_keyframe_times(video_path: str) -> list[float]:
    """
    获取视频中所有关键帧（I帧）的时间点
    
    使用 ffprobe 提取关键帧信息，返回时间点列表
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "frame=pts_time,pict_type",
        "-of", "csv=p=0",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        
        keyframe_times = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2 and parts[1].strip() == 'I':
                try:
                    keyframe_times.append(float(parts[0]))
                except ValueError:
                    continue
        
        return keyframe_times
        
    except Exception:
        return []


def _analyze_keyframe_interval(keyframe_times: list[float]) -> tuple[float, float]:
    """
    分析关键帧间隔的统计特征
    
    Returns:
        (avg_interval, std_interval): 平均间隔和标准差
    """
    if len(keyframe_times) < 2:
        return 0.0, 0.0
    
    intervals = []
    for i in range(1, len(keyframe_times)):
        intervals.append(keyframe_times[i] - keyframe_times[i-1])
    
    avg_interval = sum(intervals) / len(intervals)
    
    if len(intervals) > 1:
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_interval = variance ** 0.5
    else:
        std_interval = 0.0
    
    return avg_interval, std_interval


def _select_keyframes_by_strategy(
    keyframe_times: list[float],
    avg_interval: float,
    target_interval: float
) -> tuple[list[float], str]:
    """
    根据关键帧间隔动态选择采样策略
    
    策略：
    1. 关键帧间隔 ~3s (2.5-4s): 直接使用所有关键帧
    2. 关键帧间隔 1-2.5s: 跳帧采样（每隔 N 个关键帧取一个）
    3. 关键帧间隔 >4s: 使用关键帧 + 中间插值混合模式
    
    Args:
        keyframe_times: 所有关键帧时间点
        avg_interval: 平均关键帧间隔
        target_interval: 目标采样间隔（通常 ~3s）
    
    Returns:
        (selected_times, strategy_name): 选中的时间点列表和策略名称
    """
    if not keyframe_times:
        return [], "empty"
    
    # 策略1: 关键帧间隔接近目标间隔 (2.5-4s)，直接使用所有关键帧
    if 2.5 <= avg_interval <= 4.0:
        return keyframe_times, "all_keyframes"
    
    # 策略2: 关键帧间隔较小 (< 2.5s)，跳帧采样
    if avg_interval < 2.5:
        # 计算跳帧步长，使采样间隔接近目标
        skip_step = max(1, round(target_interval / avg_interval))
        selected = keyframe_times[::skip_step]
        
        # 确保最后一帧也被采样（如果离最后一个选中的帧较远）
        if keyframe_times and selected:
            last_keyframe = keyframe_times[-1]
            last_selected = selected[-1]
            if last_keyframe - last_selected > target_interval * 0.5:
                selected.append(last_keyframe)
        
        return selected, f"skip_{skip_step}"
    
    # 策略3: 关键帧间隔较大 (> 4s)，混合模式
    # 保留所有关键帧，并在关键帧之间插入额外采样点
    selected = []
    for i, kf_time in enumerate(keyframe_times):
        selected.append(kf_time)
        
        # 在当前关键帧和下一个关键帧之间插入中间点
        if i < len(keyframe_times) - 1:
            next_kf_time = keyframe_times[i + 1]
            gap = next_kf_time - kf_time
            
            # 如果间隔太大，插入中间采样点
            if gap > target_interval * 1.5:
                num_inserts = int(gap / target_interval) - 1
                for j in range(1, num_inserts + 1):
                    insert_time = kf_time + j * (gap / (num_inserts + 1))
                    selected.append(insert_time)
    
    # 排序（因为插入的中间点可能打乱顺序）
    selected.sort()
    return selected, "hybrid"


def _fine_scan_with_keyframes(
    video_path: str,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    worker_id: int,
    group_idx: int,
    video_idx: int,
    duration: float,
    width: int,
    height: int,
    batch_size: int,
    sample_interval: float  # 用于 segment buffer 计算
) -> tuple[list[dict], int]:
    """
    使用关键帧 seek 进行精筛（智能采样策略）
    
    策略：
    1. 关键帧间隔 ~3s (2.5-4s): 直接使用所有关键帧，最快
    2. 关键帧间隔 1-2.5s: 跳帧采样（每隔 N 个关键帧取一个），避免过度采样
    3. 关键帧间隔 >4s: 混合模式，关键帧 + 中间插值，保证覆盖率
    
    优势：
    - 直接 seek 到每个关键帧（I帧），无需解码 P/B 帧
    - 并行读取多个关键帧，速度快 5-10 倍
    - 自适应关键帧间隔，平衡速度和覆盖率
    """
    # 获取所有关键帧时间点
    keyframe_times = _get_keyframe_times(video_path)
    
    if not keyframe_times:
        # 无法获取关键帧信息，返回空
        logger.warning(f"无法获取关键帧信息: {video_path}")
        return [], 0
    
    # 分析关键帧间隔
    avg_interval, std_interval = _analyze_keyframe_interval(keyframe_times)
    
    # 根据策略选择采样点
    selected_times, strategy = _select_keyframes_by_strategy(
        keyframe_times, avg_interval, sample_interval
    )
    
    if not selected_times:
        return [], 0
    
    # 计算实际的采样间隔（用于 segment buffer）
    if len(selected_times) >= 2:
        actual_interval = (selected_times[-1] - selected_times[0]) / (len(selected_times) - 1)
    else:
        actual_interval = sample_interval
    
    # 记录策略选择（调试用）
    logger.debug(
        f"精筛策略: {strategy}, 关键帧间隔: {avg_interval:.2f}s±{std_interval:.2f}s, "
        f"原始帧数: {len(keyframe_times)}, 采样帧数: {len(selected_times)}"
    )
    
    # 并行读取帧
    max_parallel = min(4, len(selected_times))
    results: dict[float, np.ndarray | None] = {}
    
    # 判断是否需要读取非关键帧（混合模式）
    is_hybrid = strategy == "hybrid"
    non_keyframe_times = [t for t in selected_times if t not in keyframe_times] if is_hybrid else []
    pure_keyframe_times = [t for t in selected_times if t in keyframe_times]
    
    def read_frame_wrapper(seek_time: float, is_keyframe: bool = True) -> tuple[float, np.ndarray | None]:
        if is_keyframe:
            frame = _read_keyframe_at_time(video_path, seek_time, width, height)
        else:
            # 非关键帧需要精确 seek
            frame = _read_frame_at_time_precise(video_path, seek_time, width, height)
        return (seek_time, frame)
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # 提交关键帧读取任务
        futures = [executor.submit(read_frame_wrapper, t, True) for t in pure_keyframe_times]
        # 提交非关键帧读取任务（混合模式）
        futures.extend([executor.submit(read_frame_wrapper, t, False) for t in non_keyframe_times])
        
        for future in futures:
            seek_time, frame = future.result()
            results[seek_time] = frame
    
    # 按时间顺序整理结果
    all_frames = []
    all_times = []
    for t in selected_times:  # 使用 selected_times 而不是 keyframe_times
        frame = results.get(t)
        if frame is not None:
            all_frames.append(frame)
            all_times.append(t)
    
    if not all_frames:
        return [], 0
    
    # 分批发送检测请求
    tracker = SegmentTracker(sample_interval=actual_interval, video_duration=duration)
    
    # 分批处理
    for i in range(0, len(all_frames), batch_size):
        batch_frames = all_frames[i:i+batch_size]
        batch_times = all_times[i:i+batch_size]
        
        request_id = f"w{worker_id}_g{group_idx}_v{video_idx}_fine_r{i}"
        detection_queue.put((request_id, worker_id, batch_frames, batch_times))
        
        # 等待结果
        try:
            response = response_queue.get(timeout=60.0)
            resp_id, frame_results = response
            
            if resp_id == request_id:
                for frame_time, human_count in frame_results:
                    tracker.update(frame_time, human_count)
        except Empty:
            logger.warning(f"精筛超时: {video_path}")
    
    tracker.finalize()
    return tracker.get_segments(), tracker.max_person_count


def _fine_scan_with_fps_filter(
    video_path: str,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    worker_id: int,
    group_idx: int,
    video_idx: int,
    sample_interval: float,
    batch_size: int,
    max_pending_requests: int,
    decode_threads: int,
    fps: float,
    duration: float,
    total_frames: int,
    width: int,
    height: int
) -> tuple[list[dict], int]:
    """
    使用 fps 滤镜进行精筛（兼容旧逻辑）
    
    适用于关键帧间隔不规律或需要精确控制采样间隔的场景
    """
    # 计算采样帧间隔
    frame_interval = int(fps * sample_interval)
    if frame_interval < 1:
        frame_interval = 1
    
    # 流式 segment 跟踪器
    tracker = SegmentTracker(sample_interval=sample_interval, video_duration=duration)
    
    request_counter = 0
    
    # 异步预取：跟踪已发送但未收到结果的请求
    pending_requests: dict[str, list[float]] = {}
    all_results: dict[str, list[tuple[float, int]]] = {}
    
    try:
        # 使用 ffmpeg 帧读取器
        with FFmpegFrameReader(
            video_path=video_path,
            frame_interval=frame_interval,
            total_frames=total_frames,
            width=width,
            height=height,
            fps=fps,
            decode_threads=decode_threads
        ) as reader:
            
            reading_done = False
            
            while not reading_done or pending_requests:
                # 1. 尽可能多地发送请求（预取）
                while not reading_done and len(pending_requests) < max_pending_requests:
                    frames_batch, times_batch = reader.read_batch(batch_size)
                    
                    if not frames_batch:
                        reading_done = True
                        break
                    
                    request_id = f"w{worker_id}_g{group_idx}_v{video_idx}_r{request_counter}"
                    request_counter += 1
                    
                    detection_queue.put((request_id, worker_id, frames_batch, times_batch))
                    pending_requests[request_id] = times_batch
                    frames_batch = None
                
                # 2. 非阻塞地接收结果
                if pending_requests:
                    try:
                        timeout = 0.1 if not reading_done else 30.0
                        response = response_queue.get(timeout=timeout)
                        resp_id, frame_results = response
                        
                        if resp_id in pending_requests:
                            del pending_requests[resp_id]
                            all_results[resp_id] = frame_results
                            
                    except Empty:
                        if reading_done and pending_requests:
                            continue
        
        # 3. 按时间顺序处理所有结果
        all_frame_results = []
        for req_id in sorted(all_results.keys(), key=lambda x: int(x.split('_r')[-1])):
            all_frame_results.extend(all_results[req_id])
        
        all_frame_results.sort(key=lambda x: x[0])
        
        for frame_time, human_count in all_frame_results:
            tracker.update(frame_time, human_count)
            
    except Exception as e:
        logger.warning(f"处理视频失败 {video_path}: {e}")
        return [], 0
    
    tracker.finalize()
    return tracker.get_segments(), tracker.max_person_count


def _read_keyframe_at_time(
    video_path: str,
    seek_time: float,
    width: int,
    height: int
) -> np.ndarray | None:
    """
    使用关键帧 seek 快速读取指定时间点的帧
    
    原理：
    - 使用 -ss 作为 input option（放在 -i 前面）实现关键帧级别跳转
    - FFmpeg 会 seek 到目标时间前最近的关键帧，只解码少量帧
    - 比完整解码视频流快 5-10 倍
    
    Args:
        video_path: 视频路径
        seek_time: 目标时间点（秒）
        width: 视频宽度
        height: 视频高度
    
    Returns:
        帧数据（numpy 数组）或 None
    """
    frame_size = width * height * 3
    
    cmd = [
        "ffmpeg",
        "-ss", str(seek_time),      # Input seek（关键帧跳转）
        "-i", video_path,
        "-vframes", "1",            # 只读取 1 帧
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "-"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10
        )
        
        if result.returncode != 0 or len(result.stdout) != frame_size:
            return None
        
        frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))
        return frame
        
    except Exception:
        return None


def _read_frame_at_time_precise(
    video_path: str,
    seek_time: float,
    width: int,
    height: int
) -> np.ndarray | None:
    """
    精确读取指定时间点的帧（用于混合模式中的非关键帧位置）
    
    原理：
    - 使用 -ss 作为 output option（放在 -i 后面）实现精确 seek
    - FFmpeg 会先 seek 到最近的关键帧，然后解码到目标位置
    - 比关键帧 seek 慢，但可以获取任意时间点的帧
    
    Args:
        video_path: 视频路径
        seek_time: 目标时间点（秒）
        width: 视频宽度
        height: 视频高度
    
    Returns:
        帧数据（numpy 数组）或 None
    """
    frame_size = width * height * 3
    
    # 使用两阶段 seek：先 input seek 到附近关键帧，再 output seek 精确定位
    # 这比纯 output seek 快，因为减少了需要解码的帧数
    pre_seek_time = max(0, seek_time - 5)  # 先跳到 5 秒前
    fine_seek_time = seek_time - pre_seek_time
    
    cmd = [
        "ffmpeg",
        "-ss", str(pre_seek_time),   # Input seek（粗跳）
        "-i", video_path,
        "-ss", str(fine_seek_time),  # Output seek（精确定位）
        "-vframes", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "-"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10
        )
        
        if result.returncode != 0 or len(result.stdout) != frame_size:
            return None
        
        frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))
        return frame
        
    except Exception:
        return None


def _coarse_scan_video(
    video_path: str,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    worker_id: int,
    group_idx: int,
    video_idx: int,
    coarse_interval: float,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    decode_threads: int,
    batch_size: int
) -> bool:
    """
    粗筛阶段：使用关键帧 seek 快速判断视频是否包含人物
    
    优化原理：
    - H.264/H.265 关键帧（I帧）间隔通常为 1-2 秒
    - 使用 -ss input seek 直接跳转到关键帧，避免解码中间的 P/B 帧
    - 并行读取多个时间点的帧，充分利用 I/O 等待时间
    - 相比完整解码流，速度提升 5-10 倍
    
    策略：
    - 使用较大的采样间隔（如 10-15s）快速扫描整个视频
    - 只要发现任意一帧有人，立即返回 True
    - 批量收集帧后一次性发送检测，减少队列开销
    
    Returns:
        True 如果视频中有人，False 如果没有人
    """
    # 计算需要采样的时间点
    duration = total_frames / fps if fps > 0 else 0
    if duration <= 0:
        return True  # 无法获取时长，保守返回 True
    
    sample_times = []
    t = 0.0
    while t < duration:
        sample_times.append(t)
        t += coarse_interval
    
    if not sample_times:
        return True
    
    # 并行使用关键帧 seek 读取各时间点的帧
    # 限制并发数，避免同时启动过多 FFmpeg 进程
    max_parallel = min(4, len(sample_times))
    results: dict[float, np.ndarray | None] = {}
    
    def read_frame_wrapper(seek_time: float) -> tuple[float, np.ndarray | None]:
        frame = _read_keyframe_at_time(video_path, seek_time, width, height)
        return (seek_time, frame)
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(read_frame_wrapper, t) for t in sample_times]
        for future in futures:
            seek_time, frame = future.result()
            results[seek_time] = frame
    
    # 按时间顺序整理结果
    all_frames = []
    all_times = []
    for t in sample_times:
        frame = results.get(t)
        if frame is not None:
            all_frames.append(frame)
            all_times.append(t)
    
    if not all_frames:
        return True  # 无法读取帧，保守返回 True
    
    # 发送检测请求
    request_id = f"w{worker_id}_g{group_idx}_v{video_idx}_coarse"
    detection_queue.put((request_id, worker_id, all_frames, all_times))
    
    # 等待结果
    try:
        response = response_queue.get(timeout=60.0)
        resp_id, frame_results = response
        
        if resp_id == request_id:
            # 检查是否有任意帧包含人物
            for frame_time, human_count in frame_results:
                if human_count > 0:
                    return True
            return False
            
    except Empty:
        # 超时，保守起见认为有人
        logger.warning(f"粗筛超时: {video_path}")
        return True
    
    return False


class SegmentTracker:
    """
    流式 Segment 追踪器
    
    边检测边标记人物出现的时间段，segment 边界预留 buffer
    """
    
    def __init__(self, sample_interval: float, video_duration: float):
        self.sample_interval = sample_interval
        self.video_duration = video_duration
        self.segments: list[dict] = []
        self.max_person_count = 0
        
        # 当前 segment 状态
        self._current_start: float | None = None
        self._current_end: float | None = None
        self._current_max_count: int = 0
    
    def update(self, frame_time: float, human_count: int):
        """更新检测结果"""
        has_human = human_count > 0
        
        if has_human:
            self.max_person_count = max(self.max_person_count, human_count)
            
            if self._current_start is None:
                # 从无人变成有人 → 新 segment 开始
                # start 时间向前预留 buffer
                self._current_start = max(0, frame_time - self.sample_interval)
                self._current_end = frame_time
                self._current_max_count = human_count
            else:
                # 持续有人 → 延长 segment
                self._current_end = frame_time
                self._current_max_count = max(self._current_max_count, human_count)
        else:
            if self._current_start is not None:
                # 从有人变成无人 → segment 结束
                # end 时间向后预留 buffer
                self._finalize_current_segment(frame_time)
    
    def _finalize_current_segment(self, end_frame_time: float | None = None):
        """完成当前 segment"""
        if self._current_start is None:
            return
        
        # end 时间向后预留 buffer
        if end_frame_time is not None:
            end_time = min(self.video_duration, end_frame_time + self.sample_interval)
        else:
            end_time = min(self.video_duration, self._current_end + self.sample_interval)
        
        start_time = self._current_start
        
        # 只有当 segment 有实际长度时才记录
        if end_time > start_time:
            self.segments.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "max_human_count": self._current_max_count
            })
        
        # 重置状态
        self._current_start = None
        self._current_end = None
        self._current_max_count = 0
    
    def finalize(self):
        """视频结束时调用，完成最后一个 segment"""
        if self._current_start is not None:
            self._finalize_current_segment()
    
    def get_segments(self) -> list[dict]:
        """获取所有 segments"""
        return self.segments


def slice_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float
) -> bool:
    """切片视频（静默模式）"""
    try:
        duration = end_time - start_time
        if duration <= 0:
            duration = 1.0
        
        video_info = probe_video_format(input_path)
        audio_codec = video_info.get('audio_codec', '') if video_info else ''
        
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",  # 静默模式，只显示错误
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "copy",
        ]
        
        if audio_codec in INCOMPATIBLE_AUDIO_CODECS:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            cmd.extend(["-c:a", "copy"])
        
        cmd.extend([
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            output_path
        ])
        
        subprocess.run(cmd, capture_output=True, text=True)
        
        output_file = Path(output_path)
        return output_file.exists() and output_file.stat().st_size > 0
        
    except Exception as e:
        logger.error(f"切片失败: {e}")
        return False


def merge_videos(
    input_files: list[str],
    output_path: str
) -> bool:
    """合并多个视频（静默模式）"""
    if len(input_files) < 2:
        return False
    
    list_file = Path(output_path).with_suffix('.txt')
    
    try:
        with open(list_file, 'w', encoding='utf-8') as f:
            for input_file in input_files:
                abs_path = str(Path(input_file).absolute())
                escaped_path = abs_path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",  # 静默模式，只显示错误
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, text=True)
        
        output_file = Path(output_path)
        return output_file.exists() and output_file.stat().st_size > 0
        
    except Exception as e:
        logger.error(f"合并失败: {e}")
        return False
    finally:
        if list_file.exists():
            list_file.unlink()


# ============================================================
# 主处理函数
# ============================================================

def run_pipeline(
    input_dir: str,
    output_dir: str,
    time_threshold: float = 300.0,
    num_workers: int | None = None,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.5,
    sample_interval: float = 3.0,
    scan_workers: int = 8,
    camera_brand: str = "xiaomi",
    batch_size: int | None = None,
    num_detectors: int | None = None,
    coarse_interval: float | None = 15.0,  # 粗筛间隔，None 禁用两阶段
    use_keyframe_seek: bool = True  # 精筛是否使用关键帧 seek
) -> dict:
    """
    运行增量处理 Pipeline（两阶段检测 + 关键帧 seek 优化）
    
    两阶段检测策略：
    1. 粗筛阶段：使用较大的采样间隔（如 15s）+ 关键帧 seek 快速判断视频是否有人
       - 无人 → 直接跳过整个视频，节省 80%+ 解码时间
       - 有人 → 进入精筛阶段
    2. 精筛阶段：使用关键帧 seek（默认）精确定位人物出现的时间段
       - 直接 seek 到每个 I 帧，比 fps 滤镜快 5-10 倍
       - H.264/H.265 关键帧间隔通常为 1-3 秒，与 sample_interval 相近
    
    处理流程：
    1. 加载日志，验证并清理失效记录
    2. 扫描文件，使用指定品牌解析器从文件名解析时间戳
    3. 启动多个 Detector Worker（并行推理，充分利用 CPU）
    4. 启动 I/O Workers（多进程处理读帧、切片、合并）
    5. 流式分组，边分组边入队
    6. 等待所有任务完成
    
    Args:
        input_dir: 输入视频目录
        output_dir: 输出结果目录
        time_threshold: 连续性判断时间阈值（秒）
        num_workers: I/O Worker 进程数量（None 表示自动根据 CPU 核心数配置）
        model_name: YOLO 模型名称
        conf_threshold: 检测置信度阈值
        sample_interval: 精筛采样间隔（秒），仅在 use_keyframe_seek=False 时使用
        scan_workers: 文件扫描并行数
        camera_brand: 监控摄像头品牌（xiaomi/hikvision/dahua/generic/auto）
        batch_size: I/O Worker 发送给 Detector 的批处理大小（None 表示自动配置）
        num_detectors: Detector Worker 进程数量（None 表示自动配置）
        coarse_interval: 粗筛采样间隔（秒），None 或 0 表示禁用两阶段检测
        use_keyframe_seek: 精筛是否使用关键帧 seek（默认 True，速度更快）
        
    Returns:
        处理统计结果
    """
    # 自动配置
    cpu_count = get_cpu_count()
    auto_config = get_auto_config(cpu_count)
    
    # 使用自动配置或用户指定的值
    if num_workers is None:
        num_workers = auto_config['num_workers']
    if num_detectors is None:
        num_detectors = auto_config['num_detectors']
    if batch_size is None:
        batch_size = auto_config['batch_size']
    decode_threads = auto_config['decode_threads']
    
    # 处理粗筛间隔
    if coarse_interval and coarse_interval <= sample_interval:
        coarse_interval = None  # 粗筛间隔必须大于精筛间隔才有意义
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 确保目录存在
    merged_dir = output_path / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志锁
    log_lock = mp.Lock()
    log_file = output_path / "log.jsonl"
    
    # 获取时间戳解析器
    timestamp_parser = get_parser(camera_brand)
    
    print("\n" + "=" * 70)
    print("🎬 监控视频人形检测增量处理 Pipeline (关键帧 seek 优化)")
    print("=" * 70)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {merged_dir}")
    print(f"日志文件: {log_file}")
    print(f"CPU 核心数: {cpu_count}")
    print(f"I/O Worker 进程数: {num_workers}")
    print(f"Detector Worker 进程数: {num_detectors}")
    print(f"批处理大小: {batch_size}")
    print(f"ffmpeg 解码线程数: {decode_threads}")
    if use_keyframe_seek:
        print(f"精筛模式: 关键帧 seek（自动使用视频 I 帧间隔）")
    else:
        print(f"精筛模式: fps 滤镜（采样间隔 {sample_interval}s）")
    if coarse_interval:
        print(f"粗筛采样间隔: {coarse_interval}s (两阶段检测已启用)")
    else:
        print(f"粗筛采样间隔: 禁用 (单阶段检测)")
    print(f"监控品牌: {timestamp_parser.brand} ({timestamp_parser.description})")
    print("=" * 70 + "\n")
    
    # 1. 加载并验证日志（清理失效记录）
    print("📋 步骤 1: 加载处理日志...")
    processing_log = ProcessingLog(log_file, output_path, log_lock)
    log_stats = processing_log.get_stats()
    print(f"   已处理的源文件: {log_stats['total_processed_sources']} 个")
    if log_stats['invalid_entries_removed'] > 0:
        print(f"   已清理失效记录: {log_stats['invalid_entries_removed']} 条")
    
    # 2. 扫描文件（使用指定品牌解析器从文件名解析时间戳）
    print(f"\n📂 步骤 2: 扫描源文件（使用 {timestamp_parser.brand} 解析器）...")
    scanner = FileScanner(
        input_dir=input_path,
        processed_sources=processing_log.get_processed_sources(),
        scan_workers=scan_workers,
        timestamp_parser=timestamp_parser
    )
    source_files = scanner.scan()
    print(f"   待处理的源文件: {len(source_files)} 个")
    
    if not source_files:
        print("\n✅ 所有文件已处理完成，无需重新处理")
        return {
            "total_source_files": log_stats['total_processed_sources'],
            "processed_files": 0,
            "skipped_files": log_stats['total_processed_sources'],
            "output_dir": str(merged_dir)
        }
    
    # 3. 创建队列（限制大小以控制内存使用）
    task_queue = mp.Queue()  # 任务队列
    # 检测队列大小根据 Detector 数量调整
    detection_queue = mp.Queue(maxsize=num_detectors * 8)  # 检测请求队列
    
    # 为每个 I/O Worker 创建响应队列
    manager = mp.Manager()
    response_queues = {i + 1: manager.Queue() for i in range(num_workers)}
    
    # 用于追踪完成的任务数
    result_counter = mp.Value('i', 0)
    counter_lock = mp.Lock()
    
    # Detector 停止事件
    stop_event = mp.Event()
    
    # 4. 启动多个 Detector Worker（并行推理）
    print(f"\n🧠 步骤 3: 启动 {num_detectors} 个 Detector Worker...")
    detector_processes = []
    for i in range(num_detectors):
        p = mp.Process(
            target=detector_worker,
            args=(
                detection_queue,
                response_queues,
                model_name,
                conf_threshold,
                stop_event,
                i + 1  # detector_id
            )
        )
        p.start()
        detector_processes.append(p)
    
    # 5. 启动 I/O Worker 进程
    print(f"\n🚀 步骤 4: 启动 {num_workers} 个 I/O Worker 进程...")
    io_workers = []
    for i in range(num_workers):
        worker_id = i + 1
        p = mp.Process(
            target=io_worker,
            args=(
                task_queue,
                worker_id,
                str(merged_dir),
                str(log_file),
                log_lock,
                detection_queue,
                response_queues[worker_id],
                sample_interval,
                batch_size,  # I/O Worker 控制 batch_size
                result_counter,
                counter_lock,
                decode_threads,  # ffmpeg 解码线程数
                coarse_interval,  # 粗筛间隔
                use_keyframe_seek  # 精筛是否使用关键帧 seek
            )
        )
        p.start()
        io_workers.append(p)
    
    # 6. 流式分组并推送任务
    print("\n📤 步骤 5: 流式分组并推送任务...")
    grouper = ConsecutiveGrouper(time_threshold=time_threshold)
    
    total_groups = 0
    total_files_in_groups = 0
    
    # 遍历源文件，流式分组，立即入队
    for group in grouper.group_streaming(source_files):
        total_groups += 1
        total_files_in_groups += len(group)
        
        # 转换为可序列化的 dict
        group_dicts = [source_file_to_dict(f) for f in group]
        task_queue.put((total_groups, group_dicts))
        print(f"   📦 组 {total_groups} 已入队 ({len(group)} 个文件)")
    
    print(f"\n   共 {total_groups} 组，{total_files_in_groups} 个文件")
    
    # 7. 发送终止信号给 I/O Workers
    print(f"\n📥 步骤 6: 等待处理完成...")
    for _ in range(num_workers):
        task_queue.put(None)
    
    # 8. 等待所有任务完成
    last_count = 0
    while True:
        with counter_lock:
            current_count = result_counter.value
        
        # 输出进度
        if current_count > last_count:
            print(f"   处理进度: {current_count}/{total_groups} ({current_count*100//total_groups}%)")
            last_count = current_count
        
        # 检查是否全部完成
        if current_count >= total_groups:
            break
        
        # 检查 Worker 是否还在运行
        alive_workers = sum(1 for w in io_workers if w.is_alive())
        if alive_workers == 0 and current_count < total_groups:
            logger.warning("所有 I/O Worker 已退出，但还有未完成的任务")
            break
        
        # 短暂等待
        time.sleep(0.5)
    
    # 9. 停止 Detector Workers
    print(f"\n⏳ 停止 {num_detectors} 个 Detector Worker...")
    detection_queue.put(None)  # 发送终止信号（会被传递给所有 Detector）
    stop_event.set()
    
    for i, p in enumerate(detector_processes):
        p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"Detector {i+1} 进程未能正常退出，强制终止")
            p.terminate()
            p.join(timeout=5)
    
    # 10. 等待所有 I/O Worker 退出
    print("⏳ 等待 I/O Worker 进程退出...")
    for w in io_workers:
        w.join(timeout=30)
        if w.is_alive():
            logger.warning(f"I/O Worker 进程 {w.pid} 未能正常退出，强制终止")
            w.terminate()
            w.join(timeout=5)
    
    # 11. 统计结果
    # 重新加载日志获取最新统计
    final_log = ProcessingLog(log_file, output_path, log_lock)
    final_stats = final_log.get_stats()
    
    print("\n" + "=" * 70)
    print("🎉 处理完成!")
    print("=" * 70)
    print(f"源文件总数: {final_stats['total_processed_sources']}")
    print(f"本次处理组数: {total_groups}")
    print(f"输出目录: {merged_dir}")
    print("=" * 70)
    
    return {
        "total_source_files": final_stats['total_processed_sources'],
        "processed_groups": total_groups,
        "output_dir": str(merged_dir)
    }


# ============================================================
# 命令行入口
# ============================================================

def main():
    import argparse
    
    # 获取支持的品牌列表
    brands_info = "\n".join([
        f"  - {p['brand']}: {p['description']}" 
        for p in list_parsers()
    ])
    
    parser = argparse.ArgumentParser(
        description="监控视频人形检测增量处理 Pipeline (关键帧 seek 优化)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 基本用法（自动检测 CPU 核心数并配置，默认启用关键帧 seek）
  python pipeline.py -i ./test-videos -o ./results
  
  # 指定监控品牌
  python pipeline.py -i ./videos -o ./output --brand hikvision
  python pipeline.py -i ./videos -o ./output --brand auto  # 自动检测
  
  # 禁用两阶段检测（直接精筛所有视频）
  python pipeline.py -i ./videos -o ./output --coarse-interval 0
  
  # 禁用关键帧 seek，使用固定采样间隔（fps 滤镜模式）
  python pipeline.py -i ./videos -o ./output --no-keyframe-seek --interval 2
  
  # 使用更大的模型提高检测精度
  python pipeline.py -i ./videos -o ./output --model yolov8s.pt

支持的监控品牌:
{brands_info}

两阶段检测 + 关键帧 seek 优化:
  1. 粗筛阶段 (默认 15s 间隔 + 关键帧 seek):
     - 使用 -ss input seek 直接跳转到关键帧，速度快 5-10 倍
     - 并行读取多个时间点的帧
     - 无人 → 直接跳过，节省 80%+ 解码时间
     - 有人 → 进入精筛阶段
  
  2. 精筛阶段 (关键帧 seek，默认启用):
     - 直接 seek 到每个 I 帧（通常 1-3 秒间隔）
     - H.264/H.265 关键帧间隔与采样间隔 (~3s) 接近
     - 比 fps 滤镜快 5-10 倍

自动配置表 (针对 FFmpeg 解码瓶颈优化):
  | CPU 核心数 | I/O Workers | Detectors | Batch Size | Decode Threads |
  |------------|-------------|-----------|------------|----------------|
  | 1-2 核     | 1           | 1         | 16         | 2              |
  | 3-4 核     | 2           | 2         | 32         | 2              |
  | 5-8 核     | 2           | 4         | 48         | 4              |
  | 9-16 核    | 3           | 6         | 64         | 4              |
  | 17+ 核     | 4           | 8         | 64         | 6              |

优化点:
  - 关键帧 seek: 直接跳转到 I 帧，避免解码 P/B 帧，速度快 5-10 倍
  - 两阶段检测: 粗筛快速排除无人视频，大幅减少解码量
  - 并行关键帧读取: 充分利用 I/O 等待时间
  - 多 Detector 并行: 多个 YOLO 模型实例并行推理
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入视频目录路径"
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="输出结果目录路径（默认: results）"
    )
    parser.add_argument(
        "--brand", "-b",
        default="xiaomi",
        help=f"监控摄像头品牌（默认: xiaomi）。支持: {', '.join(SUPPORTED_BRANDS)}"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=300.0,
        help="连续性判断时间阈值，秒（默认: 300，即 5 分钟）"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="I/O Worker 进程数量（默认: 自动根据 CPU 核心数配置）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每批发送给 Detector 的帧数（默认: 自动根据 CPU 核心数配置）"
    )
    parser.add_argument(
        "--detectors", "-d",
        type=int,
        default=None,
        help="Detector Worker 进程数量（默认: 自动根据 CPU 核心数配置）"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n.pt",
        help="YOLO 模型名称（默认: yolov8n.pt）"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="检测置信度阈值（默认: 0.5）"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="精筛采样间隔，秒（默认: 3.0）"
    )
    parser.add_argument(
        "--coarse-interval",
        type=float,
        default=15.0,
        help="粗筛采样间隔，秒（默认: 15.0，设为 0 禁用两阶段检测）"
    )
    parser.add_argument(
        "--no-keyframe-seek",
        action="store_true",
        help="禁用关键帧 seek，使用固定采样间隔（fps 滤镜模式）"
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=8,
        help="文件扫描并行线程数（默认: 8）"
    )
    
    args = parser.parse_args()
    
    # 处理粗筛间隔
    coarse_interval = args.coarse_interval if args.coarse_interval > 0 else None
    
    # 是否使用关键帧 seek
    use_keyframe_seek = not args.no_keyframe_seek
    
    # macOS/Linux 上需要使用 spawn 方式启动进程
    if os.name != 'nt':
        mp.set_start_method('spawn', force=True)
    
    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        time_threshold=args.threshold,
        num_workers=args.workers,
        model_name=args.model,
        conf_threshold=args.confidence,
        sample_interval=args.interval,
        scan_workers=args.scan_workers,
        camera_brand=args.brand,
        batch_size=args.batch_size,
        num_detectors=args.detectors,
        coarse_interval=coarse_interval,
        use_keyframe_seek=use_keyframe_seek
    )


if __name__ == "__main__":
    main()
