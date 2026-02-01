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

优化点：
- ffmpeg 多线程解码：替代 cv2.VideoCapture，解码性能提升 50-100%
- select 滤镜跳帧：直接在解码阶段过滤帧，无需逐帧 seek
- 多 Detector Worker：多个 YOLO 模型实例并行推理，充分利用多核 CPU
- 异步预取（深度 8）：I/O Worker 可同时发送多个 batch，不必等待结果
- 自动配置：根据 CPU 核心数自动分配 Workers、Detectors 和 Batch Size

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
    
    策略：
    - I/O Workers: 稍多于 CPU 核心数（I/O 密集型，可以略超配）
    - Detectors: 约等于 CPU 核心数的 2/3（CPU 密集型，留一些给 I/O）
    - Batch Size: 根据核心数调整（核心多时用更大 batch）
    - Decode Threads: 每个 ffmpeg 进程的解码线程数
    
    优化配置表（提升 CPU 利用率）：
    | CPU 核心数 | I/O Workers | Detectors | Batch Size | Decode Threads |
    |------------|-------------|-----------|------------|----------------|
    | 1-2 核     | 2           | 1         | 16         | 2              |
    | 3-4 核     | 4           | 2         | 32         | 2              |
    | 5-8 核     | 10          | 6         | 48         | 2              |
    | 9-16 核    | 16          | 10        | 64         | 2              |
    | 17+ 核     | 20          | 14        | 64         | 4              |
    
    Args:
        cpu_count: CPU 核心数（如果为 None，自动获取）
        
    Returns:
        dict: {'num_workers': int, 'num_detectors': int, 'batch_size': int, 'decode_threads': int}
    """
    if cpu_count is None:
        cpu_count = get_cpu_count()
    
    if cpu_count <= 2:
        return {
            'num_workers': 2,
            'num_detectors': 1,
            'batch_size': 16,
            'decode_threads': 2
        }
    elif cpu_count <= 4:
        return {
            'num_workers': 4,
            'num_detectors': 2,
            'batch_size': 32,
            'decode_threads': 2
        }
    elif cpu_count <= 8:
        return {
            'num_workers': 10,
            'num_detectors': 6,
            'batch_size': 48,
            'decode_threads': 2
        }
    elif cpu_count <= 16:
        return {
            'num_workers': 16,
            'num_detectors': 10,
            'batch_size': 64,
            'decode_threads': 2
        }
    else:
        return {
            'num_workers': 20,
            'num_detectors': 14,
            'batch_size': 64,
            'decode_threads': 4
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
    decode_threads: int = 2
):
    """
    I/O Worker - 负责视频读取、帧提取、切片、合并
    
    检测任务委托给 Detector Worker
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
            
            # 处理任务
            result = process_group_streaming(
                group=group,
                output_dir=output_path,
                detection_queue=detection_queue,
                response_queue=response_queue,
                worker_id=worker_id,
                group_idx=group_idx,
                sample_interval=sample_interval,
                batch_size=batch_size,
                decode_threads=decode_threads
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
    decode_threads: int = 2
) -> dict | None:
    """
    处理一组文件，流式检测人物片段
    
    返回格式：
    - status: "has_output" (有人物，生成了输出文件) 或 "no_human" (无人物，跳过)
    - output_relative_filepath: 输出文件相对路径（仅 has_output 状态有）
    - created: 组内第一个文件的创建时间
    - src_files: 源文件列表
    
    优化设计：
    1. 使用 ffmpeg 多线程解码，性能优于 cv2
    2. 边读帧边发送检测，边接收结果边标记 segment
    3. I/O Worker 控制 batch_size，Detector 直接处理
    4. segment 边界留出 sample_interval 的 buffer
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
    
    for idx, src_file in enumerate(group, 1):
        video_path = src_file.abs_path
        
        # 处理单个视频，流式检测
        segments, video_max_count = process_video_streaming(
            video_path=video_path,
            detection_queue=detection_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            group_idx=group_idx,
            video_idx=idx,
            sample_interval=sample_interval,
            batch_size=batch_size,
            decode_threads=decode_threads
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
    - 通过 select 滤镜直接跳帧，避免逐帧 seek
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
        decode_threads: int = 4
    ):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.decode_threads = decode_threads
        
        self._proc = None
        self._frame_size = width * height * 3  # RGB24
        self._current_frame_idx = 0
    
    def start(self):
        """启动 ffmpeg 进程"""
        # 使用 select 滤镜按间隔选取帧
        # mod(n, interval) == 0 表示每隔 interval 帧取一帧
        select_filter = f"select='not(mod(n\\,{self.frame_interval}))'"
        
        cmd = [
            "ffmpeg",
            "-threads", str(self.decode_threads),  # 解码线程数
            "-i", self.video_path,
            "-vf", select_filter,
            "-vsync", "vfr",  # 可变帧率输出（配合 select）
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
            
            # 计算当前帧时间
            frame_time = self._current_frame_idx / self.fps
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
    decode_threads: int = 4
) -> tuple[list[dict], int]:
    """
    流式处理单个视频（ffmpeg 多线程解码 + 异步预取优化版）
    
    优化：
    1. 使用 ffmpeg 多线程解码，性能优于 cv2.VideoCapture
    2. 通过 select 滤镜直接跳帧，无需逐帧 seek
    3. 异步预取：同时发送多个 batch，不必等待上一个结果
    4. segment 边界预留 sample_interval buffer
    
    Args:
        max_pending_requests: 最大同时等待的请求数（预取深度）
        decode_threads: ffmpeg 解码线程数
    
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
    
    # 计算采样帧间隔
    frame_interval = int(fps * sample_interval)
    if frame_interval < 1:
        frame_interval = 1
    
    # 流式 segment 跟踪器
    tracker = SegmentTracker(sample_interval=sample_interval, video_duration=duration)
    
    request_counter = 0
    
    # 异步预取：跟踪已发送但未收到结果的请求
    pending_requests: dict[str, list[float]] = {}  # request_id -> frame_times (for ordering)
    all_results: dict[str, list[tuple[float, int]]] = {}  # request_id -> results
    
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
                    # 读取一个 batch 的帧
                    frames_batch, times_batch = reader.read_batch(batch_size)
                    
                    if not frames_batch:
                        reading_done = True
                        break
                    
                    # 发送检测请求
                    request_id = f"w{worker_id}_g{group_idx}_v{video_idx}_r{request_counter}"
                    request_counter += 1
                    
                    detection_queue.put((request_id, worker_id, frames_batch, times_batch))
                    pending_requests[request_id] = times_batch
                    
                    # 释放帧引用
                    frames_batch = None
                
                # 2. 非阻塞地接收结果
                if pending_requests:
                    try:
                        # 使用短超时，保持流水线流动
                        timeout = 0.1 if not reading_done else 30.0
                        response = response_queue.get(timeout=timeout)
                        resp_id, frame_results = response
                        
                        if resp_id in pending_requests:
                            del pending_requests[resp_id]
                            # 存储结果
                            all_results[resp_id] = frame_results
                            
                    except Empty:
                        if reading_done and pending_requests:
                            # 读取已完成但还有未返回的请求，继续等待
                            continue
        
        # 3. 按时间顺序处理所有结果
        all_frame_results = []
        for req_id in sorted(all_results.keys(), key=lambda x: int(x.split('_r')[-1])):
            all_frame_results.extend(all_results[req_id])
        
        # 按时间排序
        all_frame_results.sort(key=lambda x: x[0])
        
        # 更新 tracker
        for frame_time, human_count in all_frame_results:
            tracker.update(frame_time, human_count)
            
    except Exception as e:
        logger.warning(f"处理视频失败 {video_path}: {e}")
        return [], 0
    
    # 结束追踪，获取最终 segments
    tracker.finalize()
    
    return tracker.get_segments(), tracker.max_person_count


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
    num_detectors: int | None = None
) -> dict:
    """
    运行增量处理 Pipeline（优化架构：多 Detector Worker 并行）
    
    流程：
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
        sample_interval: 采样间隔（秒）
        scan_workers: 文件扫描并行数
        camera_brand: 监控摄像头品牌（xiaomi/hikvision/dahua/generic/auto）
        batch_size: I/O Worker 发送给 Detector 的批处理大小（None 表示自动配置）
        num_detectors: Detector Worker 进程数量（None 表示自动配置）
        
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
    print("🎬 监控视频人形检测增量处理 Pipeline (多 Detector 并行)")
    print("=" * 70)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {merged_dir}")
    print(f"日志文件: {log_file}")
    print(f"CPU 核心数: {cpu_count}")
    print(f"I/O Worker 进程数: {num_workers}")
    print(f"Detector Worker 进程数: {num_detectors}")
    print(f"批处理大小: {batch_size}")
    print(f"ffmpeg 解码线程数: {decode_threads}")
    print(f"采样间隔: {sample_interval}s")
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
                decode_threads  # ffmpeg 解码线程数
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
        description="监控视频人形检测增量处理 Pipeline (多 Detector 并行)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 基本用法（自动检测 CPU 核心数并配置）
  python pipeline.py -i ./test-videos -o ./results
  
  # 指定监控品牌
  python pipeline.py -i ./videos -o ./output --brand hikvision
  python pipeline.py -i ./videos -o ./output --brand auto  # 自动检测
  
  # 手动指定配置（覆盖自动配置）
  python pipeline.py -i ./videos -o ./output --workers 8 --detectors 4 --batch-size 64
  
  # 使用更大的模型提高检测精度
  python pipeline.py -i ./videos -o ./output --model yolov8s.pt

支持的监控品牌:
{brands_info}

自动配置表 (根据 CPU 核心数，已优化 CPU 利用率):
  | CPU 核心数 | I/O Workers | Detectors | Batch Size | Decode Threads |
  |------------|-------------|-----------|------------|----------------|
  | 1-2 核     | 2           | 1         | 16         | 2              |
  | 3-4 核     | 4           | 2         | 32         | 2              |
  | 5-8 核     | 10          | 6         | 48         | 2              |
  | 9-16 核    | 16          | 10        | 64         | 2              |
  | 17+ 核     | 20          | 14        | 64         | 4              |

架构 (ffmpeg 多线程解码 + 多 Detector 并行):
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
  │  I/O Worker 1 │   │  I/O Worker 2 │   │  I/O Worker N │
  │  ffmpeg 解码  │   │  ffmpeg 解码  │   │  ffmpeg 解码  │
  │  (多线程)     │   │  (多线程)     │   │  (多线程)     │
  │  + 切片合并   │   │  + 切片合并   │   │  + 切片合并   │
  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │ (batch_size 帧)
                  ┌───────────▼───────────┐
                  │   Detection Queue     │
                  └───────────┬───────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  │  Detector 1   │   │  Detector 2   │   │  Detector M   │
  │  YOLO 推理    │   │  YOLO 推理    │   │  YOLO 推理    │
  └───────────────┘   └───────────────┘   └───────────────┘

优化点:
  - ffmpeg 多线程解码: 替代 cv2.VideoCapture，性能提升 50-100%
  - select 滤镜跳帧: 无需逐帧 seek，直接在解码阶段过滤
  - 多 Detector 并行: 多个 YOLO 模型实例并行推理，充分利用多核 CPU
  - 异步预取 (深度 8): I/O Worker 可同时发送多个 batch，不必等待上一个结果
  - 流式 Segment 检测: 边检测边标记，segment 边界预留 buffer
  - 队列限流: 检测队列有大小限制，防止内存溢出
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
        help="采样间隔，秒（默认: 3.0）"
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=8,
        help="文件扫描并行线程数（默认: 8）"
    )
    
    args = parser.parse_args()
    
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
        num_detectors=args.detectors
    )


if __name__ == "__main__":
    main()
