#!/usr/bin/env python3
"""
智能关键帧采样策略 Benchmark

对比测试不同采样方式在有人视频中的性能：
1. 固定间隔模式（fps 滤镜）- 基准
2. 关键帧 seek（所有关键帧）
3. 智能策略（根据关键帧间隔动态选择）

主要测量：
- 帧读取总耗时
- 检测处理总耗时
- 端到端延迟
"""

import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Benchmark 结果"""
    method: str
    video_path: str
    duration: float
    keyframe_count: int
    sampled_frames: int
    avg_keyframe_interval: float
    strategy: str
    
    # 耗时（秒）
    frame_read_time: float
    detection_time: float
    total_time: float
    
    # 检测结果
    frames_with_human: int
    max_person_count: int


def probe_video_info(video_path: str) -> dict | None:
    """获取视频信息"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        duration = float(data.get('format', {}).get('duration', 0))
        frame_count = int(video_stream.get('nb_frames', 0))
        if frame_count == 0 and duration > 0:
            frame_count = int(duration * fps)
        
        return {
            'fps': fps,
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'frame_count': frame_count,
            'duration': duration,
            'readable': True
        }
    except Exception:
        return None


def get_keyframe_times(video_path: str) -> list[float]:
    """获取所有关键帧时间点"""
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time,pict_type", "-of", "csv=p=0",
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


def analyze_keyframe_interval(keyframe_times: list[float]) -> tuple[float, float]:
    """分析关键帧间隔"""
    if len(keyframe_times) < 2:
        return 0.0, 0.0
    intervals = [keyframe_times[i] - keyframe_times[i-1] for i in range(1, len(keyframe_times))]
    avg = sum(intervals) / len(intervals)
    std = (sum((x - avg) ** 2 for x in intervals) / len(intervals)) ** 0.5 if len(intervals) > 1 else 0.0
    return avg, std


def select_keyframes_by_strategy(keyframe_times: list[float], avg_interval: float, target_interval: float) -> tuple[list[float], str]:
    """智能选择采样策略"""
    if not keyframe_times:
        return [], "empty"
    
    # 策略1: 间隔接近目标 (2.5-4s)
    if 2.5 <= avg_interval <= 4.0:
        return keyframe_times, "all_keyframes"
    
    # 策略2: 间隔较小 (< 2.5s)，跳帧
    if avg_interval < 2.5:
        skip_step = max(1, round(target_interval / avg_interval))
        selected = keyframe_times[::skip_step]
        if keyframe_times and selected:
            if keyframe_times[-1] - selected[-1] > target_interval * 0.5:
                selected.append(keyframe_times[-1])
        return selected, f"skip_{skip_step}"
    
    # 策略3: 间隔较大 (> 4s)，混合模式
    selected = []
    for i, kf_time in enumerate(keyframe_times):
        selected.append(kf_time)
        if i < len(keyframe_times) - 1:
            gap = keyframe_times[i + 1] - kf_time
            if gap > target_interval * 1.5:
                num_inserts = int(gap / target_interval) - 1
                for j in range(1, num_inserts + 1):
                    selected.append(kf_time + j * (gap / (num_inserts + 1)))
    selected.sort()
    return selected, "hybrid"


def read_keyframe_at_time(video_path: str, seek_time: float, width: int, height: int) -> np.ndarray | None:
    """关键帧 seek 读取"""
    frame_size = width * height * 3
    cmd = [
        "ffmpeg", "-ss", str(seek_time), "-i", video_path,
        "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-loglevel", "error", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0 or len(result.stdout) != frame_size:
            return None
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))
    except Exception:
        return None


def read_frame_precise(video_path: str, seek_time: float, width: int, height: int) -> np.ndarray | None:
    """精确时间点读取（用于混合模式）"""
    frame_size = width * height * 3
    pre_seek = max(0, seek_time - 5)
    fine_seek = seek_time - pre_seek
    cmd = [
        "ffmpeg", "-ss", str(pre_seek), "-i", video_path,
        "-ss", str(fine_seek), "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-loglevel", "error", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0 or len(result.stdout) != frame_size:
            return None
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))
    except Exception:
        return None


def read_frames_fps_filter(video_path: str, sample_interval: float, fps: float, 
                           width: int, height: int, total_frames: int) -> list[tuple[float, np.ndarray]]:
    """使用 fps 滤镜读取固定间隔帧"""
    frame_interval = max(1, int(fps * sample_interval))
    output_fps = 1.0 / sample_interval
    frame_size = width * height * 3
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={output_fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-loglevel", "error", "-"
    ]
    
    frames = []
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            return []
        
        data = result.stdout
        num_frames = len(data) // frame_size
        
        for i in range(num_frames):
            frame_data = data[i*frame_size:(i+1)*frame_size]
            if len(frame_data) == frame_size:
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                frame_time = i * sample_interval
                frames.append((frame_time, frame))
    except Exception:
        pass
    
    return frames


def detector_process(detection_queue: mp.Queue, result_queue: mp.Queue, stop_event):
    """检测器进程"""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    
    while not stop_event.is_set():
        try:
            item = detection_queue.get(timeout=0.5)
            if item is None:
                break
            
            request_id, frames, times = item
            results = []
            
            # 批量检测
            if frames:
                preds = model(frames, verbose=False, classes=[0])
                for i, pred in enumerate(preds):
                    human_count = len([b for b in pred.boxes if b.cls == 0])
                    results.append((times[i], human_count))
            
            result_queue.put((request_id, results))
            
        except Empty:
            continue
        except Exception as e:
            print(f"Detector error: {e}")
            continue


def benchmark_fps_filter(video_path: str, sample_interval: float, batch_size: int,
                         detection_queue: mp.Queue, result_queue: mp.Queue) -> BenchmarkResult:
    """测试固定间隔模式（fps 滤镜）"""
    info = probe_video_info(video_path)
    if not info:
        return None
    
    keyframe_times = get_keyframe_times(video_path)
    avg_interval, _ = analyze_keyframe_interval(keyframe_times)
    
    # 读取帧
    t_start = time.perf_counter()
    frames_data = read_frames_fps_filter(
        video_path, sample_interval, info['fps'],
        info['width'], info['height'], info['frame_count']
    )
    t_read = time.perf_counter() - t_start
    
    if not frames_data:
        return None
    
    # 检测
    t_detect_start = time.perf_counter()
    frames_with_human = 0
    max_person = 0
    
    for i in range(0, len(frames_data), batch_size):
        batch = frames_data[i:i+batch_size]
        frames = [f[1] for f in batch]
        times = [f[0] for f in batch]
        
        request_id = f"fps_{i}"
        detection_queue.put((request_id, frames, times))
        
        try:
            resp_id, results = result_queue.get(timeout=60)
            for _, count in results:
                if count > 0:
                    frames_with_human += 1
                max_person = max(max_person, count)
        except Empty:
            pass
    
    t_detect = time.perf_counter() - t_detect_start
    t_total = time.perf_counter() - t_start
    
    return BenchmarkResult(
        method="fps_filter",
        video_path=str(video_path),
        duration=info['duration'],
        keyframe_count=len(keyframe_times),
        sampled_frames=len(frames_data),
        avg_keyframe_interval=avg_interval,
        strategy="fixed_interval",
        frame_read_time=t_read,
        detection_time=t_detect,
        total_time=t_total,
        frames_with_human=frames_with_human,
        max_person_count=max_person
    )


def benchmark_keyframe_all(video_path: str, sample_interval: float, batch_size: int,
                           detection_queue: mp.Queue, result_queue: mp.Queue) -> BenchmarkResult:
    """测试关键帧 seek（所有关键帧）"""
    info = probe_video_info(video_path)
    if not info:
        return None
    
    keyframe_times = get_keyframe_times(video_path)
    if not keyframe_times:
        return None
    
    avg_interval, _ = analyze_keyframe_interval(keyframe_times)
    
    # 并行读取所有关键帧
    t_start = time.perf_counter()
    
    frames_data = []
    max_parallel = min(4, len(keyframe_times))
    
    def read_wrapper(t):
        frame = read_keyframe_at_time(video_path, t, info['width'], info['height'])
        return (t, frame)
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(read_wrapper, t) for t in keyframe_times]
        for future in futures:
            t, frame = future.result()
            if frame is not None:
                frames_data.append((t, frame))
    
    frames_data.sort(key=lambda x: x[0])
    t_read = time.perf_counter() - t_start
    
    if not frames_data:
        return None
    
    # 检测
    t_detect_start = time.perf_counter()
    frames_with_human = 0
    max_person = 0
    
    for i in range(0, len(frames_data), batch_size):
        batch = frames_data[i:i+batch_size]
        frames = [f[1] for f in batch]
        times = [f[0] for f in batch]
        
        request_id = f"kf_all_{i}"
        detection_queue.put((request_id, frames, times))
        
        try:
            resp_id, results = result_queue.get(timeout=60)
            for _, count in results:
                if count > 0:
                    frames_with_human += 1
                max_person = max(max_person, count)
        except Empty:
            pass
    
    t_detect = time.perf_counter() - t_detect_start
    t_total = time.perf_counter() - t_start
    
    return BenchmarkResult(
        method="keyframe_all",
        video_path=str(video_path),
        duration=info['duration'],
        keyframe_count=len(keyframe_times),
        sampled_frames=len(frames_data),
        avg_keyframe_interval=avg_interval,
        strategy="all_keyframes",
        frame_read_time=t_read,
        detection_time=t_detect,
        total_time=t_total,
        frames_with_human=frames_with_human,
        max_person_count=max_person
    )


def benchmark_smart_strategy(video_path: str, sample_interval: float, batch_size: int,
                             detection_queue: mp.Queue, result_queue: mp.Queue) -> BenchmarkResult:
    """测试智能策略"""
    info = probe_video_info(video_path)
    if not info:
        return None
    
    keyframe_times = get_keyframe_times(video_path)
    if not keyframe_times:
        return None
    
    avg_interval, _ = analyze_keyframe_interval(keyframe_times)
    selected_times, strategy = select_keyframes_by_strategy(keyframe_times, avg_interval, sample_interval)
    
    if not selected_times:
        return None
    
    # 区分关键帧和非关键帧
    keyframe_set = set(keyframe_times)
    pure_keyframe_times = [t for t in selected_times if t in keyframe_set]
    non_keyframe_times = [t for t in selected_times if t not in keyframe_set]
    
    # 并行读取
    t_start = time.perf_counter()
    
    frames_data = []
    max_parallel = min(4, len(selected_times))
    
    def read_wrapper(t, is_keyframe):
        if is_keyframe:
            frame = read_keyframe_at_time(video_path, t, info['width'], info['height'])
        else:
            frame = read_frame_precise(video_path, t, info['width'], info['height'])
        return (t, frame)
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = []
        futures.extend([executor.submit(read_wrapper, t, True) for t in pure_keyframe_times])
        futures.extend([executor.submit(read_wrapper, t, False) for t in non_keyframe_times])
        
        for future in futures:
            t, frame = future.result()
            if frame is not None:
                frames_data.append((t, frame))
    
    frames_data.sort(key=lambda x: x[0])
    t_read = time.perf_counter() - t_start
    
    if not frames_data:
        return None
    
    # 检测
    t_detect_start = time.perf_counter()
    frames_with_human = 0
    max_person = 0
    
    for i in range(0, len(frames_data), batch_size):
        batch = frames_data[i:i+batch_size]
        frames = [f[1] for f in batch]
        times = [f[0] for f in batch]
        
        request_id = f"smart_{i}"
        detection_queue.put((request_id, frames, times))
        
        try:
            resp_id, results = result_queue.get(timeout=60)
            for _, count in results:
                if count > 0:
                    frames_with_human += 1
                max_person = max(max_person, count)
        except Empty:
            pass
    
    t_detect = time.perf_counter() - t_detect_start
    t_total = time.perf_counter() - t_start
    
    return BenchmarkResult(
        method="smart_strategy",
        video_path=str(video_path),
        duration=info['duration'],
        keyframe_count=len(keyframe_times),
        sampled_frames=len(frames_data),
        avg_keyframe_interval=avg_interval,
        strategy=strategy,
        frame_read_time=t_read,
        detection_time=t_detect,
        total_time=t_total,
        frames_with_human=frames_with_human,
        max_person_count=max_person
    )


def find_videos_with_human(video_dir: str, max_videos: int = 5) -> list[str]:
    """找到有人的视频（用于测试）"""
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(video_dir).rglob(f"*{ext}"))
    return [str(v) for v in sorted(video_files)[:max_videos]]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="智能关键帧采样策略 Benchmark")
    parser.add_argument("--video-dir", default="test-videos", help="视频目录")
    parser.add_argument("--sample-interval", type=float, default=3.0, help="目标采样间隔")
    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小")
    parser.add_argument("--max-videos", type=int, default=5, help="最大测试视频数")
    args = parser.parse_args()
    
    print("=" * 80)
    print("智能关键帧采样策略 Benchmark")
    print("=" * 80)
    print(f"目标采样间隔: {args.sample_interval}s")
    print(f"批处理大小: {args.batch_size}")
    print()
    
    # 启动检测器进程
    detection_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()
    
    detector = mp.Process(target=detector_process, args=(detection_queue, result_queue, stop_event))
    detector.start()
    
    # 预热
    print("预热检测器...")
    time.sleep(2)
    
    # 获取测试视频
    videos = find_videos_with_human(args.video_dir, args.max_videos)
    print(f"找到 {len(videos)} 个测试视频\n")
    
    all_results = []
    
    for video_path in videos:
        print(f"\n{'='*60}")
        print(f"测试视频: {Path(video_path).name}")
        print("=" * 60)
        
        # 获取视频基本信息
        info = probe_video_info(video_path)
        keyframe_times = get_keyframe_times(video_path)
        avg_interval, std_interval = analyze_keyframe_interval(keyframe_times)
        
        print(f"时长: {info['duration']:.1f}s, 关键帧: {len(keyframe_times)}, 间隔: {avg_interval:.2f}s ± {std_interval:.2f}s")
        print()
        
        # 测试三种方法
        methods = [
            ("固定间隔(fps滤镜)", benchmark_fps_filter),
            ("关键帧seek(全部)", benchmark_keyframe_all),
            ("智能策略", benchmark_smart_strategy),
        ]
        
        results = {}
        for method_name, benchmark_func in methods:
            print(f"测试 {method_name}...", end=" ", flush=True)
            result = benchmark_func(video_path, args.sample_interval, args.batch_size,
                                   detection_queue, result_queue)
            if result:
                results[method_name] = result
                print(f"完成 ({result.total_time:.2f}s)")
            else:
                print("失败")
        
        # 输出对比结果
        if results:
            print(f"\n{'方法':<20} {'采样帧':>8} {'策略':<15} {'读取':>8} {'检测':>8} {'总耗时':>8} {'有人帧':>8}")
            print("-" * 90)
            
            baseline = results.get("固定间隔(fps滤镜)")
            
            for method_name, result in results.items():
                speedup = ""
                if baseline and method_name != "固定间隔(fps滤镜)":
                    speedup = f" ({baseline.total_time / result.total_time:.1f}x)"
                
                print(f"{method_name:<20} {result.sampled_frames:>8} {result.strategy:<15} "
                      f"{result.frame_read_time:>7.2f}s {result.detection_time:>7.2f}s "
                      f"{result.total_time:>7.2f}s{speedup} {result.frames_with_human:>8}")
            
            all_results.append(results)
    
    # 总结
    if all_results:
        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        
        # 计算平均加速比
        fps_times = []
        kf_all_times = []
        smart_times = []
        
        for results in all_results:
            if "固定间隔(fps滤镜)" in results:
                fps_times.append(results["固定间隔(fps滤镜)"].total_time)
            if "关键帧seek(全部)" in results:
                kf_all_times.append(results["关键帧seek(全部)"].total_time)
            if "智能策略" in results:
                smart_times.append(results["智能策略"].total_time)
        
        if fps_times and kf_all_times:
            avg_speedup_kf = sum(fps_times) / sum(kf_all_times)
            print(f"关键帧seek vs fps滤镜: 平均加速 {avg_speedup_kf:.2f}x")
        
        if fps_times and smart_times:
            avg_speedup_smart = sum(fps_times) / sum(smart_times)
            print(f"智能策略 vs fps滤镜: 平均加速 {avg_speedup_smart:.2f}x")
        
        if kf_all_times and smart_times:
            avg_diff = sum(kf_all_times) / sum(smart_times)
            print(f"智能策略 vs 关键帧seek(全部): {avg_diff:.2f}x")
    
    # 停止检测器
    stop_event.set()
    detection_queue.put(None)
    detector.join(timeout=5)
    if detector.is_alive():
        detector.terminate()
    
    print("\nBenchmark 完成!")


if __name__ == "__main__":
    main()
