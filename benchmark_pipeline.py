"""
Pipeline 多进程性能分析

深入分析多进程架构中的瓶颈：
1. I/O Worker 等待 Detector 响应的时间
2. Detector 处理 batch 的时间
3. Queue 的排队延迟
4. 各进程的实际 CPU 利用情况
"""

import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import numpy as np
import argparse
import os
import json
from dataclasses import dataclass


def probe_video_info(video_path: str) -> dict | None:
    """获取视频信息"""
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
        
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        fps_str = video_stream.get('r_frame_rate', '30/1')
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
        except:
            fps = 30.0
        
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
        }
    except Exception as e:
        print(f"Error probing {video_path}: {e}")
        return None


def get_video_files(input_dir: str) -> list[Path]:
    """获取视频文件"""
    input_path = Path(input_dir)
    videos = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        videos.extend(input_path.rglob(f"*{ext}"))
    return sorted(videos)


def simulate_io_worker_timing(
    video_path: str,
    sample_interval: float,
    batch_size: int,
    decode_threads: int = 2
) -> dict:
    """
    模拟 I/O Worker 的时序，测量各阶段耗时
    """
    info = probe_video_info(video_path)
    if not info:
        return {'error': 'Cannot probe video'}
    
    fps = info['fps']
    width = info['width']
    height = info['height']
    duration = info['duration']
    total_frames = info['frame_count']
    
    frame_interval = max(1, int(fps * sample_interval))
    frame_size = width * height * 3
    expected_frames = total_frames // frame_interval
    
    # 启动 ffmpeg
    select_filter = f"select='not(mod(n\\,{frame_interval}))'"
    cmd = [
        "ffmpeg",
        "-threads", str(decode_threads),
        "-i", video_path,
        "-vf", select_filter,
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "-"
    ]
    
    timings = {
        'batch_read_times': [],  # 每个 batch 读取帧的时间
        'batch_sizes': [],       # 每个 batch 实际帧数
        'total_frames': 0,
    }
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_size * 4
    )
    
    batch_count = 0
    total_frames_read = 0
    
    while True:
        batch_start = time.perf_counter()
        frames_in_batch = 0
        
        for _ in range(batch_size):
            raw_data = proc.stdout.read(frame_size)
            if len(raw_data) != frame_size:
                break
            frames_in_batch += 1
        
        batch_end = time.perf_counter()
        
        if frames_in_batch == 0:
            break
        
        timings['batch_read_times'].append(batch_end - batch_start)
        timings['batch_sizes'].append(frames_in_batch)
        total_frames_read += frames_in_batch
        batch_count += 1
    
    proc.stdout.close()
    proc.wait()
    
    timings['total_frames'] = total_frames_read
    timings['batch_count'] = batch_count
    timings['video_duration'] = duration
    
    return timings


def detector_worker_simulation(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    stats_queue: mp.Queue,
    model_name: str,
    stop_event: mp.Event,
    detector_id: int
):
    """
    模拟 Detector Worker，记录处理时间
    """
    from ultralytics import YOLO
    
    model = YOLO(model_name)
    
    stats = {
        'batch_inference_times': [],
        'batch_sizes': [],
        'queue_wait_times': [],
        'total_processed': 0,
    }
    
    while not stop_event.is_set():
        wait_start = time.perf_counter()
        try:
            request = request_queue.get(timeout=0.5)
        except Empty:
            continue
        wait_end = time.perf_counter()
        
        if request is None:
            break
        
        request_id, frames, frame_times = request
        stats['queue_wait_times'].append(wait_end - wait_start)
        
        if frames:
            infer_start = time.perf_counter()
            results = model(frames, verbose=False, conf=0.5, classes=[0])
            infer_end = time.perf_counter()
            
            stats['batch_inference_times'].append(infer_end - infer_start)
            stats['batch_sizes'].append(len(frames))
            stats['total_processed'] += len(frames)
            
            frame_results = [(frame_times[i], len(results[i].boxes)) for i in range(len(results))]
            response_queue.put((request_id, frame_results))
        else:
            response_queue.put((request_id, []))
    
    stats_queue.put((detector_id, stats))


def io_worker_simulation(
    video_path: str,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    stats_queue: mp.Queue,
    worker_id: int,
    sample_interval: float,
    batch_size: int,
    max_pending: int = 8,
    decode_threads: int = 2
):
    """
    模拟 I/O Worker，记录各阶段时间
    """
    info = probe_video_info(video_path)
    if not info:
        stats_queue.put((worker_id, {'error': 'Cannot probe video'}))
        return
    
    fps = info['fps']
    width = info['width']
    height = info['height']
    frame_interval = max(1, int(fps * sample_interval))
    frame_size = width * height * 3
    
    stats = {
        'decode_times': [],      # 每个 batch 解码时间
        'wait_times': [],        # 等待响应时间
        'batch_sizes': [],
        'total_decode_time': 0,
        'total_wait_time': 0,
        'total_frames': 0,
    }
    
    select_filter = f"select='not(mod(n\\,{frame_interval}))'"
    cmd = [
        "ffmpeg",
        "-threads", str(decode_threads),
        "-i", video_path,
        "-vf", select_filter,
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "-"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_size * 4)
    
    pending_requests = {}
    request_counter = 0
    reading_done = False
    current_frame_idx = 0
    
    while not reading_done or pending_requests:
        # 发送请求
        while not reading_done and len(pending_requests) < max_pending:
            decode_start = time.perf_counter()
            
            frames = []
            frame_times = []
            for _ in range(batch_size):
                raw_data = proc.stdout.read(frame_size)
                if len(raw_data) != frame_size:
                    reading_done = True
                    break
                frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                frames.append(frame.copy())
                frame_time = current_frame_idx / fps
                frame_times.append(frame_time)
                current_frame_idx += frame_interval
            
            decode_end = time.perf_counter()
            
            if not frames:
                reading_done = True
                break
            
            decode_time = decode_end - decode_start
            stats['decode_times'].append(decode_time)
            stats['batch_sizes'].append(len(frames))
            stats['total_decode_time'] += decode_time
            stats['total_frames'] += len(frames)
            
            request_id = f"w{worker_id}_r{request_counter}"
            request_counter += 1
            
            detection_queue.put((request_id, frames, frame_times))
            pending_requests[request_id] = time.perf_counter()
        
        # 接收响应
        if pending_requests:
            try:
                timeout = 0.1 if not reading_done else 30.0
                wait_start = time.perf_counter()
                response = response_queue.get(timeout=timeout)
                resp_id, frame_results = response
                
                if resp_id in pending_requests:
                    wait_time = time.perf_counter() - pending_requests[resp_id]
                    stats['wait_times'].append(wait_time)
                    stats['total_wait_time'] += wait_time
                    del pending_requests[resp_id]
            except Empty:
                continue
    
    proc.stdout.close()
    proc.wait()
    
    stats_queue.put((worker_id, stats))


def run_pipeline_simulation(
    video_files: list[Path],
    num_workers: int,
    num_detectors: int,
    sample_interval: float,
    batch_size: int
):
    """
    运行模拟的 Pipeline，收集统计数据
    """
    print(f"\n模拟 Pipeline 配置:")
    print(f"  I/O Workers: {num_workers}")
    print(f"  Detectors: {num_detectors}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sample Interval: {sample_interval}s")
    print(f"  视频数量: {len(video_files)}")
    
    detection_queue = mp.Queue()
    response_queues = {i: mp.Queue() for i in range(num_workers)}
    stats_queue = mp.Queue()
    stop_event = mp.Event()
    
    # 启动 Detectors
    detectors = []
    for i in range(num_detectors):
        # 创建一个转发响应的包装
        p = mp.Process(
            target=detector_worker_wrapper,
            args=(detection_queue, response_queues, stats_queue, "yolov8n.pt", stop_event, i)
        )
        p.start()
        detectors.append(p)
    
    # 启动 I/O Workers
    workers = []
    for i, video in enumerate(video_files[:num_workers]):
        p = mp.Process(
            target=io_worker_simulation,
            args=(str(video), detection_queue, response_queues[i], stats_queue, i, sample_interval, batch_size)
        )
        p.start()
        workers.append(p)
    
    # 等待 Workers 完成
    for w in workers:
        w.join()
    
    # 停止 Detectors
    for _ in range(num_detectors):
        detection_queue.put(None)
    stop_event.set()
    
    for d in detectors:
        d.join(timeout=5)
    
    # 收集统计数据
    all_stats = {}
    while not stats_queue.empty():
        try:
            worker_id, stats = stats_queue.get_nowait()
            all_stats[worker_id] = stats
        except Empty:
            break
    
    return all_stats


def detector_worker_wrapper(
    detection_queue: mp.Queue,
    response_queues: dict,
    stats_queue: mp.Queue,
    model_name: str,
    stop_event: mp.Event,
    detector_id: int
):
    """
    Detector Worker 包装器
    """
    from ultralytics import YOLO
    
    model = YOLO(model_name)
    
    stats = {
        'batch_inference_times': [],
        'batch_sizes': [],
        'queue_wait_times': [],
        'total_processed': 0,
    }
    
    while not stop_event.is_set():
        wait_start = time.perf_counter()
        try:
            request = detection_queue.get(timeout=0.5)
        except Empty:
            continue
        wait_end = time.perf_counter()
        
        if request is None:
            detection_queue.put(None)  # 传递给其他 detector
            break
        
        request_id, worker_id, frames, frame_times = request
        stats['queue_wait_times'].append(wait_end - wait_start)
        
        if frames:
            infer_start = time.perf_counter()
            results = model(frames, verbose=False, conf=0.5, classes=[0])
            infer_end = time.perf_counter()
            
            stats['batch_inference_times'].append(infer_end - infer_start)
            stats['batch_sizes'].append(len(frames))
            stats['total_processed'] += len(frames)
            
            frame_results = [(frame_times[i], len(results[i].boxes)) for i in range(len(results))]
            if worker_id in response_queues:
                response_queues[worker_id].put((request_id, frame_results))
        else:
            if worker_id in response_queues:
                response_queues[worker_id].put((request_id, []))
    
    stats_queue.put((f"detector_{detector_id}", stats))


def analyze_single_video_timing(video_path: str, sample_interval: float, batch_size: int):
    """
    分析单个视频处理的时序
    """
    print(f"\n分析视频: {Path(video_path).name}")
    
    timings = simulate_io_worker_timing(video_path, sample_interval, batch_size)
    
    if 'error' in timings:
        print(f"  错误: {timings['error']}")
        return
    
    batch_read_times = timings['batch_read_times']
    batch_sizes = timings['batch_sizes']
    
    print(f"\n  视频时长: {timings['video_duration']:.1f}s")
    print(f"  总帧数: {timings['total_frames']}")
    print(f"  Batch 数: {timings['batch_count']}")
    
    if batch_read_times:
        avg_read_time = np.mean(batch_read_times)
        avg_batch_size = np.mean(batch_sizes)
        
        print(f"\n  每 Batch 平均读取时间: {avg_read_time*1000:.1f} ms")
        print(f"  每 Batch 平均帧数: {avg_batch_size:.1f}")
        print(f"  解码速率: {avg_batch_size / avg_read_time:.1f} fps")
        
        # 估算推理时间
        # 根据之前测试，YOLO 大约 17 fps = 58ms/frame
        estimated_infer_time = avg_batch_size * 0.058
        
        print(f"\n  📊 时间分配估算 (每 batch):")
        print(f"     解码时间: {avg_read_time*1000:.1f} ms")
        print(f"     推理时间 (估): {estimated_infer_time*1000:.1f} ms")
        print(f"     总时间: {(avg_read_time + estimated_infer_time)*1000:.1f} ms")
        
        if avg_read_time > estimated_infer_time:
            print(f"\n  ⚠️ 瓶颈: 解码 (占 {avg_read_time/(avg_read_time+estimated_infer_time)*100:.0f}%)")
        else:
            print(f"\n  ⚠️ 瓶颈: 推理 (占 {estimated_infer_time/(avg_read_time+estimated_infer_time)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Pipeline 多进程性能分析")
    parser.add_argument("input", help="输入视频目录")
    parser.add_argument("--interval", type=float, default=3.0, help="采样间隔（秒）")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch 大小")
    args = parser.parse_args()
    
    if os.name != 'nt':
        mp.set_start_method('spawn', force=True)
    
    videos = get_video_files(args.input)
    
    if not videos:
        print(f"未找到视频文件: {args.input}")
        return
    
    print(f"找到 {len(videos)} 个视频文件")
    
    # 1. 分析单视频时序
    print("\n" + "="*60)
    print("1. 单视频时序分析")
    print("="*60)
    
    for video in videos[:3]:
        analyze_single_video_timing(str(video), args.interval, args.batch_size)
    
    # 2. 分析多进程 Pipeline
    print("\n" + "="*60)
    print("2. 多进程 Pipeline 时序分析")
    print("="*60)
    
    # 使用 2 个 worker 和 2 个 detector 做简单测试
    print("\n运行简化的 Pipeline 模拟...")
    
    # 这里我们手动测试一下单个 worker + 单个 detector 的情况
    print("\n测试单 Worker + 单 Detector 配置...")
    
    detection_queue = mp.Queue()
    response_queue = mp.Queue()
    stats_queue = mp.Queue()
    stop_event = mp.Event()
    
    # 启动 Detector
    detector = mp.Process(
        target=simple_detector_worker,
        args=(detection_queue, response_queue, stats_queue, "yolov8n.pt", stop_event, 0)
    )
    detector.start()
    
    # 模拟 I/O Worker
    video = videos[0]
    worker_stats = simple_io_worker(
        str(video),
        detection_queue,
        response_queue,
        args.interval,
        args.batch_size
    )
    
    # 停止 Detector
    detection_queue.put(None)
    detector.join(timeout=10)
    
    # 获取 Detector 统计
    detector_stats = {}
    try:
        while not stats_queue.empty():
            det_id, stats = stats_queue.get_nowait()
            detector_stats[det_id] = stats
    except:
        pass
    
    # 分析结果
    print(f"\n📊 I/O Worker 统计:")
    print(f"   总帧数: {worker_stats['total_frames']}")
    print(f"   总解码时间: {worker_stats['total_decode_time']:.2f}s")
    print(f"   总等待响应时间: {worker_stats['total_wait_time']:.2f}s")
    print(f"   解码时间占比: {worker_stats['total_decode_time'] / (worker_stats['total_decode_time'] + worker_stats['total_wait_time']) * 100:.1f}%")
    print(f"   等待时间占比: {worker_stats['total_wait_time'] / (worker_stats['total_decode_time'] + worker_stats['total_wait_time']) * 100:.1f}%")
    
    if worker_stats['wait_times']:
        print(f"\n   每 Batch 等待时间:")
        print(f"     平均: {np.mean(worker_stats['wait_times'])*1000:.1f} ms")
        print(f"     最大: {np.max(worker_stats['wait_times'])*1000:.1f} ms")
        print(f"     最小: {np.min(worker_stats['wait_times'])*1000:.1f} ms")
    
    if 0 in detector_stats:
        det_stats = detector_stats[0]
        print(f"\n📊 Detector 统计:")
        print(f"   总处理帧数: {det_stats['total_processed']}")
        if det_stats['batch_inference_times']:
            print(f"   每 Batch 推理时间:")
            print(f"     平均: {np.mean(det_stats['batch_inference_times'])*1000:.1f} ms")
            print(f"     每帧: {np.mean(det_stats['batch_inference_times']) / np.mean(det_stats['batch_sizes']) * 1000:.1f} ms")
    
    print("\n" + "="*60)
    print("🔍 瓶颈总结")
    print("="*60)
    
    decode_ratio = worker_stats['total_decode_time'] / (worker_stats['total_decode_time'] + worker_stats['total_wait_time'])
    wait_ratio = worker_stats['total_wait_time'] / (worker_stats['total_decode_time'] + worker_stats['total_wait_time'])
    
    if decode_ratio > 0.6:
        print(f"\n⚠️  主要瓶颈: FFmpeg 解码 ({decode_ratio*100:.0f}%)")
        print("   原因: 解码 2304x1296 高分辨率视频需要大量 CPU")
        print("   建议:")
        print("   1. 减少 I/O Worker 数量，每个 Worker 分配更多 CPU 资源")
        print("   2. 增加 ffmpeg decode_threads")
        print("   3. 使用更低分辨率的视频源")
    elif wait_ratio > 0.6:
        print(f"\n⚠️  主要瓶颈: Detector 推理 ({wait_ratio*100:.0f}%)")
        print("   原因: YOLO 推理速度限制了整体吞吐")
        print("   建议:")
        print("   1. 增加 Detector Worker 数量")
        print("   2. 使用更快的模型 (如 yolov8n)")
        print("   3. 使用 GPU 加速")
    else:
        print(f"\n解码和推理时间相近:")
        print(f"   解码: {decode_ratio*100:.0f}%")
        print(f"   推理: {wait_ratio*100:.0f}%")
        print("   需要同时优化两个环节")


def simple_detector_worker(
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    stats_queue: mp.Queue,
    model_name: str,
    stop_event: mp.Event,
    detector_id: int
):
    """简化版 Detector Worker"""
    from ultralytics import YOLO
    
    model = YOLO(model_name)
    
    stats = {
        'batch_inference_times': [],
        'batch_sizes': [],
        'total_processed': 0,
    }
    
    while True:
        try:
            request = detection_queue.get(timeout=1.0)
        except Empty:
            if stop_event.is_set():
                break
            continue
        
        if request is None:
            break
        
        request_id, frames, frame_times = request
        
        if frames:
            infer_start = time.perf_counter()
            results = model(frames, verbose=False, conf=0.5, classes=[0])
            infer_end = time.perf_counter()
            
            stats['batch_inference_times'].append(infer_end - infer_start)
            stats['batch_sizes'].append(len(frames))
            stats['total_processed'] += len(frames)
            
            frame_results = [(frame_times[i], len(results[i].boxes)) for i in range(len(results))]
            response_queue.put((request_id, frame_results))
        else:
            response_queue.put((request_id, []))
    
    stats_queue.put((detector_id, stats))


def simple_io_worker(
    video_path: str,
    detection_queue: mp.Queue,
    response_queue: mp.Queue,
    sample_interval: float,
    batch_size: int,
    decode_threads: int = 2
) -> dict:
    """简化版 I/O Worker，返回统计数据"""
    info = probe_video_info(video_path)
    if not info:
        return {'error': 'Cannot probe video'}
    
    fps = info['fps']
    width = info['width']
    height = info['height']
    frame_interval = max(1, int(fps * sample_interval))
    frame_size = width * height * 3
    
    stats = {
        'decode_times': [],
        'wait_times': [],
        'batch_sizes': [],
        'total_decode_time': 0,
        'total_wait_time': 0,
        'total_frames': 0,
    }
    
    select_filter = f"select='not(mod(n\\,{frame_interval}))'"
    cmd = [
        "ffmpeg",
        "-threads", str(decode_threads),
        "-i", video_path,
        "-vf", select_filter,
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "-"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_size * 4)
    
    pending_requests = {}
    request_counter = 0
    reading_done = False
    current_frame_idx = 0
    max_pending = 8
    
    while not reading_done or pending_requests:
        # 发送请求（预取）
        while not reading_done and len(pending_requests) < max_pending:
            decode_start = time.perf_counter()
            
            frames = []
            frame_times = []
            for _ in range(batch_size):
                raw_data = proc.stdout.read(frame_size)
                if len(raw_data) != frame_size:
                    reading_done = True
                    break
                frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                frames.append(frame.copy())
                frame_time = current_frame_idx / fps
                frame_times.append(frame_time)
                current_frame_idx += frame_interval
            
            decode_end = time.perf_counter()
            
            if not frames:
                reading_done = True
                break
            
            decode_time = decode_end - decode_start
            stats['decode_times'].append(decode_time)
            stats['batch_sizes'].append(len(frames))
            stats['total_decode_time'] += decode_time
            stats['total_frames'] += len(frames)
            
            request_id = f"r{request_counter}"
            request_counter += 1
            
            detection_queue.put((request_id, frames, frame_times))
            pending_requests[request_id] = time.perf_counter()
        
        # 接收响应
        if pending_requests:
            try:
                timeout = 0.1 if not reading_done else 30.0
                response = response_queue.get(timeout=timeout)
                resp_id, _ = response
                
                if resp_id in pending_requests:
                    wait_time = time.perf_counter() - pending_requests[resp_id]
                    stats['wait_times'].append(wait_time)
                    stats['total_wait_time'] += wait_time
                    del pending_requests[resp_id]
            except Empty:
                continue
    
    proc.stdout.close()
    proc.wait()
    
    return stats


if __name__ == "__main__":
    main()
