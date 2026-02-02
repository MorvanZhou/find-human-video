"""
性能瓶颈测试脚本

分别测试各个环节的耗时：
1. FFmpeg 解码 + 跳帧读取
2. 帧数据传输（Queue）
3. YOLO 模型推理
4. 视频切片
5. 视频合并

使用方法:
    uv run python benchmark.py ./test-videos
"""

import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import numpy as np
import argparse
import os
import sys


def get_video_files(input_dir: str, extensions: set = {'.mp4', '.avi', '.mov', '.mkv'}) -> list[Path]:
    """递归获取视频文件"""
    input_path = Path(input_dir)
    videos = []
    for ext in extensions:
        videos.extend(input_path.rglob(f"*{ext}"))
    return sorted(videos)[:5]  # 只取前 5 个测试


def probe_video_info(video_path: str) -> dict | None:
    """获取视频信息"""
    import json
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


def benchmark_ffmpeg_decode(video_path: str, sample_interval: float = 3.0, decode_threads: int = 2) -> dict:
    """测试 FFmpeg 解码速度"""
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
    
    start_time = time.perf_counter()
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_size * 4
    )
    
    frames_read = 0
    frames_data = []
    
    while True:
        raw_data = proc.stdout.read(frame_size)
        if len(raw_data) != frame_size:
            break
        
        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        frames_data.append(frame)
        frames_read += 1
    
    proc.stdout.close()
    proc.wait()
    
    elapsed = time.perf_counter() - start_time
    
    expected_frames = total_frames // frame_interval
    
    return {
        'video_duration': duration,
        'expected_frames': expected_frames,
        'frames_read': frames_read,
        'decode_time': elapsed,
        'fps_decode': frames_read / elapsed if elapsed > 0 else 0,
        'realtime_ratio': duration / elapsed if elapsed > 0 else 0,
        'frame_size_mb': frame_size / 1024 / 1024,
        'total_data_mb': frames_read * frame_size / 1024 / 1024,
    }


def benchmark_yolo_inference(frames: list[np.ndarray], batch_size: int = 32) -> dict:
    """测试 YOLO 推理速度"""
    from ultralytics import YOLO
    
    print(f"  加载 YOLO 模型...")
    model = YOLO("yolov8n.pt")
    
    # 预热
    if frames:
        _ = model(frames[0], verbose=False)
    
    total_frames = len(frames)
    results_count = 0
    
    start_time = time.perf_counter()
    
    # 按 batch 推理
    for i in range(0, total_frames, batch_size):
        batch = frames[i:i+batch_size]
        results = model(batch, verbose=False, conf=0.5, classes=[0])
        results_count += len(results)
    
    elapsed = time.perf_counter() - start_time
    
    return {
        'total_frames': total_frames,
        'batch_size': batch_size,
        'inference_time': elapsed,
        'fps_inference': total_frames / elapsed if elapsed > 0 else 0,
        'ms_per_frame': (elapsed / total_frames * 1000) if total_frames > 0 else 0,
    }


def benchmark_queue_transfer(frames: list[np.ndarray], batch_size: int = 32) -> dict:
    """测试 Queue 传输速度（简化版：测量序列化开销）"""
    import pickle
    
    total_frames = len(frames)
    total_bytes = sum(f.nbytes for f in frames)
    
    # 测量序列化开销（这是 Queue 传输的主要成本）
    start_time = time.perf_counter()
    
    serialized_batches = []
    for i in range(0, total_frames, batch_size):
        batch = frames[i:i+batch_size]
        serialized = pickle.dumps(batch)
        serialized_batches.append(serialized)
    
    serialize_time = time.perf_counter() - start_time
    
    # 测量反序列化
    start_time = time.perf_counter()
    
    for serialized in serialized_batches:
        _ = pickle.loads(serialized)
    
    deserialize_time = time.perf_counter() - start_time
    
    total_time = serialize_time + deserialize_time
    serialized_size = sum(len(s) for s in serialized_batches)
    
    return {
        'total_frames': total_frames,
        'total_mb': total_bytes / 1024 / 1024,
        'serialized_mb': serialized_size / 1024 / 1024,
        'serialize_time': serialize_time,
        'deserialize_time': deserialize_time,
        'transfer_time': total_time,
        'fps_transfer': total_frames / total_time if total_time > 0 else 0,
        'throughput_mb_s': (total_bytes / 1024 / 1024) / total_time if total_time > 0 else 0,
    }


def benchmark_video_slice(video_path: str, segments: list[tuple[float, float]]) -> dict:
    """测试视频切片速度"""
    import tempfile
    
    if not segments:
        segments = [(0, 10), (20, 30)]  # 默认切两段
    
    temp_files = []
    
    start_time = time.perf_counter()
    
    for i, (start, end) in enumerate(segments):
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_files.append(temp_file.name)
        
        # 使用 -an 忽略音频，避免编解码器问题
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(end - start),
            "-c:v", "copy",
            "-an",  # 忽略音频
            "-loglevel", "error",
            temp_file.name
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            # 如果 copy 失败，尝试重编码
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(end - start),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-an",
                "-loglevel", "error",
                temp_file.name
            ]
            subprocess.run(cmd, capture_output=True)
    
    elapsed = time.perf_counter() - start_time
    
    # 清理临时文件
    for f in temp_files:
        try:
            os.unlink(f)
        except:
            pass
    
    total_duration = sum(end - start for start, end in segments)
    
    return {
        'segments': len(segments),
        'total_duration': total_duration,
        'slice_time': elapsed,
        'realtime_ratio': total_duration / elapsed if elapsed > 0 else 0,
    }


def benchmark_video_merge(video_path: str, num_segments: int = 3) -> dict:
    """测试视频合并速度"""
    import tempfile
    
    info = probe_video_info(video_path)
    if not info:
        return {'error': 'Cannot probe video'}
    
    duration = info['duration']
    segment_len = duration / (num_segments + 1)
    
    # 先切片
    temp_segments = []
    for i in range(num_segments):
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_segments.append(temp_file.name)
        
        start = i * segment_len
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(segment_len),
            "-c", "copy",
            "-loglevel", "error",
            temp_file.name
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    # 创建 concat 文件
    concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for seg in temp_segments:
        concat_file.write(f"file '{seg}'\n")
    concat_file.close()
    
    output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    
    start_time = time.perf_counter()
    
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file.name,
        "-c", "copy",
        "-loglevel", "error",
        output_file.name
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    elapsed = time.perf_counter() - start_time
    
    # 清理
    for f in temp_segments + [concat_file.name, output_file.name]:
        try:
            os.unlink(f)
        except:
            pass
    
    total_duration = segment_len * num_segments
    
    return {
        'segments': num_segments,
        'total_duration': total_duration,
        'merge_time': elapsed,
        'realtime_ratio': total_duration / elapsed if elapsed > 0 else 0,
    }


def benchmark_full_pipeline_single_video(video_path: str, sample_interval: float = 3.0, decode_threads: int = 2) -> dict:
    """完整 pipeline 单视频测试（不启动多进程）"""
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print(f"测试视频: {Path(video_path).name}")
    print(f"{'='*60}")
    
    # 1. 探测视频信息
    info = probe_video_info(video_path)
    if not info:
        return {'error': 'Cannot probe video'}
    
    print(f"视频信息: {info['duration']:.1f}s, {info['width']}x{info['height']}, {info['fps']:.1f}fps")
    print(f"解码线程数: {decode_threads}")
    
    results = {
        'video_duration': info['duration'],
        'video_resolution': f"{info['width']}x{info['height']}",
        'video_fps': info['fps'],
        'decode_threads': decode_threads,
    }
    
    # 2. 测试 FFmpeg 解码
    print(f"\n[1/5] 测试 FFmpeg 解码...")
    decode_result = benchmark_ffmpeg_decode(video_path, sample_interval, decode_threads)
    results['decode'] = decode_result
    print(f"  ✓ 解码 {decode_result['frames_read']} 帧, 耗时 {decode_result['decode_time']:.2f}s")
    print(f"    解码速度: {decode_result['fps_decode']:.1f} fps")
    print(f"    实时倍率: {decode_result['realtime_ratio']:.1f}x")
    
    # 获取帧数据用于后续测试
    print(f"\n[2/5] 重新读取帧数据用于推理测试...")
    info = probe_video_info(video_path)
    fps = info['fps']
    width = info['width']
    height = info['height']
    frame_interval = max(1, int(fps * sample_interval))
    frame_size = width * height * 3
    
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
    frames = []
    while True:
        raw_data = proc.stdout.read(frame_size)
        if len(raw_data) != frame_size:
            break
        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame.copy())  # 复制数据，因为后续 buffer 会被覆盖
    proc.stdout.close()
    proc.wait()
    print(f"  ✓ 读取了 {len(frames)} 帧")
    
    # 3. 测试 YOLO 推理
    print(f"\n[3/5] 测试 YOLO 推理...")
    if frames:
        inference_result = benchmark_yolo_inference(frames, batch_size=32)
        results['inference'] = inference_result
        print(f"  ✓ 推理 {inference_result['total_frames']} 帧, 耗时 {inference_result['inference_time']:.2f}s")
        print(f"    推理速度: {inference_result['fps_inference']:.1f} fps")
        print(f"    每帧耗时: {inference_result['ms_per_frame']:.1f} ms")
    else:
        print(f"  ⚠ 无帧数据，跳过推理测试")
    
    # 4. 测试 Queue 传输
    print(f"\n[4/5] 测试跨进程 Queue 传输...")
    if frames:
        transfer_result = benchmark_queue_transfer(frames, batch_size=32)
        results['transfer'] = transfer_result
        print(f"  ✓ 传输 {transfer_result['total_frames']} 帧 ({transfer_result['total_mb']:.1f} MB)")
        print(f"    传输耗时: {transfer_result['transfer_time']:.2f}s")
        print(f"    传输速度: {transfer_result['fps_transfer']:.1f} fps")
        print(f"    吞吐量: {transfer_result['throughput_mb_s']:.1f} MB/s")
    
    # 5. 测试视频切片
    print(f"\n[5/5] 测试视频切片...")
    slice_result = benchmark_video_slice(video_path, [(5, 15), (25, 35)])
    results['slice'] = slice_result
    print(f"  ✓ 切片 {slice_result['segments']} 段, 耗时 {slice_result['slice_time']:.2f}s")
    print(f"    实时倍率: {slice_result['realtime_ratio']:.1f}x")
    
    # 计算总结
    print(f"\n{'='*60}")
    print("性能瓶颈分析")
    print(f"{'='*60}")
    
    total_time = (
        decode_result['decode_time'] + 
        results.get('inference', {}).get('inference_time', 0) +
        slice_result['slice_time']
    )
    
    print(f"\n假设处理 {info['duration']:.1f}s 视频的各环节耗时占比:")
    print(f"  [解码]  {decode_result['decode_time']:.2f}s ({decode_result['decode_time']/total_time*100:.1f}%)")
    if 'inference' in results:
        print(f"  [推理]  {results['inference']['inference_time']:.2f}s ({results['inference']['inference_time']/total_time*100:.1f}%)")
    print(f"  [切片]  {slice_result['slice_time']:.2f}s ({slice_result['slice_time']/total_time*100:.1f}%)")
    print(f"  ────────────────────────")
    print(f"  [总计]  {total_time:.2f}s")
    print(f"  [理论处理倍率] {info['duration']/total_time:.1f}x 实时")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="性能瓶颈测试")
    parser.add_argument("input", help="输入视频目录")
    parser.add_argument("--interval", type=float, default=3.0, help="采样间隔（秒）")
    parser.add_argument("--decode-threads", type=int, default=2, help="FFmpeg 解码线程数")
    args = parser.parse_args()
    
    # macOS/Linux 多进程设置
    if os.name != 'nt':
        mp.set_start_method('spawn', force=True)
    
    videos = get_video_files(args.input)
    
    if not videos:
        print(f"未找到视频文件: {args.input}")
        return
    
    print(f"找到 {len(videos)} 个视频文件")
    
    all_results = []
    
    for video in videos:
        result = benchmark_full_pipeline_single_video(str(video), args.interval, args.decode_threads)
        all_results.append(result)
    
    # 汇总
    print(f"\n\n{'='*60}")
    print("汇总分析")
    print(f"{'='*60}")
    
    if all_results:
        avg_decode_fps = np.mean([r['decode']['fps_decode'] for r in all_results if 'decode' in r])
        avg_inference_fps = np.mean([r['inference']['fps_inference'] for r in all_results if 'inference' in r])
        avg_transfer_fps = np.mean([r['transfer']['fps_transfer'] for r in all_results if 'transfer' in r])
        
        print(f"\n平均性能指标:")
        print(f"  FFmpeg 解码速度:  {avg_decode_fps:.1f} fps")
        print(f"  YOLO 推理速度:    {avg_inference_fps:.1f} fps")
        print(f"  Queue 传输速度:   {avg_transfer_fps:.1f} fps")
        
        # 瓶颈判断
        min_fps = min(avg_decode_fps, avg_inference_fps, avg_transfer_fps)
        
        print(f"\n🔍 瓶颈分析:")
        if min_fps == avg_decode_fps:
            print(f"  ⚠️  FFmpeg 解码是瓶颈 ({avg_decode_fps:.1f} fps)")
            print(f"     建议: 增加 decode_threads 或减少 I/O Worker 数量")
        elif min_fps == avg_inference_fps:
            print(f"  ⚠️  YOLO 推理是瓶颈 ({avg_inference_fps:.1f} fps)")
            print(f"     建议: 增加 Detector Worker 数量或使用更小的模型")
        else:
            print(f"  ⚠️  Queue 传输是瓶颈 ({avg_transfer_fps:.1f} fps)")
            print(f"     建议: 增大 batch_size 减少传输次数")
        
        # 理论吞吐量
        video_duration = np.mean([r['video_duration'] for r in all_results])
        frames_per_video = np.mean([r['decode']['frames_read'] for r in all_results if 'decode' in r])
        
        print(f"\n📊 理论分析 (每个 {video_duration:.0f}s 视频约 {frames_per_video:.0f} 帧):")
        print(f"  单进程理论处理速度: {min_fps / frames_per_video * video_duration:.1f}x 实时")


if __name__ == "__main__":
    main()
