# Surveillance Video Human Detector + Auto Merging

A high-performance Python tool for detecting human presence in surveillance videos, extracting video segments containing people, and merging them into highlight clips. Uses YOLOv8 for person detection and FFmpeg for video processing with multi-threaded decoding.

监控视频人形检测工具 - 自动检测监控视频中的人物，裁剪包含人物的片段，并合并输出。采用多进程架构和 FFmpeg 多线程解码，高效利用多核 CPU。

[中文](README_CN.md) | [English](README.md)

---

## English

### Features

- **Human Detection**: Detect people in surveillance videos using YOLOv8
- **Smart Video Slicing**: Extract only video clips containing people using FFmpeg
- **Automatic Merging**: Merge consecutive video clips into single files
- **Incremental Processing**: Skip already processed files using JSONL log
- **Smart Grouping**: Detect and group consecutive surveillance videos based on creation time
- **Multi-Brand Support**: Extensible timestamp parsers for different camera brands (Xiaomi, Hikvision, Dahua, etc.)
- **High-Performance Pipeline**: Separate I/O Workers and Detector Workers with FFmpeg multi-threaded decoding
- **Auto Configuration**: Automatically configures workers and batch size based on CPU cores
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Supported Camera Brands

| Brand | Parser | File Naming Pattern |
|-------|--------|---------------------|
| **Xiaomi (小米)** | `xiaomi` | Folder: `YYYYMMDDHH` + File: `MMmSSs_TIMESTAMP.mp4` |
| **Hikvision (海康威视)** | `hikvision` | `YYYYMMDDHHMMSS.mp4` or `ch01_YYYYMMDDHHMMSS.mp4` |
| **Dahua (大华)** | `dahua` | `YYYY-MM-DD HH-MM-SS.mp4` or `YYYYMMDD_HHMMSS.mp4` |
| **Generic** | `generic` | Auto-detect common timestamp formats |
| **Auto** | `auto` | Try all parsers in sequence |

### Requirements

- Python 3.10+
- FFmpeg (for video processing and decoding)
- GPU (optional, for faster detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/surveillance-video-human-detector.git
   cd surveillance-video-human-detector
   ```

2. **Install dependencies using uv**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment and install dependencies
   uv sync
   ```

3. **Install FFmpeg**
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Windows (using Chocolatey)
   choco install ffmpeg
   ```

### Usage

#### Quick Start

```bash
# Basic usage (default: Xiaomi camera, auto-configured workers)
uv run python pipeline.py ./test-videos

# Specify camera brand
uv run python pipeline.py ./videos --brand hikvision
uv run python pipeline.py ./videos --brand dahua
uv run python pipeline.py ./videos --brand auto  # Auto-detect

# With custom output directory
uv run python pipeline.py ./videos -o ./output

# Use a more accurate model
uv run python pipeline.py ./videos --model yolov8s.pt
```

**Options:**
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `input` | | (required) | Input video directory path |
| `--output` | `-o` | `results` | Output results directory path |
| `--brand` | `-b` | `xiaomi` | Camera brand (xiaomi/hikvision/dahua/generic/auto) |
| `--threshold` | `-t` | `300` | Continuity threshold in seconds (default: 5 minutes) |
| `--workers` | `-w` | auto | I/O Worker processes (auto-configured by CPU cores) |
| `--detectors` | `-d` | auto | Detector Worker processes (auto-configured by CPU cores) |
| `--batch-size` | | auto | Batch size for detection (auto-configured by CPU cores) |
| `--model` | `-m` | `yolov8n.pt` | YOLO model name |
| `--confidence` | `-c` | `0.5` | Detection confidence threshold |
| `--interval` | | `3.0` | Sample interval in seconds |
| `--scan-workers` | | `8` | File scanning parallel threads |

#### Auto Configuration

The pipeline automatically configures workers based on CPU cores:

| CPU Cores | I/O Workers | Detectors | Batch Size |
|-----------|-------------|-----------|------------|
| 1-2       | 2           | 1         | 16         |
| 3-4       | 4           | 2         | 32         |
| 5-8       | 10          | 6         | 48         |
| 9-16      | 16          | 10        | 64         |
| 17+       | 20          | 14        | 64         |

#### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Producer (Main Process)                      │
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
│  FFmpeg decode│   │  FFmpeg decode│   │  FFmpeg decode│
│  (multi-thread)│  │  (multi-thread)│  │  (multi-thread)│
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────▼───────────┐
                │   Detection Queue     │
                │   (batch frames)      │
                └───────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────────┐
        │                   │                       │
        ▼                   ▼                       ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Detector 1   │   │  Detector 2   │   │  Detector 3   │
│  YOLO Model   │   │  YOLO Model   │   │  YOLO Model   │
│  batch infer  │   │  batch infer  │   │  batch infer  │
└───────────────┘   └───────────────┘   └───────────────┘
```

**Optimizations:**
- **FFmpeg multi-threaded decoding**: 50-100% faster than cv2.VideoCapture
- **Select filter frame skipping**: Skip frames at decode stage, no seek overhead
- **Multiple Detector Workers**: Parallel YOLO inference on multiple CPU cores
- **Async prefetch (depth 8)**: I/O workers can send multiple batches without waiting

#### Processing Flow

```
input-videos/
    │
    ▼ Step 1: Scan & Parse Timestamps
    │ (Use brand-specific parser to extract creation time from filenames)
    │
    ▼ Step 2: Deduplicate
    │ (Check log.jsonl, clean invalid entries)
    │
    ▼ Step 3: Group by Continuity
    │ (Group files based on creation time and duration)
    │
    ▼ Step 4: Parallel Processing
    │ ├─ [Group A: file1, file2, file3] → Detect → Slice → Merge → output1.mp4
    │ └─ [Group B: file4, file5]        → Detect → Slice → Merge → output2.mp4
    │
    ▼ Step 5: Write Log
    │ (Append results to log.jsonl with process lock)
    │
    ▼
output/
├── log.jsonl             # Processing log (JSONL format)
└── merged/               # Output videos
    ├── 20231115_143052_p_2.mp4        # Single video output
    └── 20231115_143052_p_3_merged.mp4 # Merged video output
```

#### Incremental Processing

The pipeline automatically tracks processed files using `log.jsonl`:

- **Skip processed files**: Files already in the log are skipped
- **Resume interrupted processing**: Just run the command again
- **Auto-cleanup**: Invalid log entries (missing output files) are automatically removed
- **Reprocess files**: Delete the corresponding entry in `log.jsonl`

### Adding Support for New Camera Brands

To add support for a new camera brand, create a new parser class in `timestamp_parser.py`:

```python
from timestamp_parser import BaseTimestampParser, register_parser
from pathlib import Path

class MyBrandParser(BaseTimestampParser):
    brand = "mybrand"
    description = "My Brand: custom naming pattern"
    
    def parse(self, file_path: Path) -> float | None:
        # Implement your parsing logic here
        # Return Unix timestamp or None if parsing fails
        filename = file_path.stem
        # ... parse timestamp from filename
        return timestamp

# Register the parser
register_parser("mybrand", MyBrandParser, aliases=["mb", "我的品牌"])
```

### Output Naming

- **Single video with single segment**: `{time_prefix}_p_{max_person}.mp4`
- **Multiple segments or merged videos**: `{time_prefix}_p_{max_person}_merged.mp4`

Where:
- `time_prefix`: Creation time from filename in format `YYYYMMDD_HHMMSS`
- `max_person`: Maximum number of people detected

---

## Project Structure

```
surveillance-video-human-detector/
├── pipeline.py             # Main entry point & parallel processing pipeline
├── timestamp_parser.py     # Extensible timestamp parsers for camera brands
├── file_time_utils.py      # File time utilities
├── pyproject.toml          # Project dependencies
├── yolov8n.pt              # YOLO model weights
└── README.md               # This file
```

---

## License

MIT License
