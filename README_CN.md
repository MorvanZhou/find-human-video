# 监控视频人形检测+合并视频工具

监控视频人形检测工具 - 自动检测监控视频中的人物，裁剪包含人物的片段，并合并输出。采用多进程架构和 FFmpeg 多线程解码，高效利用多核 CPU。

[English](README.md) | [中文](README_CN.md)

---

## 中文说明

### 功能特点

- **人物检测**：使用 YOLOv8 检测监控视频中的人物
- **智能裁剪**：使用 FFmpeg 仅提取包含人物的视频片段
- **自动合并**：自动合并连续的视频片段
- **增量处理**：使用 JSONL 日志自动跳过已处理的文件
- **智能分组**：根据创建时间自动识别并分组连续的监控视频
- **多品牌支持**：可扩展的时间戳解析器，支持不同品牌监控（小米、海康威视、大华等）
- **高性能架构**：I/O Worker 与 Detector Worker 分离，FFmpeg 多线程解码
- **自动配置**：根据 CPU 核心数自动配置最佳参数
- **跨平台**：支持 Windows、macOS 和 Linux

### 支持的监控品牌

| 品牌 | 解析器 | 文件命名规则 |
|------|--------|-------------|
| **小米** | `xiaomi` | 文件夹: `YYYYMMDDHH` + 文件名: `MMmSSs_TIMESTAMP.mp4` |
| **海康威视** | `hikvision` | `YYYYMMDDHHMMSS.mp4` 或 `ch01_YYYYMMDDHHMMSS.mp4` |
| **大华** | `dahua` | `YYYY-MM-DD HH-MM-SS.mp4` 或 `YYYYMMDD_HHMMSS.mp4` |
| **通用** | `generic` | 自动检测多种常见时间格式 |
| **自动** | `auto` | 依次尝试所有解析器 |

### 系统要求

- Python 3.10+
- FFmpeg（用于视频处理和解码）
- GPU（可选，用于加速检测）

### 安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/yourusername/surveillance-video-human-detector.git
   cd surveillance-video-human-detector
   ```

2. **使用 uv 安装依赖**
   ```bash
   # 安装 uv（如果尚未安装）
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # 创建虚拟环境并安装依赖
   uv sync
   ```

3. **安装 FFmpeg**
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Windows（使用 Chocolatey）
   choco install ffmpeg
   ```

### 使用方法

#### 快速开始

```bash
# 基本用法（默认小米监控，自动配置参数）
uv run python pipeline.py ./test-videos

# 指定监控品牌
uv run python pipeline.py ./videos --brand hikvision
uv run python pipeline.py ./videos --brand dahua
uv run python pipeline.py ./videos --brand auto  # 自动检测

# 指定输出目录
uv run python pipeline.py ./videos -o ./output

# 使用更精确的模型
uv run python pipeline.py ./videos --model yolov8s.pt
```

**参数说明：**
| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `input` | | (必填) | 输入视频目录路径 |
| `--output` | `-o` | `results` | 输出结果目录路径 |
| `--brand` | `-b` | `xiaomi` | 监控品牌（xiaomi/hikvision/dahua/generic/auto） |
| `--threshold` | `-t` | `300` | 连续性判断时间阈值（秒，默认 5 分钟） |
| `--workers` | `-w` | 自动 | I/O Worker 进程数量（根据 CPU 核心数自动配置） |
| `--detectors` | `-d` | 自动 | Detector Worker 进程数量（根据 CPU 核心数自动配置） |
| `--batch-size` | | 自动 | 检测批大小（根据 CPU 核心数自动配置） |
| `--model` | `-m` | `yolov8n.pt` | YOLO 模型名称 |
| `--confidence` | `-c` | `0.5` | 检测置信度阈值 |
| `--interval` | | `3.0` | 采样间隔（秒） |
| `--scan-workers` | | `8` | 文件扫描并行线程数 |

#### 自动配置

Pipeline 根据 CPU 核心数自动配置参数：

| CPU 核心数 | I/O Workers | Detectors | Batch Size |
|------------|-------------|-----------|------------|
| 1-2 核     | 2           | 1         | 16         |
| 3-4 核     | 4           | 2         | 32         |
| 5-8 核     | 10          | 6         | 48         |
| 9-16 核    | 16          | 10        | 64         |
| 17+ 核     | 20          | 14        | 64         |

#### Pipeline 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Producer (主进程)                            │
│  扫描 → 分组 → 推入任务队列                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │       任务队列         │
                └───────────┬───────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  I/O Worker 1 │   │  I/O Worker 2 │   │  I/O Worker 3 │
│  FFmpeg 解码  │   │  FFmpeg 解码  │   │  FFmpeg 解码  │
│  (多线程)     │   │  (多线程)     │   │  (多线程)     │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────▼───────────┐
                │     Detection Queue   │
                │     (批量帧数据)       │
                └───────────┬───────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Detector 1   │   │  Detector 2   │   │  Detector 3   │
│  YOLO 模型    │   │  YOLO 模型    │   │  YOLO 模型    │
│  批量推理     │   │  批量推理     │   │  批量推理     │
└───────────────┘   └───────────────┘   └───────────────┘
```

**优化点：**
- **FFmpeg 多线程解码**：比 cv2.VideoCapture 快 50-100%
- **select 滤镜跳帧**：直接在解码阶段过滤帧，无 seek 开销
- **多 Detector Worker**：多个 YOLO 模型实例并行推理
- **异步预取（深度 8）**：I/O Worker 可同时发送多个 batch，无需等待

#### 处理流程

```
input-videos/
    │
    ▼ 步骤 1: 扫描与解析时间戳
    │ (使用指定品牌的解析器从文件名提取创建时间)
    │
    ▼ 步骤 2: 去重
    │ (检查 log.jsonl，清理失效记录)
    │
    ▼ 步骤 3: 连续性分组
    │ (根据创建时间和视频时长分组)
    │
    ▼ 步骤 4: 并行处理
    │ ├─ [组A: 文件1, 文件2, 文件3] → 检测 → 切片 → 合并 → output1.mp4
    │ └─ [组B: 文件4, 文件5]        → 检测 → 切片 → 合并 → output2.mp4
    │
    ▼ 步骤 5: 写入日志
    │ (将结果追加到 log.jsonl，使用进程锁)
    │
    ▼
output/
├── log.jsonl             # 处理日志（JSONL 格式）
└── merged/               # 输出视频
    ├── 20231115_143052_p_2.mp4        # 单视频输出
    └── 20231115_143052_p_3_merged.mp4 # 合并视频输出
```

#### 增量处理

Pipeline 使用 `log.jsonl` 自动跟踪已处理的文件，支持：

- **跳过已处理文件**：日志中已存在的文件会被跳过
- **恢复中断的处理**：直接重新运行命令即可继续
- **自动清理**：输出文件不存在的失效日志记录会被自动移除
- **重新处理文件**：删除 `log.jsonl` 中对应的记录

### 添加新品牌支持

要添加新的监控品牌支持，在 `timestamp_parser.py` 中创建新的解析器类：

```python
from timestamp_parser import BaseTimestampParser, register_parser
from pathlib import Path

class MyBrandParser(BaseTimestampParser):
    brand = "mybrand"
    description = "我的品牌: 自定义命名规则"
    
    def parse(self, file_path: Path) -> float | None:
        # 实现解析逻辑
        # 返回 Unix 时间戳，解析失败返回 None
        filename = file_path.stem
        # ... 从文件名解析时间戳
        return timestamp

# 注册解析器
register_parser("mybrand", MyBrandParser, aliases=["mb", "我的品牌"])
```

### 输出文件命名

- **单视频单片段**：`{time_prefix}_p_{max_person}.mp4`
- **多片段或合并视频**：`{time_prefix}_p_{max_person}_merged.mp4`

其中：
- `time_prefix`：从文件名解析的创建时间，格式为 `YYYYMMDD_HHMMSS`
- `max_person`：检测到的最大人数

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
