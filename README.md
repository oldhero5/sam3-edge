# SAM3-Edge

Real-time video segmentation on NVIDIA Jetson using SAM3 + TensorRT + DeepStream.

## Overview

SAM3-Edge brings Meta's Segment Anything Model 3 (SAM3) to edge devices, specifically optimized for NVIDIA Jetson AGX Orin. It provides:

- **TensorRT Acceleration**: ViT encoder and mask decoder exported to TensorRT FP16 engines
- **DeepStream Integration**: GStreamer pipeline with nvinfer plugin for hardware-accelerated video processing
- **FastAPI Server**: REST API for inference on images and video streams
- **CUDA Streams**: Async pipelined inference to minimize latency

## Architecture

```
sam3-edge/
├── sam3_deepstream/          # Main package
│   ├── api/                  # FastAPI REST server
│   ├── export/               # TensorRT export utilities
│   ├── inference/            # Runtime inference engines
│   └── deepstream/           # DeepStream pipeline configs
└── pyproject.toml            # UV workspace config (pulls sam3 from fork)
```

**Dependencies:**
- [sam3](https://github.com/oldhero5/sam3) - SAM3 model with Jetson optimizations (pulled as git dependency)

## Quick Start

### Prerequisites

- NVIDIA Jetson AGX Orin with JetPack 6.0+
- CUDA 12.x, TensorRT 10.x
- Python 3.10+, UV package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/oldhero5/sam3-edge.git
cd sam3-edge

# Install dependencies (UV will fetch sam3 from fork)
uv sync

# Download SAM3 checkpoint
# Place in ~/.cache/sam3_deepstream/checkpoints/

# Export TensorRT engines
uv run python -m sam3_deepstream.scripts.export_engines \
    --checkpoint ~/.cache/sam3_deepstream/checkpoints/sam3_hiera_large.pt \
    --output-dir ~/.cache/sam3_deepstream/engines \
    --precision fp16
```

### Run API Server

```bash
# Start FastAPI server
uv run sam3-api

# Or with Docker
cd sam3_deepstream
docker compose -f docker-compose.jetson.yml up
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Segment an image
curl -X POST "http://localhost:8000/segment" \
  -F "image=@test.jpg" \
  -F "points=[[512,512]]" \
  -F "labels=[1]"
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | HuggingFace token for model download |
| `SAM3_CHECKPOINT` | - | Path to SAM3 checkpoint |
| `SAM3_ENCODER_ENGINE` | - | Path to encoder TensorRT engine |
| `SAM3_DECODER_ENGINE` | - | Path to decoder TensorRT engine |
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8000 | API server port |

## Performance

On Jetson AGX Orin (64GB):

| Component | Precision | Latency |
|-----------|-----------|---------|
| ViT Encoder | FP16 | ~35ms |
| Mask Decoder | FP16 | ~8ms |
| End-to-end | FP16 | ~50ms |

With keyframe optimization (reusing embeddings):
- Keyframe: ~50ms
- Non-keyframe: ~8ms (decoder only)

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest sam3_deepstream/tests/ -v

# Build Docker image
cd sam3_deepstream
docker build -f Dockerfile.jetson -t sam3-edge:jetson .
```

## License

MIT License - see LICENSE file.

## Acknowledgments

- [Meta AI SAM3](https://github.com/facebookresearch/sam3) - Original SAM3 implementation
- [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) - Video analytics SDK
