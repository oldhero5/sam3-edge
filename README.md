# SAM3-Edge

Real-time video segmentation on NVIDIA Jetson using SAM3 + TensorRT + DeepStream.

## Overview

SAM3-Edge brings Meta's Segment Anything Model 3 (SAM3) to edge devices, specifically optimized for NVIDIA Jetson AGX Orin. It provides:

- **TensorRT Acceleration**: ViT encoder and mask decoder exported to TensorRT FP16 engines
- **DeepStream Integration**: GStreamer pipeline with nvinfer plugin for hardware-accelerated video processing
- **FastAPI Server**: REST API for inference on images and video streams
- **CUDA Streams**: Async pipelined inference to minimize latency
- **Natural Language Queries**: Text-based object detection ("green traffic lights") with semantic search
- **FAISS Vector Search**: GPU-accelerated similarity search across detected objects
- **SQLite Storage**: Persistent detection storage for offline query

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

## Natural Language Query (NLQ)

SAM3-Edge supports text-based object detection and semantic search, enabling you to:
1. Detect objects using natural language (e.g., "green traffic lights")
2. Store detections with semantic embeddings
3. Search across all detections using natural language queries

### Text-Based Detection

Submit a video with a text prompt to detect specific objects:

```bash
# Detect objects with text prompt
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@traffic_video.mp4" \
  -F "text_prompt=green traffic lights" \
  -F "confidence_threshold=0.5"

# Response
{
  "job_id": "abc123",
  "video_id": "xyz789",
  "status": "processing"
}
```

### Semantic Search

Search across all detected objects using natural language:

```bash
# Search for similar objects
curl "http://localhost:8000/api/v1/search?q=traffic+signal&limit=50"

# Response
{
  "query": "traffic signal",
  "results": [
    {
      "detection_id": 123,
      "video_id": "xyz789",
      "frame_idx": 150,
      "text_prompt": "green traffic lights",
      "confidence": 0.87,
      "similarity": 0.92,
      "bbox": [0.1, 0.2, 0.15, 0.25]
    }
  ]
}
```

### List Detected Objects

Aggregate detected objects by label:

```bash
# List all detected object types
curl "http://localhost:8000/api/v1/objects?group_by=label"

# Response
{
  "objects": [
    {"label": "traffic_light", "count": 42, "avg_confidence": 0.78},
    {"label": "vehicle", "count": 156, "avg_confidence": 0.85}
  ]
}
```

### NLQ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/detect` | POST | Submit text prompt detection job |
| `/api/v1/search` | GET | Semantic search across detections |
| `/api/v1/objects` | GET | List/aggregate detected objects |
| `/api/v1/detections/{id}` | GET | Get single detection details |
| `/api/v1/export` | GET | Export detections for federation |

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
| `SAM3_DB_PATH` | ~/.cache/sam3_deepstream/db/detections.db | SQLite database path |
| `SAM3_FAISS_INDEX` | ~/.cache/sam3_deepstream/db/embeddings.index | FAISS index path |
| `SAM3_USE_GPU_FAISS` | true | Use GPU for FAISS operations |

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

## Jetson Installation

For complete Jetson AGX Orin setup, use the installation script:

```bash
# Full installation (system deps, Docker, database)
./install_jetson.sh

# With TensorRT engine export
./install_jetson.sh --checkpoint /path/to/sam3_checkpoint.pt

# Docker image only
./install_jetson.sh --docker-only

# Health check
./install_jetson.sh --health-check

# Initialize database only
./install_jetson.sh --init-db
```

### Installation Options

| Option | Description |
|--------|-------------|
| `--checkpoint <path>` | Export TensorRT engines from checkpoint |
| `--docker-only` | Only build Docker image |
| `--skip-docker` | Skip Docker image build |
| `--init-db` | Initialize database only |
| `--health-check` | Run system health checks |
| `--precision <fp16\|fp32\|int8>` | TensorRT precision (default: fp16) |
| `--push` | Push image to DockerHub |
| `--repo <user/repo>` | DockerHub repository name |

## DockerHub

### Pulling Pre-built Images

```bash
# Pull Jetson image
docker pull sam3edge/sam3-edge:jetson-latest

# Run with docker-compose
cd sam3_deepstream
docker compose -f docker-compose.jetson.yml up -d
```

### Publishing to DockerHub

```bash
# Build and push to your DockerHub account
./install_jetson.sh --push --repo yourusername/sam3-edge

# Or manually:
docker login
docker tag sam3-edge:jetson yourusername/sam3-edge:jetson-latest
docker push yourusername/sam3-edge:jetson-latest
```

## Federation (Future)

SAM3-Edge is designed for future multi-device federation where many Jetson devices process video at the edge and sync detections to a central server.

Current design supports:
- `device_id` tracking in all database tables
- Export endpoint for syncing detections
- Embedding-based deduplication

Future features (planned):
- Automatic sync to central aggregation server
- Federated learning across devices
- Distributed query across device network

## License

MIT License - see LICENSE file.

## Acknowledgments

- [Meta AI SAM3](https://github.com/facebookresearch/sam3) - Original SAM3 implementation
- [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) - Video analytics SDK
