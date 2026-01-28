# SAM3 DeepStream

TensorRT + DeepStream integration for SAM3 on Jetson AGX Orin.

## Installation

Use the parent directory's install script for full Jetson setup:

```bash
cd ..
./install_jetson.sh
```

Or install dependencies manually:

```bash
# From parent directory (uses UV workspace)
uv sync --package sam3-deepstream
```

## Export TensorRT Engines

**Important:** TensorRT engines must be built with the same TRT version used at runtime. On Jetson, use Docker to ensure compatibility:

```bash
# Recommended: Use install script (exports inside Docker)
cd ..
./install_jetson.sh

# Manual export inside Docker container
docker run --rm --runtime nvidia --gpus all \
    -v /path/to/checkpoint:/workspace/checkpoints:ro \
    -v ~/.cache/sam3_deepstream/engines:/workspace/engines \
    -e PYTHONPATH=/workspace:/workspace/sam3 \
    sam3-edge:jetson \
    python3 scripts/export_engines.py \
        --checkpoint /workspace/checkpoints/sam3.pt \
        --output-dir /workspace/engines \
        --precision fp16
```

Direct export (only if TensorRT versions match):

```bash
python -m sam3_deepstream.scripts.export_engines \
    --checkpoint /path/to/sam3.pt \
    --output-dir ./engines \
    --precision fp16
```

## API Server

```bash
# Run directly
sam3-api

# Or via module
python -m sam3_deepstream.api.server

# With custom config
SAM3_ENCODER_ENGINE=./engines/encoder.engine \
SAM3_DECODER_ENGINE=./engines/decoder.engine \
sam3-api --host 0.0.0.0 --port 8000
```

## Docker

```bash
# Build for Jetson (from sam3_deepstream directory)
docker compose -f docker-compose.jetson.yml build

# Run
docker compose -f docker-compose.jetson.yml up -d

# Check health
curl http://localhost:8000/health

# Stop
docker compose -f docker-compose.jetson.yml down
```

## Testing

The project includes comprehensive pytest tests for API endpoints and export functionality.

### Running Tests Inside Container

```bash
# Run all tests
docker exec sam3_deepstream-sam3-api-1 pytest /workspace/tests/ -v

# Run only API tests
docker exec sam3_deepstream-sam3-api-1 pytest /workspace/tests/test_api.py -v

# Run only export tests
docker exec sam3_deepstream-sam3-api-1 pytest /workspace/tests/test_export.py -v
```

### Running Tests from Host

```bash
# Run standalone API test script
python sam3_deepstream/scripts/run_api_tests.py --host localhost --port 8000
```

### Curl Smoke Test

```bash
# Text-based segmentation
curl -X POST http://localhost:8000/api/v1/segment \
  -F "file=@test_data/racing.jpeg" \
  -F "text_prompt=car" \
  -F "return_json=true"

# Point-based segmentation
curl -X POST "http://localhost:8000/segment?points=0.5,0.5,1&return_json=true" \
  -F "file=@test_data/racing.jpeg"
```

### Test Data

Test images should be placed in `test_data/` at the project root:
- `racing.jpeg` - Image with race cars for segmentation testing
- `Target.mp4` - Video file for video processing tests

The test data directory is automatically mounted into the container at `/workspace/test_data`.

## Dependencies (Jetson)

The Dockerfile.jetson includes all required dependencies:

- **triton>=2.1.0** - Required for SAM3 EDT (Euclidean Distance Transform) operations
- **decord2>=3.0.0** - ARM64-compatible video decoding library
- **pycocotools>=2.0.0** - COCO evaluation tools
- **pytest, pytest-asyncio, requests** - Testing framework

## Package Structure

```
sam3_deepstream/
├── api/
│   └── server.py         # FastAPI inference server
├── config/
│   └── __init__.py       # Configuration management
├── export/
│   ├── encoder_export.py # ViT encoder TRT export
│   ├── decoder_export.py # Mask decoder TRT export
│   └── engine_manager.py # TRT engine lifecycle
├── inference/
│   └── async_trt_runtime.py  # CUDA streams pipelined inference
├── deepstream/
│   └── sam3_infer_config.txt # DeepStream nvinfer config
└── scripts/
    ├── export_engines.py     # Main export CLI
    └── export_minimal.py     # Lightweight export
```
