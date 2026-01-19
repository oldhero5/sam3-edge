# SAM3 DeepStream

TensorRT + DeepStream integration for SAM3 on Jetson.

## Installation

```bash
# From parent directory
uv sync

# Or install directly
pip install -e .
```

## Export TensorRT Engines

```bash
# Export both encoder and decoder
python -m sam3_deepstream.scripts.export_engines \
    --checkpoint /path/to/sam3_hiera_large.pt \
    --output-dir ./engines \
    --precision fp16

# Encoder only
python -m sam3_deepstream.scripts.export_engines \
    --checkpoint /path/to/checkpoint.pt \
    --encoder-only

# Minimal export (avoids full sam3 import)
python scripts/export_minimal.py \
    --checkpoint /path/to/checkpoint.pt \
    --output-dir ./engines
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
# Build for Jetson
docker build -f Dockerfile.jetson -t sam3-deepstream:jetson .

# Run
docker compose -f docker-compose.jetson.yml up
```

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
