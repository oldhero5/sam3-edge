# SAM3 Inference API - Jetson Container
#
# Build:
#   docker build -t sam3-api:latest .
#
# Run:
#   docker run --runtime nvidia -p 8000:8000 \
#     -v /path/to/engines:/workspace/engines \
#     sam3-api:latest
#
# For Jetson, use the l4t-pytorch base image:
#   docker build --build-arg BASE_IMAGE=dustynv/l4t-pytorch:r36.4.0 -t sam3-api:jetson .

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${BASE_IMAGE}

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for caching
COPY requirements-api.txt /workspace/
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the sam3_deepstream package
COPY sam3_deepstream /workspace/sam3_deepstream
COPY pyproject.toml /workspace/

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for engines and data
RUN mkdir -p /workspace/engines /workspace/data

# Environment variables for configuration
ENV SAM3_ENCODER_ENGINE=/workspace/engines/sam3_encoder.engine
ENV SAM3_DECODER_ENGINE=/workspace/engines/sam3_decoder.engine
ENV CUDA_DEVICE=0
ENV USE_ASYNC=true
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "sam3_deepstream.api.server"]
