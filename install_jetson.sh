#!/bin/bash
# SAM3-Edge Jetson AGX Orin Installation Script
#
# Complete installation for Jetson devices with JetPack 6.x
# Includes: Docker setup, TensorRT export, database initialization
#
# Usage:
#   ./install_jetson.sh                    # Full installation
#   ./install_jetson.sh --docker-only      # Just build Docker image
#   ./install_jetson.sh --init-db          # Initialize database only
#   ./install_jetson.sh --health-check     # Run health checks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${CYAN}$1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# Default values
DOCKER_ONLY=false
INIT_DB_ONLY=false
HEALTH_CHECK_ONLY=false
CHECKPOINT_PATH=""
ENGINE_DIR="$HOME/.cache/sam3_deepstream/engines"
DB_DIR="$HOME/.cache/sam3_deepstream/db"
PRECISION="fp16"
SKIP_DOCKER=false
DOCKERHUB_PUSH=false
DOCKERHUB_REPO=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker-only)
            DOCKER_ONLY=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --init-db)
            INIT_DB_ONLY=true
            shift
            ;;
        --health-check)
            HEALTH_CHECK_ONLY=true
            shift
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --engine-dir)
            ENGINE_DIR="$2"
            shift 2
            ;;
        --db-dir)
            DB_DIR="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --push)
            DOCKERHUB_PUSH=true
            shift
            ;;
        --repo)
            DOCKERHUB_REPO="$2"
            shift 2
            ;;
        -h|--help)
            echo "SAM3-Edge Jetson Installation Script"
            echo ""
            echo "Usage: ./install_jetson.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docker-only     Only build Docker image"
            echo "  --skip-docker     Skip Docker image build"
            echo "  --init-db         Initialize database only"
            echo "  --health-check    Run health checks only"
            echo "  --checkpoint      Path to SAM3 checkpoint for TensorRT export"
            echo "  --engine-dir      Directory for TensorRT engines (default: ~/.cache/sam3_deepstream/engines)"
            echo "  --db-dir          Directory for database files (default: ~/.cache/sam3_deepstream/db)"
            echo "  --precision       TensorRT precision: fp16, fp32, int8 (default: fp16)"
            echo "  --push            Push Docker image to DockerHub"
            echo "  --repo            DockerHub repository (e.g., username/sam3-edge)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./install_jetson.sh                                    # Full installation"
            echo "  ./install_jetson.sh --checkpoint ~/models/sam3.pt      # With TensorRT export"
            echo "  ./install_jetson.sh --docker-only                      # Just build Docker"
            echo "  ./install_jetson.sh --push --repo myuser/sam3-edge     # Build and push to DockerHub"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Health check only
if [ "$HEALTH_CHECK_ONLY" = true ]; then
    print_header "Running Health Checks"

    # Check Jetson
    if [ -f /etc/nv_tegra_release ]; then
        print_success "Jetson platform detected"
        cat /etc/nv_tegra_release
    else
        print_warning "Not running on Jetson"
    fi

    # Check Docker
    if docker info &> /dev/null; then
        print_success "Docker is running"

        # Check NVIDIA runtime
        if docker info 2>/dev/null | grep -q "nvidia"; then
            print_success "NVIDIA Docker runtime available"
        else
            print_warning "NVIDIA Docker runtime not configured"
        fi
    else
        print_error "Docker is not running"
    fi

    # Check CUDA
    if command -v nvcc &> /dev/null; then
        print_success "CUDA: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
    else
        print_warning "nvcc not found (may still work with JetPack)"
    fi

    # Check TensorRT
    if python3 -c "import tensorrt" 2>/dev/null; then
        TRT_VERSION=$(python3 -c "import tensorrt; print(tensorrt.__version__)")
        print_success "TensorRT: $TRT_VERSION"
    else
        print_warning "TensorRT Python bindings not available"
    fi

    # Check API endpoint
    if curl -s http://localhost:8000/health &> /dev/null; then
        print_success "API server is responding"
        curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || true
    else
        print_info "API server not running (start with: docker compose up)"
    fi

    # Check database
    if [ -f "$DB_DIR/detections.db" ]; then
        print_success "Database exists: $DB_DIR/detections.db"
        DB_SIZE=$(du -h "$DB_DIR/detections.db" | cut -f1)
        print_info "Database size: $DB_SIZE"
    else
        print_info "Database not initialized yet"
    fi

    # Check FAISS index
    if [ -f "$DB_DIR/embeddings.index" ]; then
        print_success "FAISS index exists: $DB_DIR/embeddings.index"
    else
        print_info "FAISS index not created yet"
    fi

    exit 0
fi

# Initialize database only
if [ "$INIT_DB_ONLY" = true ]; then
    print_header "Initializing Database"

    mkdir -p "$DB_DIR"

    print_info "Creating SQLite database and FAISS index..."

    python3 << EOF
import sys
sys.path.insert(0, '.')

from sam3_deepstream.api.services.detection_store import DetectionStore
import asyncio

async def init():
    store = DetectionStore(
        db_path="$DB_DIR/detections.db",
        index_path="$DB_DIR/embeddings.index",
        use_gpu=True
    )
    await store.initialize()
    print("Database initialized successfully")

asyncio.run(init())
EOF

    print_success "Database initialized at $DB_DIR"
    exit 0
fi

# Print banner
echo -e "${CYAN}"
cat << 'EOF'
  ____    _    __  __ _____       _____    _
 / ___|  / \  |  \/  |___ /      | ____|__| | __ _  ___
 \___ \ / _ \ | |\/| | |_ \ _____|  _| / _` |/ _` |/ _ \
  ___) / ___ \| |  | |___) |_____| |__| (_| | (_| |  __/
 |____/_/   \_\_|  |_|____/      |_____\__,_|\__, |\___|
                                             |___/
         Jetson AGX Orin Installation
EOF
echo -e "${NC}"

# Verify Jetson platform
print_header "Verifying Jetson Platform"

if [ -f /etc/nv_tegra_release ]; then
    JETSON_VERSION=$(cat /etc/nv_tegra_release | head -1)
    print_success "Jetson detected: $JETSON_VERSION"
else
    print_warning "Not running on Jetson hardware"
    print_info "This script is optimized for Jetson AGX Orin"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check JetPack version
if [ -f /etc/nv_tegra_release ]; then
    if grep -q "R36" /etc/nv_tegra_release; then
        print_success "JetPack 6.x detected (recommended)"
    elif grep -q "R35" /etc/nv_tegra_release; then
        print_warning "JetPack 5.x detected - some features may not work"
    fi
fi

if [ "$DOCKER_ONLY" = false ]; then
    # Check and install system dependencies
    print_header "Installing System Dependencies"

    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        SUDO="sudo"
    else
        SUDO=""
    fi

    # Update package lists
    print_info "Updating package lists..."
    $SUDO apt-get update

    # Install required packages
    print_info "Installing required packages..."
    $SUDO apt-get install -y --no-install-recommends \
        python3.10-venv \
        python3-pip \
        libsqlite3-dev \
        curl \
        git \
        bc

    print_success "System dependencies installed"

    # Configure NVIDIA Docker runtime
    print_header "Configuring Docker Runtime"

    if command -v docker &> /dev/null; then
        print_success "Docker is installed"

        # Check if nvidia-ctk is available
        if command -v nvidia-ctk &> /dev/null; then
            print_info "Configuring NVIDIA Container Toolkit..."
            $SUDO nvidia-ctk runtime configure --runtime=docker
            $SUDO systemctl restart docker
            print_success "NVIDIA Docker runtime configured"
        else
            print_warning "nvidia-ctk not found - NVIDIA runtime may not work"
            print_info "Install with: sudo apt-get install nvidia-container-toolkit"
        fi
    else
        print_error "Docker not installed"
        print_info "Install Docker: https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html"
        exit 1
    fi

    # Install UV package manager
    print_header "Installing UV Package Manager"

    if command -v uv &> /dev/null; then
        print_success "UV already installed: $(uv --version)"
    else
        print_info "Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        print_success "UV installed"
    fi

    # Install Python dependencies
    print_header "Installing Python Dependencies"

    # Verify Jetson PyTorch is available before proceeding
    if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        print_success "Jetson PyTorch with CUDA detected: $TORCH_VERSION"
    else
        print_error "System PyTorch with CUDA not found"
        print_info "Install PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/"
        exit 1
    fi

    # Verify torchvision is available
    if python3 -c "import torchvision" 2>/dev/null; then
        TV_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)")
        print_success "Jetson torchvision detected: $TV_VERSION"
    else
        print_error "torchvision not found in system packages"
        print_info ""
        print_info "On Jetson, torchvision must be installed from NVIDIA's wheel server."
        print_info "DO NOT use 'pip install torchvision' - it will break your CUDA PyTorch!"
        print_info ""
        print_info "Run these commands manually:"
        echo ""
        echo "  sudo pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \\"
        echo "      --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126"
        echo ""
        print_info "Then re-run this script."
        exit 1
    fi

    # Double-check that PyTorch still has CUDA after torchvision import
    # (This catches the case where a broken torchvision pulled in CPU-only torch)
    if ! python3 -c "import torch; import torchvision; assert torch.cuda.is_available(), 'CUDA broken'" 2>/dev/null; then
        print_error "PyTorch CUDA is broken after loading torchvision!"
        print_info "This usually means a CPU-only torch was installed from PyPI."
        print_info ""
        print_info "Fix with:"
        echo ""
        echo "  sudo pip3 uninstall -y torch torchvision"
        echo "  sudo pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \\"
        echo "      --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126"
        echo ""
        exit 1
    fi

    print_info "Creating virtual environment with system site packages..."
    # Use system-site-packages to inherit Jetson's PyTorch with CUDA
    uv venv --system-site-packages

    print_info "Syncing workspace dependencies..."
    # Skip torch/torchvision/triton - Jetson uses NVIDIA's pre-built PyTorch from JetPack
    uv sync --package sam3-deepstream --no-install-package torch --no-install-package torchvision --no-install-package triton
    print_success "Python dependencies installed"

    # Create directories for database and engines
    print_header "Preparing Directories"

    mkdir -p "$DB_DIR"
    mkdir -p "$ENGINE_DIR"

    print_success "Database directory: $DB_DIR"
    print_success "Engine directory: $ENGINE_DIR"
    print_info "Database will be initialized on first API request"
fi

# Build Docker image FIRST (needed for TRT export to match container's TRT version)
if [ "$SKIP_DOCKER" = false ]; then
    print_header "Building Docker Image"

    cd sam3_deepstream

    print_info "Building Jetson Docker image..."
    docker build -f Dockerfile.jetson -t sam3-edge:jetson .

    print_success "Docker image built: sam3-edge:jetson"

    # Tag with version
    VERSION=$(grep 'version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
    docker tag sam3-edge:jetson sam3-edge:jetson-$VERSION
    print_success "Tagged as: sam3-edge:jetson-$VERSION"

    cd ..
fi

# Export TensorRT engines INSIDE Docker container
# This ensures TRT version matches the runtime environment
if [ "$DOCKER_ONLY" = false ]; then
    print_header "Exporting TensorRT Engines (inside Docker)"

    # Check if engines already exist
    ENCODER_ENGINE="$ENGINE_DIR/sam3_encoder.engine"
    DECODER_ENGINE="$ENGINE_DIR/sam3_decoder.engine"

    if [ -f "$ENCODER_ENGINE" ] && [ -f "$DECODER_ENGINE" ]; then
        print_success "TensorRT engines already exist at $ENGINE_DIR"
        print_info "To re-export, delete existing engines and run again"
    else
        # If no checkpoint provided, search common locations
        if [ -z "$CHECKPOINT_PATH" ]; then
            print_info "Searching for SAM3 checkpoint..."

            # Check common locations for SAM3 checkpoint
            SEARCH_PATHS=(
                "./sam3/checkpoints/sam3.pt"
                "$HOME/.cache/sam3/checkpoints/sam3.pt"
                "$HOME/.cache/sam3_deepstream/checkpoints/sam3.pt"
                "./checkpoints/sam3.pt"
                "../checkpoints/sam3.pt"
            )

            for path in "${SEARCH_PATHS[@]}"; do
                if [ -f "$path" ]; then
                    CHECKPOINT_PATH="$(realpath "$path")"
                    print_success "Found SAM3 checkpoint: $CHECKPOINT_PATH"
                    break
                fi
            done
        fi

        # If still no checkpoint, try to download from HuggingFace with authentication
        if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
            print_info "SAM3 checkpoint not found locally, attempting HuggingFace download..."
            CHECKPOINT_PATH="$HOME/.cache/sam3_deepstream/checkpoints/sam3.pt"
            mkdir -p "$(dirname "$CHECKPOINT_PATH")"

            # Download checkpoint using huggingface_hub with authentication
            uv run python << 'EOF'
import os
import sys

checkpoint_path = os.path.expanduser("~/.cache/sam3_deepstream/checkpoints/sam3.pt")

if os.path.exists(checkpoint_path):
    print(f"Checkpoint already exists: {checkpoint_path}")
    sys.exit(0)

# Check for HF token
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    # Try to get from huggingface-cli login
    try:
        from huggingface_hub import HfFolder
        hf_token = HfFolder.get_token()
    except:
        pass

if not hf_token:
    print("ERROR: HuggingFace authentication required for SAM3 model download.")
    print("Please either:")
    print("  1. Run: huggingface-cli login")
    print("  2. Set HF_TOKEN environment variable")
    print("  3. Provide checkpoint manually: --checkpoint /path/to/sam3.pt")
    sys.exit(1)

print("Downloading SAM3 checkpoint from HuggingFace (authenticated)...")
try:
    from huggingface_hub import hf_hub_download

    # Download SAM3 model (adjust repo_id to actual SAM3 repo)
    downloaded = hf_hub_download(
        repo_id="facebook/sam3-hiera-large",  # SAM3 repo
        filename="sam3_hiera_large.pt",
        local_dir=os.path.dirname(checkpoint_path),
        token=hf_token
    )
    # Rename to expected name if different
    if downloaded != checkpoint_path:
        os.rename(downloaded, checkpoint_path)
    print(f"Downloaded to: {checkpoint_path}")
except Exception as e:
    print(f"ERROR: Failed to download SAM3 checkpoint: {e}")
    print("Please provide checkpoint manually: --checkpoint /path/to/sam3.pt")
    sys.exit(1)
EOF
            if [ $? -ne 0 ]; then
                print_error "Failed to obtain SAM3 checkpoint"
                print_info "Please provide checkpoint manually: --checkpoint /path/to/sam3.pt"
                exit 1
            fi
            print_success "SAM3 checkpoint ready"
        fi

        if [ ! -f "$CHECKPOINT_PATH" ]; then
            print_error "Checkpoint not found: $CHECKPOINT_PATH"
            exit 1
        fi

        print_info "Exporting engines from: $CHECKPOINT_PATH"
        print_info "Output directory: $ENGINE_DIR"
        print_info "Precision: $PRECISION"
        print_info "Running TRT export inside Docker to match container TensorRT version..."

        # Get the directory containing the checkpoint for mounting
        CHECKPOINT_DIR="$(dirname "$CHECKPOINT_PATH")"
        CHECKPOINT_NAME="$(basename "$CHECKPOINT_PATH")"

        # Run export inside Docker container with GPU access
        # PYTHONPATH includes both /workspace and /workspace/sam3 for sam3 package
        docker run --rm \
            --runtime nvidia \
            --gpus all \
            -v "$CHECKPOINT_DIR:/workspace/checkpoints:ro" \
            -v "$ENGINE_DIR:/workspace/engines" \
            -v "$(pwd)/sam3:/workspace/sam3:ro" \
            -v "$(pwd)/sam3_deepstream:/workspace/sam3_deepstream:ro" \
            -e PYTHONPATH=/workspace:/workspace/sam3 \
            sam3-edge:jetson \
            python3 /workspace/sam3_deepstream/scripts/export_engines.py \
                --checkpoint "/workspace/checkpoints/$CHECKPOINT_NAME" \
                --output-dir /workspace/engines \
                --precision "$PRECISION"

        if [ $? -eq 0 ]; then
            print_success "TensorRT engines exported to $ENGINE_DIR"
            # Show TRT version used
            print_info "Engines built with Docker's TensorRT version (compatible with runtime)"
        else
            print_error "Failed to export TensorRT engines"
            exit 1
        fi
    fi
fi

# Push to DockerHub
if [ "$DOCKERHUB_PUSH" = true ]; then
    print_header "Pushing to DockerHub"

    if [ -z "$DOCKERHUB_REPO" ]; then
        print_error "--repo is required for --push"
        print_info "Usage: --push --repo username/sam3-edge"
        exit 1
    fi

    print_info "Logging into DockerHub..."
    docker login

    VERSION=$(grep 'version' sam3_deepstream/pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')

    # Tag images
    docker tag sam3-edge:jetson $DOCKERHUB_REPO:jetson-latest
    docker tag sam3-edge:jetson $DOCKERHUB_REPO:jetson-$VERSION

    # Push
    print_info "Pushing images..."
    docker push $DOCKERHUB_REPO:jetson-latest
    docker push $DOCKERHUB_REPO:jetson-$VERSION

    print_success "Images pushed to DockerHub"
    print_info "Pull with: docker pull $DOCKERHUB_REPO:jetson-latest"
fi

# Summary
print_header "Installation Complete!"

echo -e "${GREEN}SAM3-Edge is ready for Jetson AGX Orin${NC}\n"

echo "Quick Start:"
echo "─────────────────────────────────────────────────────────────"
echo ""
echo "1. Start the API server with Docker:"
echo -e "   ${CYAN}cd sam3_deepstream && docker compose -f docker-compose.jetson.yml up -d${NC}"
echo ""
echo "2. Check the health endpoint:"
echo -e "   ${CYAN}curl http://localhost:8000/health${NC}"
echo ""
echo "3. Submit a text prompt detection:"
echo -e "   ${CYAN}curl -X POST http://localhost:8000/api/v1/detect \\"
echo "     -F 'file=@video.mp4' \\"
echo -e "     -F 'text_prompt=green traffic lights'${NC}"
echo ""
echo "4. Search detected objects:"
echo -e "   ${CYAN}curl 'http://localhost:8000/api/v1/search?q=traffic+light'${NC}"
echo ""
echo "5. List all detected objects:"
echo -e "   ${CYAN}curl http://localhost:8000/api/v1/objects${NC}"
echo ""

echo -e "TensorRT engines location: ${CYAN}$ENGINE_DIR${NC}"
echo ""
echo "For more information, see README.md"
echo ""
print_success "Installation successful!"
