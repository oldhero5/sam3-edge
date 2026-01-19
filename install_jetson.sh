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

    print_info "Syncing workspace dependencies..."
    uv sync
    print_success "Python dependencies installed"

    # Initialize database
    print_header "Initializing Database"

    mkdir -p "$DB_DIR"
    mkdir -p "$ENGINE_DIR"

    print_info "Creating SQLite database and FAISS index..."

    uv run python3 << EOF
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, '.')

try:
    from sam3_deepstream.api.services.detection_store import DetectionStore
    import asyncio

    async def init():
        store = DetectionStore(
            db_path=Path("$DB_DIR/detections.db"),
            index_path=Path("$DB_DIR/embeddings.index"),
            use_gpu=True
        )
        await store.initialize()
        print("Database initialized successfully")

    asyncio.run(init())
except ImportError as e:
    print(f"Warning: Could not initialize database (will be created on first run): {e}")
EOF

    print_success "Database directory initialized at $DB_DIR"

    # Export TensorRT engines if checkpoint provided
    if [ -n "$CHECKPOINT_PATH" ]; then
        print_header "Exporting TensorRT Engines"

        if [ ! -f "$CHECKPOINT_PATH" ]; then
            print_error "Checkpoint not found: $CHECKPOINT_PATH"
            exit 1
        fi

        print_info "Exporting engines from: $CHECKPOINT_PATH"
        print_info "Output directory: $ENGINE_DIR"
        print_info "Precision: $PRECISION"

        uv run python -m sam3_deepstream.scripts.export_engines \
            --checkpoint "$CHECKPOINT_PATH" \
            --output-dir "$ENGINE_DIR" \
            --precision "$PRECISION"

        print_success "TensorRT engines exported to $ENGINE_DIR"
    else
        print_info "No checkpoint provided, skipping TensorRT export"
        print_info "To export later: ./install_jetson.sh --checkpoint /path/to/sam3.pt"
    fi
fi

# Build Docker image
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
echo "   ${CYAN}cd sam3_deepstream && docker compose -f docker-compose.jetson.yml up -d${NC}"
echo ""
echo "2. Check the health endpoint:"
echo "   ${CYAN}curl http://localhost:8000/health${NC}"
echo ""
echo "3. Submit a text prompt detection:"
echo "   ${CYAN}curl -X POST http://localhost:8000/api/v1/detect \\"
echo "     -F 'file=@video.mp4' \\"
echo "     -F 'text_prompt=green traffic lights'${NC}"
echo ""
echo "4. Search detected objects:"
echo "   ${CYAN}curl 'http://localhost:8000/api/v1/search?q=traffic+light'${NC}"
echo ""
echo "5. List all detected objects:"
echo "   ${CYAN}curl http://localhost:8000/api/v1/objects${NC}"
echo ""

if [ -z "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}Note:${NC} TensorRT engines not exported."
    echo "For optimal performance, export engines with:"
    echo "   ${CYAN}./install_jetson.sh --checkpoint /path/to/sam3_checkpoint.pt${NC}"
    echo ""
fi

echo "For more information, see README.md"
echo ""
print_success "Installation successful!"
