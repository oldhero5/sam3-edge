#!/bin/bash
# SAM3-Edge Installation Script
# Handles dependency installation, TensorRT export, and Docker builds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
INSTALL_DEV=false
EXPORT_ENGINES=false
BUILD_DOCKER=false
CHECKPOINT_PATH=""
ENGINE_DIR="$HOME/.cache/sam3_deepstream/engines"
PRECISION="fp16"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --export)
            EXPORT_ENGINES=true
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
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --docker)
            BUILD_DOCKER=true
            shift
            ;;
        --all)
            INSTALL_DEV=true
            BUILD_DOCKER=true
            shift
            ;;
        -h|--help)
            echo "SAM3-Edge Installation Script"
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev           Install development dependencies (pytest, etc.)"
            echo "  --export        Export TensorRT engines (requires --checkpoint)"
            echo "  --checkpoint    Path to SAM3 checkpoint file"
            echo "  --engine-dir    Output directory for engines (default: ~/.cache/sam3_deepstream/engines)"
            echo "  --precision     TensorRT precision: fp16, fp32, int8 (default: fp16)"
            echo "  --docker        Build Docker image for Jetson"
            echo "  --all           Install dev deps and build Docker"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./install.sh                              # Basic installation"
            echo "  ./install.sh --dev                        # With dev dependencies"
            echo "  ./install.sh --export --checkpoint ~/models/sam3.pt"
            echo "  ./install.sh --docker                     # Build Docker image"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "SAM3-Edge Installation"

# Check prerequisites
print_header "Checking Prerequisites"

# Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 1 ]]; then
        print_success "Python $PYTHON_VERSION"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found"
    exit 1
fi

# UV package manager
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version | head -1)
    print_success "UV: $UV_VERSION"
else
    print_warning "UV not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "UV installed"
fi

# CUDA (optional but recommended)
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    print_success "CUDA $CUDA_VERSION"
else
    print_warning "CUDA not found (required for GPU inference)"
fi

# Check for Jetson
if [ -f /etc/nv_tegra_release ]; then
    JETSON_VERSION=$(cat /etc/nv_tegra_release | head -1)
    print_success "Jetson detected: $JETSON_VERSION"
    IS_JETSON=true
else
    IS_JETSON=false
fi

# Install dependencies
print_header "Installing Dependencies"

if [ "$INSTALL_DEV" = true ]; then
    echo "Installing with dev dependencies..."
    uv sync --group dev
else
    echo "Installing production dependencies..."
    uv sync
fi

print_success "Dependencies installed"

# Export TensorRT engines
if [ "$EXPORT_ENGINES" = true ]; then
    print_header "Exporting TensorRT Engines"

    if [ -z "$CHECKPOINT_PATH" ]; then
        print_error "--checkpoint is required for --export"
        exit 1
    fi

    if [ ! -f "$CHECKPOINT_PATH" ]; then
        print_error "Checkpoint not found: $CHECKPOINT_PATH"
        exit 1
    fi

    mkdir -p "$ENGINE_DIR"

    echo "Exporting engines from: $CHECKPOINT_PATH"
    echo "Output directory: $ENGINE_DIR"
    echo "Precision: $PRECISION"

    uv run python -m sam3_deepstream.scripts.export_engines \
        --checkpoint "$CHECKPOINT_PATH" \
        --output-dir "$ENGINE_DIR" \
        --precision "$PRECISION"

    print_success "TensorRT engines exported to $ENGINE_DIR"
fi

# Run tests
if [ "$INSTALL_DEV" = true ]; then
    print_header "Running Tests"

    if [ -d "sam3_deepstream/tests" ]; then
        uv run pytest sam3_deepstream/tests/ -v --tb=short || {
            print_warning "Some tests failed (this may be expected without GPU/engines)"
        }
    else
        print_warning "No tests found"
    fi
fi

# Build Docker image
if [ "$BUILD_DOCKER" = true ]; then
    print_header "Building Docker Image"

    cd sam3_deepstream

    if [ "$IS_JETSON" = true ]; then
        echo "Building Jetson Docker image..."
        docker build -f Dockerfile.jetson -t sam3-edge:jetson .
        print_success "Docker image built: sam3-edge:jetson"
    else
        echo "Building x86 Docker image..."
        docker build -t sam3-edge:latest .
        print_success "Docker image built: sam3-edge:latest"
    fi

    cd ..
fi

# Summary
print_header "Installation Complete"

echo "Next steps:"
echo ""
echo "1. Start the API server:"
echo "   uv run sam3-api"
echo ""
if [ "$EXPORT_ENGINES" = false ]; then
    echo "2. Export TensorRT engines (if not done):"
    echo "   ./install.sh --export --checkpoint /path/to/sam3_checkpoint.pt"
    echo ""
fi
if [ "$BUILD_DOCKER" = true ]; then
    echo "3. Run with Docker:"
    if [ "$IS_JETSON" = true ]; then
        echo "   cd sam3_deepstream && docker compose -f docker-compose.jetson.yml up"
    else
        echo "   cd sam3_deepstream && docker compose up"
    fi
    echo ""
fi

echo "For API usage, see README.md"
print_success "SAM3-Edge is ready!"
