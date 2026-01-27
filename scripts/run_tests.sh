#!/bin/bash
# SAM3-Edge Test Runner
# Runs tests inside Docker container with GPU access
#
# Usage:
#   ./scripts/run_tests.sh                    # Run all tests
#   ./scripts/run_tests.sh tests/test_api.py  # Run specific test file
#   ./scripts/run_tests.sh -k "test_health"   # Run tests matching pattern
#   ./scripts/run_tests.sh --quick            # Run quick tests only (no GPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check for --quick flag (runs without Docker)
if [[ "$1" == "--quick" ]]; then
    shift
    echo -e "${CYAN}Running quick tests (no GPU required)...${NC}"
    cd "$PROJECT_ROOT"
    uv run pytest sam3/tests/test_platform.py -v "$@"
    exit $?
fi

# Check Docker image exists
if ! docker images sam3-edge:jetson --format "{{.Repository}}" | grep -q "sam3-edge"; then
    echo -e "${RED}Docker image 'sam3-edge:jetson' not found.${NC}"
    echo -e "${YELLOW}Build it with: cd sam3_deepstream && docker build -f Dockerfile.jetson -t sam3-edge:jetson .${NC}"
    exit 1
fi

# Check NVIDIA runtime available
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime may not be configured.${NC}"
fi

echo -e "${CYAN}Running tests inside Docker with GPU access...${NC}"

# Default test path if none specified
TEST_PATH="${1:-/workspace/sam3_deepstream/tests/}"
shift 2>/dev/null || true

# Run tests in Docker container
docker run --rm \
    --runtime nvidia \
    --gpus all \
    -v "$PROJECT_ROOT/sam3:/workspace/sam3:ro" \
    -v "$PROJECT_ROOT/sam3_deepstream:/workspace/sam3_deepstream:ro" \
    -v "$PROJECT_ROOT/test_data:/workspace/test_data:ro" \
    -e PYTHONPATH=/workspace:/workspace/sam3 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -w /workspace/sam3_deepstream \
    sam3-edge:jetson \
    pytest "$TEST_PATH" -v "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed.${NC}"
else
    echo -e "\n${RED}Some tests failed.${NC}"
fi

exit $EXIT_CODE
