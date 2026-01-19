#!/bin/bash
# Start the SAM3 FastAPI server
#
# Usage:
#   ./scripts/start_api.sh
#   ./scripts/start_api.sh --port 8080
#   ./scripts/start_api.sh --workers 4
#
# Environment variables:
#   SAM3_ENCODER_ENGINE - Path to encoder TRT engine
#   SAM3_DECODER_ENGINE - Path to decoder TRT engine
#   CUDA_DEVICE - GPU device index (default: 0)
#   USE_ASYNC - Use async inference (default: true)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default engine paths if not provided
if [ -z "$SAM3_ENCODER_ENGINE" ]; then
    export SAM3_ENCODER_ENGINE="$PROJECT_DIR/engines/sam3_encoder.engine"
fi

if [ -z "$SAM3_DECODER_ENGINE" ]; then
    export SAM3_DECODER_ENGINE="$PROJECT_DIR/engines/sam3_decoder.engine"
fi

# Check if engines exist
if [ -f "$SAM3_ENCODER_ENGINE" ]; then
    echo "Encoder engine: $SAM3_ENCODER_ENGINE"
else
    echo "Warning: Encoder engine not found at $SAM3_ENCODER_ENGINE"
fi

if [ -f "$SAM3_DECODER_ENGINE" ]; then
    echo "Decoder engine: $SAM3_DECODER_ENGINE"
else
    echo "Warning: Decoder engine not found at $SAM3_DECODER_ENGINE"
fi

echo ""
echo "Starting SAM3 API server..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Log level: $LOG_LEVEL"
echo ""

# Run the server
cd "$PROJECT_DIR"
exec uvicorn sam3_deepstream.api.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    $RELOAD
