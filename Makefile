# SAM3-Edge Development Makefile
# Standard commands for development workflow

.PHONY: help test test-quick test-api lint format typecheck docker-build docker-up docker-down clean

help:
	@echo "SAM3-Edge Development Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run full test suite in Docker (with GPU)"
	@echo "  make test-quick    Run quick tests without Docker (no GPU)"
	@echo "  make test-api      Run API tests against running server"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Format code with ufmt (Black + usort)"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make check         Run all quality checks (lint + typecheck)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build sam3-edge:jetson Docker image"
	@echo "  make docker-up     Start API server in Docker"
	@echo "  make docker-down   Stop API server"
	@echo ""
	@echo "Other:"
	@echo "  make clean         Remove build artifacts and caches"
	@echo "  make export        Export TensorRT engines (inside Docker)"

# =============================================================================
# Testing
# =============================================================================

# Run full test suite in Docker with GPU
test:
	./scripts/run_tests.sh

# Run quick tests without Docker (platform tests only)
test-quick:
	uv run pytest sam3/tests/test_platform.py -v

# Run API tests against a running server
test-api:
	uv run pytest sam3_deepstream/tests/test_api.py -v --api-host localhost --api-port 8000

# Run specific test file in Docker
test-%:
	./scripts/run_tests.sh tests/$*.py

# =============================================================================
# Code Quality
# =============================================================================

# Format code with ufmt (Black + usort)
format:
	uv run ufmt format sam3 sam3_deepstream

# Run ruff linter
lint:
	uv run ruff check sam3 sam3_deepstream

# Fix auto-fixable lint issues
lint-fix:
	uv run ruff check --fix sam3 sam3_deepstream

# Run mypy type checker
typecheck:
	uv run mypy sam3 sam3_deepstream --config-file sam3/pyproject.toml

# Run all quality checks
check: lint typecheck
	@echo "All quality checks passed."

# =============================================================================
# Docker
# =============================================================================

# Build Docker image for Jetson
docker-build:
	cd sam3_deepstream && docker build -f Dockerfile.jetson -t sam3-edge:jetson .

# Start API server in Docker
docker-up:
	cd sam3_deepstream && docker compose -f docker-compose.jetson.yml up -d

# Stop API server
docker-down:
	cd sam3_deepstream && docker compose -f docker-compose.jetson.yml down

# View logs
docker-logs:
	cd sam3_deepstream && docker compose -f docker-compose.jetson.yml logs -f

# =============================================================================
# TensorRT Export
# =============================================================================

# Export TensorRT engines inside Docker
export:
	docker run --rm \
		--runtime nvidia \
		--gpus all \
		-v "$$(pwd)/sam3/checkpoints:/workspace/checkpoints:ro" \
		-v "$$(pwd)/sam3_deepstream/engines:/workspace/engines" \
		-v "$$(pwd)/sam3:/workspace/sam3:ro" \
		-v "$$(pwd)/sam3_deepstream:/workspace/sam3_deepstream:ro" \
		-e PYTHONPATH=/workspace:/workspace/sam3 \
		sam3-edge:jetson \
		python3 /workspace/sam3_deepstream/scripts/export_engines.py \
			--checkpoint /workspace/checkpoints/sam3.pt \
			--output-dir /workspace/engines \
			--precision fp16

# =============================================================================
# Cleanup
# =============================================================================

# Remove build artifacts and caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts."
