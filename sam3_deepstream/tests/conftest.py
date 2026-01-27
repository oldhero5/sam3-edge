"""Pytest fixtures for SAM3 API and export testing.

This is a GPU-first project. Tests requiring CUDA will FAIL (not skip)
if GPU is not available. Run tests inside Docker for proper GPU access:

    ./scripts/run_tests.sh

Or use the Makefile:

    make test
"""

import os
from pathlib import Path
from typing import Generator, Optional

import pytest


# API configuration
DEFAULT_API_HOST = os.environ.get("SAM3_API_HOST", "localhost")
DEFAULT_API_PORT = int(os.environ.get("SAM3_API_PORT", "8000"))

# Environment flag to allow CPU-only testing (for CI without GPU)
ALLOW_CPU_TESTS = os.environ.get("SAM3_ALLOW_CPU_TESTS", "0") == "1"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU/CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def _check_cuda_available() -> tuple[bool, str]:
    """Check if CUDA is available and return status with message."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA available: {device_name}"
        else:
            return False, "CUDA not available (torch.cuda.is_available() = False)"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"CUDA check failed: {e}"


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available. Returns True/False without failing."""
    available, _ = _check_cuda_available()
    return available


@pytest.fixture(scope="session")
def require_cuda():
    """Require CUDA to be available. FAILS if not available.

    This is a GPU-first project. If you need to run without GPU,
    set environment variable: SAM3_ALLOW_CPU_TESTS=1
    """
    available, message = _check_cuda_available()

    if not available and not ALLOW_CPU_TESTS:
        pytest.fail(
            f"{message}\n\n"
            "This is a GPU-first project. Tests must run with GPU access.\n"
            "Run tests inside Docker: ./scripts/run_tests.sh\n"
            "Or use: make test\n\n"
            "To force CPU testing (not recommended): SAM3_ALLOW_CPU_TESTS=1"
        )

    return available


def pytest_addoption(parser):
    """Add custom command-line options for API testing."""
    parser.addoption(
        "--api-host",
        action="store",
        default=DEFAULT_API_HOST,
        help="API host to test against",
    )
    parser.addoption(
        "--api-port",
        action="store",
        default=DEFAULT_API_PORT,
        type=int,
        help="API port to test against",
    )
    parser.addoption(
        "--skip-api",
        action="store_true",
        default=False,
        help="Skip API tests (requires running server)",
    )


@pytest.fixture(scope="session")
def api_host(request) -> str:
    """Get API host from command line or environment."""
    return request.config.getoption("--api-host")


@pytest.fixture(scope="session")
def api_port(request) -> int:
    """Get API port from command line or environment."""
    return request.config.getoption("--api-port")


@pytest.fixture(scope="session")
def api_base_url(api_host: str, api_port: int) -> str:
    """Get base URL for API requests."""
    return f"http://{api_host}:{api_port}"


@pytest.fixture(scope="session")
def skip_api_tests(request) -> bool:
    """Check if API tests should be skipped."""
    return request.config.getoption("--skip-api")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get test data directory."""
    # Check multiple possible locations (for running in container vs locally)
    possible_paths = [
        Path("/workspace/test_data"),  # Container mount path
        project_root / "test_data",  # Local development path
        Path(__file__).parent.parent.parent / "test_data",  # Relative to sam3_deepstream
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return project_root / "test_data"  # Return default for skip message


@pytest.fixture(scope="session")
def test_image_path(test_data_dir: Path) -> Path:
    """Get path to test image (racing.jpeg)."""
    path = test_data_dir / "racing.jpeg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path


@pytest.fixture(scope="session")
def test_video_path(test_data_dir: Path) -> Path:
    """Get path to test video (Target.mp4)."""
    path = test_data_dir / "Target.mp4"
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    return path


@pytest.fixture(scope="session")
def checkpoint_dir(project_root: Path) -> Path:
    """Get checkpoints directory."""
    return project_root / "sam3" / "checkpoints"


@pytest.fixture(scope="session")
def engine_dir() -> Path:
    """Get TensorRT engines directory.

    Checks both container path and local path.
    FAILS if engines directory doesn't exist or is empty.
    """
    # Check container path first, then local
    possible_paths = [
        Path("/workspace/sam3_deepstream/engines"),  # Container mount
        Path(__file__).parent.parent / "engines",  # Local
    ]

    for path in possible_paths:
        if path.exists() and list(path.glob("*.engine")):
            return path

    # No engines found - fail with helpful message
    local_path = Path(__file__).parent.parent / "engines"
    pytest.fail(
        f"TensorRT engines not found in {local_path}\n\n"
        "Build engines first:\n"
        "  make export\n\n"
        "Or run: ./install_jetson.sh --checkpoint /path/to/sam3.pt"
    )


@pytest.fixture(scope="session")
def api_session(api_base_url: str, skip_api_tests: bool):
    """Create a requests session for API testing."""
    if skip_api_tests:
        pytest.skip("API tests skipped via --skip-api flag")

    try:
        import requests
    except ImportError:
        pytest.skip("requests library not installed")

    session = requests.Session()
    session.base_url = api_base_url

    # Test connectivity
    try:
        response = session.get(f"{api_base_url}/health", timeout=10)
        if response.status_code != 200:
            pytest.skip(f"API not healthy: {response.status_code}")
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Cannot connect to API at {api_base_url}")
    except requests.exceptions.Timeout:
        pytest.skip(f"API connection timeout at {api_base_url}")

    yield session

    session.close()


@pytest.fixture
def api_health_check(api_session, api_base_url: str) -> dict:
    """Get health check response from API."""
    response = api_session.get(f"{api_base_url}/health")
    return response.json()


@pytest.fixture
def require_model_loaded(api_session, api_base_url: str):
    """Skip test if model is not loaded."""
    response = api_session.get(f"{api_base_url}/health")
    data = response.json()
    if not data.get("model_loaded"):
        pytest.skip("Model not loaded - skipping test that requires inference")
