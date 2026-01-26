#!/usr/bin/env python3
"""
Standalone API test runner for SAM3 FastAPI service.

Runs a suite of API tests against a running SAM3 container without
requiring pytest. Uses only requests library for minimal dependencies.

Usage:
    python run_api_tests.py [--host HOST] [--port PORT] [--image PATH]

Exit codes:
    0 - All tests passed
    1 - One or more tests failed
    2 - Cannot connect to API
    3 - Test image not found

Examples:
    # Test against local container
    python run_api_tests.py

    # Test against custom host/port
    python run_api_tests.py --host 192.168.1.100 --port 8080

    # Use custom test image
    python run_api_tests.py --image /path/to/image.jpg
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(2)


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    """Add ANSI color to text."""
    return f"{color}{text}{Colors.RESET}"


class TestResult:
    """Result of a single test."""

    def __init__(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


class APITester:
    """Test runner for SAM3 API."""

    def __init__(self, base_url: str, test_image_path: Optional[Path] = None):
        self.base_url = base_url
        self.test_image_path = test_image_path
        self.session = requests.Session()
        self.results: List[TestResult] = []

    def run_test(self, name: str, test_fn: Callable[[], Tuple[bool, str]]) -> TestResult:
        """Run a single test and record result."""
        start = time.perf_counter()
        try:
            passed, message = test_fn()
        except Exception as e:
            passed = False
            message = f"Exception: {e}"
        duration_ms = (time.perf_counter() - start) * 1000

        result = TestResult(name, passed, message, duration_ms)
        self.results.append(result)
        return result

    def print_result(self, result: TestResult):
        """Print a single test result."""
        status = colorize("PASS", Colors.GREEN) if result.passed else colorize("FAIL", Colors.RED)
        print(f"  [{status}] {result.name} ({result.duration_ms:.1f}ms)")
        if not result.passed and result.message:
            print(f"         {colorize(result.message, Colors.YELLOW)}")

    # =========================================================================
    # Health Tests
    # =========================================================================

    def test_health_endpoint(self) -> Tuple[bool, str]:
        """Test /health endpoint returns 200."""
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        if response.status_code != 200:
            return False, f"Expected 200, got {response.status_code}"
        return True, ""

    def test_health_model_loaded(self) -> Tuple[bool, str]:
        """Test that model is loaded."""
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        data = response.json()
        if not data.get("model_loaded"):
            return False, f"Model not loaded. Status: {data.get('status')}"
        return True, f"Model loaded, status: {data.get('status')}"

    def test_health_gpu_available(self) -> Tuple[bool, str]:
        """Test that GPU is available."""
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        data = response.json()
        if not data.get("gpu_available"):
            return False, "GPU not available"
        gpu_name = data.get("gpu_name", "unknown")
        return True, f"GPU: {gpu_name}"

    # =========================================================================
    # Stats Tests
    # =========================================================================

    def test_stats_endpoint(self) -> Tuple[bool, str]:
        """Test /stats endpoint returns 200."""
        response = self.session.get(f"{self.base_url}/stats", timeout=10)
        if response.status_code != 200:
            return False, f"Expected 200, got {response.status_code}"
        return True, ""

    # =========================================================================
    # Segmentation Tests
    # =========================================================================

    def test_text_segmentation(self) -> Tuple[bool, str]:
        """Test text-based segmentation with 'car' prompt."""
        if not self.test_image_path or not self.test_image_path.exists():
            return False, "Test image not found"

        with open(self.test_image_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/api/v1/segment",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
                timeout=60,
            )

        if response.status_code != 200:
            return False, f"Expected 200, got {response.status_code}: {response.text[:200]}"

        data = response.json()
        if not data.get("success"):
            return False, "Response success=false"

        num_detections = data.get("num_detections", 0)
        if num_detections < 1:
            return False, f"Expected >= 1 detection, got {num_detections}"

        inference_time = data.get("inference_time_ms", 0)
        return True, f"Found {num_detections} car(s) in {inference_time:.1f}ms"

    def test_segmentation_bbox_format(self) -> Tuple[bool, str]:
        """Test that bounding boxes have valid format."""
        if not self.test_image_path or not self.test_image_path.exists():
            return False, "Test image not found"

        with open(self.test_image_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/api/v1/segment",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
                timeout=60,
            )

        data = response.json()
        detections = data.get("detections", [])

        if not detections:
            return True, "No detections to validate"

        for i, det in enumerate(detections):
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                return False, f"Detection {i}: bbox has {len(bbox)} elements, expected 4"

            x1, y1, x2, y2 = bbox
            if x1 < 0 or y1 < 0:
                return False, f"Detection {i}: negative coordinates ({x1}, {y1})"
            if x2 < x1 or y2 < y1:
                return False, f"Detection {i}: invalid bbox (x2 < x1 or y2 < y1)"

            score = det.get("score", -1)
            if not (0 <= score <= 1):
                return False, f"Detection {i}: score {score} not in [0, 1]"

        return True, f"Validated {len(detections)} detection(s)"

    def test_segmentation_inference_time(self) -> Tuple[bool, str]:
        """Test that inference time is reasonable (< 5s)."""
        if not self.test_image_path or not self.test_image_path.exists():
            return False, "Test image not found"

        with open(self.test_image_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/api/v1/segment",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
                timeout=60,
            )

        data = response.json()
        inference_time = data.get("inference_time_ms", 0)

        if inference_time > 5000:
            return False, f"Inference took {inference_time:.1f}ms, expected < 5000ms"

        return True, f"Inference time: {inference_time:.1f}ms"

    def test_image_overlay_response(self) -> Tuple[bool, str]:
        """Test that image overlay response returns PNG."""
        if not self.test_image_path or not self.test_image_path.exists():
            return False, "Test image not found"

        with open(self.test_image_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/api/v1/segment",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"text_prompt": "car"},
                timeout=60,
            )

        if response.status_code != 200:
            return False, f"Expected 200, got {response.status_code}"

        content_type = response.headers.get("content-type", "")
        if content_type != "image/png":
            return False, f"Expected image/png, got {content_type}"

        # Check for custom headers
        if "X-Inference-Time-Ms" not in response.headers:
            return False, "Missing X-Inference-Time-Ms header"

        return True, f"PNG response with inference time header"

    def test_point_segmentation(self) -> Tuple[bool, str]:
        """Test point-based segmentation."""
        if not self.test_image_path or not self.test_image_path.exists():
            return False, "Test image not found"

        with open(self.test_image_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/segment",
                files={"file": ("test.jpg", f, "image/jpeg")},
                params={"points": "0.5,0.5,1", "return_json": "true"},
                timeout=60,
            )

        if response.status_code != 200:
            return False, f"Expected 200, got {response.status_code}"

        data = response.json()
        if not data.get("success"):
            return False, "Response success=false"

        return True, f"Point segmentation successful, {data.get('num_detections', 0)} detection(s)"

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all_tests(self) -> bool:
        """Run all tests and return True if all passed."""
        print(colorize("\n=== SAM3 API Tests ===\n", Colors.BOLD))
        print(f"Target: {self.base_url}")
        if self.test_image_path:
            print(f"Test image: {self.test_image_path}")
        print()

        # Health tests
        print(colorize("Health Checks:", Colors.BLUE))
        self.print_result(self.run_test("Health endpoint returns 200", self.test_health_endpoint))
        self.print_result(self.run_test("Model is loaded", self.test_health_model_loaded))
        self.print_result(self.run_test("GPU is available", self.test_health_gpu_available))
        print()

        # Stats tests
        print(colorize("Stats Endpoint:", Colors.BLUE))
        self.print_result(self.run_test("Stats endpoint returns 200", self.test_stats_endpoint))
        print()

        # Segmentation tests (only if test image available)
        if self.test_image_path and self.test_image_path.exists():
            print(colorize("Text Segmentation:", Colors.BLUE))
            self.print_result(self.run_test("Text segmentation finds cars", self.test_text_segmentation))
            self.print_result(self.run_test("Bounding boxes valid format", self.test_segmentation_bbox_format))
            self.print_result(self.run_test("Inference time < 5s", self.test_segmentation_inference_time))
            self.print_result(self.run_test("Image overlay returns PNG", self.test_image_overlay_response))
            print()

            print(colorize("Point Segmentation:", Colors.BLUE))
            self.print_result(self.run_test("Point segmentation works", self.test_point_segmentation))
            print()
        else:
            print(colorize("Skipping segmentation tests (no test image)", Colors.YELLOW))
            print()

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print(colorize("=== Summary ===", Colors.BOLD))
        print(f"  Passed: {colorize(str(passed), Colors.GREEN)}")
        print(f"  Failed: {colorize(str(failed), Colors.RED if failed else Colors.GREEN)}")
        print(f"  Total:  {total}")
        print()

        if failed == 0:
            print(colorize("All tests passed!", Colors.GREEN + Colors.BOLD))
        else:
            print(colorize(f"{failed} test(s) failed", Colors.RED + Colors.BOLD))

        return failed == 0


def wait_for_api(base_url: str, timeout: int = 120) -> bool:
    """Wait for API to become available."""
    print(f"Waiting for API at {base_url}...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("model_loaded"):
                    print(colorize("API ready!", Colors.GREEN))
                    return True
                else:
                    print(f"  API responding but model not loaded yet... ({data.get('status')})")
            else:
                print(f"  API returned {response.status_code}...")
        except requests.exceptions.ConnectionError:
            print("  Connection refused, retrying...")
        except requests.exceptions.Timeout:
            print("  Request timeout, retrying...")

        time.sleep(5)

    print(colorize("Timeout waiting for API", Colors.RED))
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run API tests against SAM3 FastAPI service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--image", type=Path, help="Path to test image")
    parser.add_argument("--wait", type=int, default=0, help="Wait N seconds for API to start (0=no wait)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        Colors.GREEN = ""
        Colors.RED = ""
        Colors.YELLOW = ""
        Colors.BLUE = ""
        Colors.RESET = ""
        Colors.BOLD = ""

    base_url = f"http://{args.host}:{args.port}"

    # Find test image
    test_image = args.image
    if not test_image:
        # Try default locations
        candidates = [
            Path(__file__).parent.parent.parent / "test_data" / "racing.jpeg",
            Path("/workspace/test_data/racing.jpeg"),
            Path("test_data/racing.jpeg"),
        ]
        for candidate in candidates:
            if candidate.exists():
                test_image = candidate
                break

    if test_image and not test_image.exists():
        print(colorize(f"Test image not found: {test_image}", Colors.RED))
        sys.exit(3)

    # Wait for API if requested
    if args.wait > 0:
        if not wait_for_api(base_url, timeout=args.wait):
            sys.exit(2)
    else:
        # Quick connectivity check
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
        except requests.exceptions.ConnectionError:
            print(colorize(f"Cannot connect to API at {base_url}", Colors.RED))
            print("Is the container running? Try: docker ps")
            sys.exit(2)
        except requests.exceptions.Timeout:
            print(colorize(f"Connection timeout to {base_url}", Colors.RED))
            sys.exit(2)

    # Run tests
    tester = APITester(base_url, test_image)
    all_passed = tester.run_all_tests()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
