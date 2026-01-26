"""API endpoint tests for SAM3 FastAPI server.

Run with:
    pytest tests/test_api.py -v --api-host localhost --api-port 8000

Or against a running container:
    docker exec sam3_deepstream-sam3-api-1 pytest /workspace/tests/test_api.py -v
"""

import io
from pathlib import Path

import pytest

# Optional imports for type hints
try:
    import requests
except ImportError:
    requests = None


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self, api_session, api_base_url: str):
        """Health endpoint should return 200 status."""
        response = api_session.get(f"{api_base_url}/health")
        assert response.status_code == 200

    def test_health_response_schema(self, api_session, api_base_url: str):
        """Health response should match expected schema."""
        response = api_session.get(f"{api_base_url}/health")
        data = response.json()

        # Required fields
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "gpu_available" in data

        # Types
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["gpu_available"], bool)

    def test_health_reports_model_loaded(self, api_session, api_base_url: str):
        """Health should report whether model is loaded."""
        response = api_session.get(f"{api_base_url}/health")
        data = response.json()

        # Model should be loaded for healthy status
        if data["status"] == "healthy":
            assert data["model_loaded"] is True
        elif data["status"] == "degraded":
            assert data["model_loaded"] is False

    def test_health_reports_gpu_info(self, api_session, api_base_url: str):
        """Health should report GPU information when available."""
        response = api_session.get(f"{api_base_url}/health")
        data = response.json()

        if data["gpu_available"]:
            # Should have GPU details
            assert "gpu_name" in data
            assert data["gpu_name"] is not None or data["gpu_name"] == ""


class TestStatsEndpoint:
    """Test /stats endpoint."""

    def test_stats_returns_200(self, api_session, api_base_url: str):
        """Stats endpoint should return 200 status."""
        response = api_session.get(f"{api_base_url}/stats")
        assert response.status_code == 200

    def test_stats_response_schema(self, api_session, api_base_url: str):
        """Stats response should match expected schema."""
        response = api_session.get(f"{api_base_url}/stats")
        data = response.json()

        # Required fields
        assert "total_requests" in data
        assert "text_requests" in data
        assert "point_requests" in data
        assert "avg_inference_time_ms" in data
        assert "errors" in data

        # Types
        assert isinstance(data["total_requests"], int)
        assert isinstance(data["text_requests"], int)
        assert isinstance(data["point_requests"], int)
        assert isinstance(data["avg_inference_time_ms"], (int, float))
        assert isinstance(data["errors"], int)


@pytest.mark.skipif(requests is None, reason="requests not installed")
class TestTextSegmentation:
    """Test /api/v1/segment endpoint (text-based segmentation)."""

    def test_segment_requires_file(self, api_session, api_base_url: str):
        """Segment endpoint should require image file."""
        response = api_session.post(
            f"{api_base_url}/api/v1/segment",
            data={"text_prompt": "car"},
        )
        assert response.status_code == 422  # Unprocessable Entity

    def test_segment_requires_prompt(self, api_session, api_base_url: str, test_image_path: Path):
        """Segment endpoint should require text prompt."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 422

    def test_segment_racing_image_finds_cars(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Segmenting racing.jpeg with 'car' should find at least one car."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
            )

        assert response.status_code == 200
        data = response.json()

        # Should detect at least one car
        assert data["success"] is True
        assert data["num_detections"] >= 1, "Expected at least 1 car detection in racing.jpeg"

    def test_segment_returns_valid_detections(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Segment detections should have valid bbox format."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
            )

        assert response.status_code == 200
        data = response.json()

        if data["num_detections"] > 0:
            for detection in data["detections"]:
                # Validate bbox format [x1, y1, x2, y2]
                assert "bbox" in detection
                assert len(detection["bbox"]) == 4
                x1, y1, x2, y2 = detection["bbox"]

                # Coordinates should be reasonable (not negative, x2 > x1, etc.)
                assert x1 >= 0, f"x1 should be >= 0, got {x1}"
                assert y1 >= 0, f"y1 should be >= 0, got {y1}"
                assert x2 >= x1, f"x2 should be >= x1, got x1={x1}, x2={x2}"
                assert y2 >= y1, f"y2 should be >= y1, got y1={y1}, y2={y2}"

                # Score should be between 0 and 1
                assert "score" in detection
                assert 0 <= detection["score"] <= 1

    def test_segment_json_response_schema(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """JSON response should match SegmentResponse schema."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
            )

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "success" in data
        assert "inference_time_ms" in data
        assert "num_detections" in data
        assert "detections" in data

        # Types
        assert isinstance(data["success"], bool)
        assert isinstance(data["inference_time_ms"], (int, float))
        assert isinstance(data["num_detections"], int)
        assert isinstance(data["detections"], list)

    def test_segment_returns_image_overlay(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Without return_json, should return PNG image."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car"},
            )

        assert response.status_code == 200
        assert response.headers.get("content-type") == "image/png"

        # Should have custom headers
        assert "X-Inference-Time-Ms" in response.headers
        assert "X-Num-Detections" in response.headers

    def test_segment_inference_time_reasonable(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Inference time should be under 5 seconds."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
            )

        assert response.status_code == 200
        data = response.json()

        assert data["inference_time_ms"] < 5000, f"Inference took {data['inference_time_ms']}ms, expected < 5000ms"

    def test_segment_with_confidence_threshold(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Higher confidence threshold should return fewer/same detections."""
        # Low threshold
        with open(test_image_path, "rb") as f:
            response_low = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true", "confidence_threshold": "0.1"},
            )

        # High threshold
        with open(test_image_path, "rb") as f:
            response_high = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true", "confidence_threshold": "0.9"},
            )

        data_low = response_low.json()
        data_high = response_high.json()

        # High threshold should return <= low threshold detections
        assert data_high["num_detections"] <= data_low["num_detections"]


@pytest.mark.skipif(requests is None, reason="requests not installed")
class TestPointSegmentation:
    """Test /segment endpoint (point/box-based segmentation)."""

    def test_point_segment_center_point(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Segmenting with center point should return results."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                params={"points": "0.5,0.5,1", "return_json": "true"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_point_segment_returns_mask(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Point segmentation should return mask area > 0."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                params={"points": "0.5,0.5,1", "return_json": "true"},
            )

        assert response.status_code == 200
        data = response.json()

        if data["num_detections"] > 0:
            # At least one detection should have mask area > 0
            total_mask_area = sum(d.get("mask_area", 0) for d in data["detections"])
            assert total_mask_area > 0, "Expected non-zero mask area"

    def test_multiple_points_segment(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Multiple points should work."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                params={"points": "0.3,0.3,1;0.7,0.7,1", "return_json": "true"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_point_segment_returns_image(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Without return_json, should return PNG image with overlay."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                params={"points": "0.5,0.5,1"},
            )

        assert response.status_code == 200
        assert response.headers.get("content-type") == "image/png"


class TestStatsTracking:
    """Test that stats are tracked correctly."""

    def test_stats_increments_after_text_request(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Stats should increment after a text segmentation request."""
        # Get initial stats
        response = api_session.get(f"{api_base_url}/stats")
        initial_stats = response.json()

        # Make a text request
        with open(test_image_path, "rb") as f:
            api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "car", "return_json": "true"},
            )

        # Get updated stats
        response = api_session.get(f"{api_base_url}/stats")
        updated_stats = response.json()

        assert updated_stats["total_requests"] > initial_stats["total_requests"]
        assert updated_stats["text_requests"] > initial_stats["text_requests"]

    def test_stats_increments_after_point_request(self, api_session, api_base_url: str, test_image_path: Path, require_model_loaded):
        """Stats should increment after a point segmentation request."""
        # Get initial stats
        response = api_session.get(f"{api_base_url}/stats")
        initial_stats = response.json()

        # Make a point request
        with open(test_image_path, "rb") as f:
            api_session.post(
                f"{api_base_url}/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                params={"points": "0.5,0.5,1", "return_json": "true"},
            )

        # Get updated stats
        response = api_session.get(f"{api_base_url}/stats")
        updated_stats = response.json()

        assert updated_stats["total_requests"] > initial_stats["total_requests"]
        assert updated_stats["point_requests"] > initial_stats["point_requests"]


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_image_format(self, api_session, api_base_url: str):
        """Should handle invalid image format gracefully."""
        invalid_data = b"not an image"
        response = api_session.post(
            f"{api_base_url}/api/v1/segment",
            files={"file": ("test.jpg", io.BytesIO(invalid_data), "image/jpeg")},
            data={"text_prompt": "car", "return_json": "true"},
        )

        # Should return error, not crash
        # 503 is acceptable if model not loaded
        assert response.status_code in [400, 422, 500, 503]

    def test_empty_text_prompt(self, api_session, api_base_url: str, test_image_path: Path):
        """Empty text prompt should be handled."""
        with open(test_image_path, "rb") as f:
            response = api_session.post(
                f"{api_base_url}/api/v1/segment",
                files={"file": ("racing.jpeg", f, "image/jpeg")},
                data={"text_prompt": "", "return_json": "true"},
            )

        # May return 422 validation error or handle empty string
        assert response.status_code in [200, 400, 422]
