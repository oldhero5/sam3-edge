"""Pydantic models for API requests and responses."""

from .requests import VideoProcessRequest, BoundingBox, PointPrompt
from .responses import (
    VideoProcessingResponse,
    JobStatusResponse,
    SegmentationResult,
    RLEMask,
)

__all__ = [
    "VideoProcessRequest",
    "BoundingBox",
    "PointPrompt",
    "VideoProcessingResponse",
    "JobStatusResponse",
    "SegmentationResult",
    "RLEMask",
]
