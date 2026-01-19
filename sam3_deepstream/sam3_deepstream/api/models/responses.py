"""Response models for SAM3 DeepStream API."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RLEMask(BaseModel):
    """Run-length encoded mask."""
    object_id: int = Field(description="Unique object identifier")
    counts: str = Field(description="RLE counts string")
    size: Tuple[int, int] = Field(description="Mask dimensions (height, width)")
    score: float = Field(description="Confidence score")
    box: Tuple[float, float, float, float] = Field(
        description="Normalized bounding box (x1, y1, x2, y2)"
    )
    text_prompt: Optional[str] = Field(
        default=None,
        description="Text prompt used for detection"
    )
    label: Optional[str] = Field(
        default=None,
        description="Detected object label"
    )


class SegmentationResult(BaseModel):
    """Segmentation result for a single frame."""
    frame_idx: int = Field(description="Frame index")
    is_keyframe: bool = Field(description="Whether this was a keyframe")
    masks: List[RLEMask] = Field(description="Detected masks")
    latency_ms: float = Field(description="Processing latency in milliseconds")


class VideoProcessingResponse(BaseModel):
    """Response for video processing request."""
    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    message: str = Field(default="", description="Status message")


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Processing progress (0-1)",
    )
    frames_processed: Optional[int] = Field(
        default=None,
        description="Number of frames processed",
    )
    total_frames: Optional[int] = Field(
        default=None,
        description="Total frames in video",
    )
    created_at: datetime = Field(description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Job completion timestamp",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed",
    )
    detection_count: Optional[int] = Field(
        default=None,
        description="Number of detections stored",
    )
    text_prompt: Optional[str] = Field(
        default=None,
        description="Text prompt used for detection",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    engines_available: bool = Field(description="Whether TRT engines are loaded")
    gpu_available: bool = Field(description="Whether GPU is available")
    deepstream_available: bool = Field(description="Whether DeepStream is available")


class EngineInfoResponse(BaseModel):
    """TensorRT engine information."""
    encoder: dict = Field(description="Encoder engine info")
    decoder: dict = Field(description="Decoder engine info")


class StreamResultMessage(BaseModel):
    """WebSocket message for stream results."""
    type: str = Field(default="result", description="Message type")
    frame_idx: int = Field(description="Frame index")
    latency_ms: float = Field(description="Processing latency")
    masks: List[RLEMask] = Field(description="Segmentation masks")
