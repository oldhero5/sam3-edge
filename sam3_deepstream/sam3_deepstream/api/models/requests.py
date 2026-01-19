"""Request models for SAM3 DeepStream API."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    """Output format for processed videos."""
    VIDEO = "video"  # MP4 with mask overlay
    MASKS = "masks"  # JSON with RLE-encoded masks


class BoundingBox(BaseModel):
    """Bounding box prompt."""
    x: float = Field(ge=0.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(ge=0.0, le=1.0, description="Normalized y coordinate")
    width: float = Field(ge=0.0, le=1.0, description="Normalized width")
    height: float = Field(ge=0.0, le=1.0, description="Normalized height")
    label: int = Field(default=1, description="Box label (1 for foreground)")


class PointPrompt(BaseModel):
    """Point prompt for segmentation."""
    x: float = Field(ge=0.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(ge=0.0, le=1.0, description="Normalized y coordinate")
    label: int = Field(default=1, description="Point label (1=foreground, 0=background)")


class VideoProcessRequest(BaseModel):
    """Request to process a video."""
    text_prompt: Optional[str] = Field(
        default=None,
        description="Text description of objects to segment",
    )
    box_prompts: Optional[List[BoundingBox]] = Field(
        default=None,
        description="Bounding box prompts for specific regions",
    )
    point_prompts: Optional[List[PointPrompt]] = Field(
        default=None,
        description="Point prompts for specific locations",
    )
    keyframe_interval: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Frames between full inference",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.VIDEO,
        description="Output format (video with overlay or mask JSON)",
    )
    max_objects: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum objects to track",
    )
    segmentation_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mask confidence threshold",
    )


class StreamFrameRequest(BaseModel):
    """Request for WebSocket frame processing."""
    frame_data: str = Field(description="Base64-encoded frame data")
    frame_idx: int = Field(ge=0, description="Frame index in video")
    prompts: Optional[dict] = Field(
        default=None,
        description="Optional prompts for this frame",
    )
