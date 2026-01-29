"""Global configuration for SAM3 DeepStream."""

import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class OutputFormat(str, Enum):
    """Output format options for processed videos."""
    VIDEO = "video"  # MP4 with mask overlay
    MASKS = "masks"  # JSON with RLE-encoded masks


class Precision(str, Enum):
    """TensorRT precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass
class TRTConfig:
    """TensorRT engine configuration."""
    precision: Precision = Precision.FP16
    workspace_size_gb: int = 4
    use_dla: bool = False
    dla_core: int = 0
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "sam3_deepstream" / "engines"
    )

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Inference runtime configuration."""
    input_size: int = 1008  # SAM3 input resolution
    keyframe_interval: int = 5  # Run full inference every N frames
    max_objects: int = 50  # Maximum tracked objects per stream
    segmentation_threshold: float = 0.5
    batch_size: int = 1


@dataclass
class DeepStreamConfig:
    """DeepStream pipeline configuration."""
    gpu_id: int = 0
    streammux_width: int = 1920
    streammux_height: int = 1080
    batched_push_timeout: int = 40000  # microseconds
    tracker_type: str = "NvDCF"


@dataclass
class APIConfig:
    """FastAPI server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_jobs: int = 4
    upload_dir: Path = field(
        default_factory=lambda: Path("/tmp/sam3_deepstream/uploads")
    )
    output_dir: Path = field(
        default_factory=lambda: Path("/tmp/sam3_deepstream/outputs")
    )
    max_upload_size_mb: int = 500

    def __post_init__(self):
        self.upload_dir = Path(self.upload_dir)
        self.output_dir = Path(self.output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DatabaseConfig:
    """Database configuration for detection storage."""
    db_path: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "sam3_deepstream" / "db" / "detections.db"
    )

    def __post_init__(self):
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class FederationConfig:
    """Federation configuration for multi-device deployment."""
    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hostname: str = field(default_factory=lambda: os.uname().nodename)
    enable_sync: bool = False
    sync_endpoint: Optional[str] = None


@dataclass
class SAM3DeepStreamConfig:
    """Main configuration container."""
    trt: TRTConfig = field(default_factory=TRTConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    deepstream: DeepStreamConfig = field(default_factory=DeepStreamConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)

    # Path to SAM3 model checkpoint
    sam3_checkpoint: Optional[Path] = None

    # Engine file paths (populated after export)
    encoder_engine: Optional[Path] = None
    decoder_engine: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "SAM3DeepStreamConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override from environment
        if checkpoint := os.getenv("SAM3_CHECKPOINT"):
            config.sam3_checkpoint = Path(checkpoint)

        if engine_dir := os.getenv("SAM3_ENGINE_DIR"):
            engine_path = Path(engine_dir)
            config.encoder_engine = engine_path / "sam3_encoder.engine"
            config.decoder_engine = engine_path / "sam3_decoder.engine"

        if precision := os.getenv("SAM3_PRECISION"):
            config.trt.precision = Precision(precision.lower())

        if keyframe := os.getenv("SAM3_KEYFRAME_INTERVAL"):
            config.inference.keyframe_interval = int(keyframe)

        # Database configuration
        if db_path := os.getenv("SAM3_DB_PATH"):
            config.database.db_path = Path(db_path)

        # API configuration
        if output_dir := os.getenv("SAM3_OUTPUT_DIR"):
            config.api.output_dir = Path(output_dir)
            config.api.output_dir.mkdir(parents=True, exist_ok=True)

        # Federation configuration
        if device_id := os.getenv("SAM3_DEVICE_ID"):
            config.federation.device_id = device_id

        return config


# Global configuration instance
_config: Optional[SAM3DeepStreamConfig] = None


def get_config() -> SAM3DeepStreamConfig:
    """Get or create the global configuration."""
    global _config
    if _config is None:
        _config = SAM3DeepStreamConfig.from_env()
    return _config


def set_config(config: SAM3DeepStreamConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config
