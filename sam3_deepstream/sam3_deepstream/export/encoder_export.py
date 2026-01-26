"""Encoder TensorRT export - wraps existing SAM3 ViT export utilities."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

# Import from sam3.model.trt_export (the actual module path)
from sam3.model.trt_export import (
    TRTViTWrapper,
    export_vit_to_onnx,
    build_trt_engine,
    convert_sam3_vit_to_trt,
)

from ..config import get_config, Precision

logger = logging.getLogger(__name__)


def export_encoder_to_tensorrt(
    sam3_model: nn.Module,
    output_dir: Optional[Union[str, Path]] = None,
    precision: Precision = Precision.FP16,
    dla_core: Optional[int] = None,
) -> Path:
    """
    Export SAM3 encoder (ViT backbone) to TensorRT.

    This wraps the existing SAM3 export utilities and integrates
    with sam3_deepstream configuration.

    Args:
        sam3_model: SAM3 model (Sam3Image or Sam3VideoInference)
        output_dir: Directory for output files. If None, uses config cache dir.
        precision: TensorRT precision mode
        dla_core: DLA core to use (0/1 on Jetson, None for GPU)

    Returns:
        Path to the TensorRT engine file
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.trt.cache_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use existing SAM3 export utility
    fp16 = precision in (Precision.FP16, Precision.INT8)

    engine_path = convert_sam3_vit_to_trt(
        sam3_model=sam3_model,
        output_dir=str(output_dir),
        fp16=fp16,
        dla_core=dla_core,
    )

    # Rename to standard name
    standard_name = output_dir / "sam3_encoder.engine"
    if Path(engine_path) != standard_name:
        os.rename(engine_path, standard_name)

    logger.info(f"Encoder engine saved to: {standard_name}")
    return standard_name


def get_encoder_engine_path() -> Optional[Path]:
    """Get path to existing encoder engine if available."""
    config = get_config()

    if config.encoder_engine and config.encoder_engine.exists():
        return config.encoder_engine

    # Check default cache location
    default_path = config.trt.cache_dir / "sam3_encoder.engine"
    if default_path.exists():
        return default_path

    return None
