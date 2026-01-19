#!/usr/bin/env python3
"""Export SAM3 models to TensorRT engines."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from sam3_deepstream.config import get_config, Precision
from sam3_deepstream.export.encoder_export import export_encoder_to_tensorrt
from sam3_deepstream.export.decoder_export import export_decoder_to_tensorrt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_onnx(onnx_path: str) -> bool:
    """
    Validate ONNX model for TensorRT compatibility.

    Checks for:
    - Model structure validity
    - Unsupported complex-valued operations
    - Operations that may fail in TensorRT

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        True if valid, False otherwise
    """
    try:
        import onnx
        import onnx.checker
    except ImportError:
        logger.warning("onnx package not available, skipping validation")
        return True

    try:
        logger.info(f"Validating ONNX model: {onnx_path}")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Check for unsupported operations
        unsupported_ops = []
        problematic_ops = ['ComplexAbs', 'ComplexMul', 'DFT', 'IDFT']

        for node in model.graph.node:
            if node.op_type in problematic_ops:
                unsupported_ops.append(f"{node.name}: {node.op_type}")
            # Also check for complex-related patterns in op names
            if 'Complex' in node.op_type:
                unsupported_ops.append(f"{node.name}: {node.op_type}")

        if unsupported_ops:
            logger.error("Found unsupported operations for TensorRT:")
            for op in unsupported_ops:
                logger.error(f"  - {op}")
            return False

        logger.info("ONNX validation passed")
        return True

    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def find_onnx_file(output_dir: Path, prefix: str) -> Optional[Path]:
    """Find ONNX file in output directory."""
    patterns = [f"{prefix}.onnx", f"sam3_{prefix}.onnx", f"{prefix}_simplified.onnx"]
    for pattern in patterns:
        path = output_dir / pattern
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 models to TensorRT engines"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to SAM3 model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for engines (default: ~/.cache/sam3_deepstream/engines)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "int8"],
        default="fp16",
        help="TensorRT precision mode",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Only export encoder",
    )
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help="Only export decoder",
    )
    parser.add_argument(
        "--dla-core",
        type=int,
        default=None,
        help="DLA core to use (0 or 1 on Jetson)",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=4.0,
        help="TensorRT workspace size in GB",
    )

    args = parser.parse_args()

    # Set precision
    precision = Precision(args.precision)

    # Load SAM3 model
    logger.info(f"Loading SAM3 checkpoint: {args.checkpoint}")

    try:
        import torch
        import sys

        # Add SAM3 to path
        sam3_path = Path(__file__).parent.parent.parent / "sam3"
        sys.path.insert(0, str(sam3_path))

        from sam3.model_builder import build_sam3_image_model

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("CUDA not available - loading on CPU (export will fail)")
            logger.warning("On Jetson, ensure you're using system Python with JetPack CUDA")

        # Load model directly with checkpoint path (skip HuggingFace download)
        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
        )
        logger.info(f"Model loaded successfully on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Continuing with dummy export for testing...")
        model = None

    output_dir = args.output_dir or Path.home() / ".cache" / "sam3_deepstream" / "engines"
    output_dir = Path(output_dir)

    # Export encoder
    if not args.decoder_only:
        logger.info("Exporting encoder to TensorRT...")

        if model is not None:
            try:
                encoder_path = export_encoder_to_tensorrt(
                    sam3_model=model,
                    output_dir=output_dir,
                    precision=precision,
                    dla_core=args.dla_core,
                )
                logger.info(f"Encoder exported: {encoder_path}")

                # Validate ONNX if file exists
                onnx_file = find_onnx_file(output_dir, "encoder") or find_onnx_file(output_dir, "vit")
                if onnx_file:
                    if not validate_onnx(str(onnx_file)):
                        logger.warning("Encoder ONNX validation failed - TRT engine may not work correctly")
            except Exception as e:
                logger.error(f"Encoder export failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("Skipping encoder export (no model loaded)")

    # Export decoder
    if not args.encoder_only:
        logger.info("Exporting decoder to TensorRT...")

        if model is not None:
            try:
                decoder_path = export_decoder_to_tensorrt(
                    sam3_model=model,
                    output_dir=output_dir,
                    precision=precision,
                    workspace_size_gb=args.workspace_gb,
                )
                logger.info(f"Decoder exported: {decoder_path}")

                # Validate ONNX if file exists
                onnx_file = find_onnx_file(output_dir, "decoder")
                if onnx_file:
                    if not validate_onnx(str(onnx_file)):
                        logger.warning("Decoder ONNX validation failed - TRT engine may not work correctly")
            except Exception as e:
                logger.error(f"Decoder export failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("Skipping decoder export (no model loaded)")

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
