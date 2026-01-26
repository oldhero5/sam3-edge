#!/usr/bin/env python3
"""Export SAM3 models to TensorRT engines."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the package directory to path when running script directly
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from sam3_deepstream.config import get_config, Precision
from sam3_deepstream.export.encoder_export import export_encoder_to_tensorrt
from sam3_deepstream.export.decoder_export import export_decoder_to_tensorrt
from sam3_deepstream.export.pe_encoder_export import (
    export_pe_encoder_to_onnx,
    build_pe_encoder_engine,
    export_pe_model_engines,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_export_summary(results: dict) -> None:
    """Print a summary of export results."""
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    # Model loading
    status = "OK" if results.get("model_loaded") else "FAILED"
    print(f"  Model Loading:      [{status}]")

    # PE Encoder
    if results.get("pe_encoder_onnx"):
        print(f"  PE Encoder ONNX:    [OK] {results['pe_encoder_onnx']}")
    elif results.get("pe_encoder_onnx") is False:
        print("  PE Encoder ONNX:    [FAILED]")

    if results.get("pe_encoder_engine"):
        print(f"  PE Encoder Engine:  [OK] {results['pe_encoder_engine']}")
    elif results.get("pe_encoder_engine") is False:
        print("  PE Encoder Engine:  [FAILED]")

    # Standard Encoder
    if results.get("encoder_onnx"):
        print(f"  Encoder ONNX:       [OK] {results['encoder_onnx']}")
    elif results.get("encoder_onnx") is False:
        print("  Encoder ONNX:       [FAILED]")

    if results.get("encoder_engine"):
        print(f"  Encoder Engine:     [OK] {results['encoder_engine']}")
    elif results.get("encoder_engine") is False:
        print("  Encoder Engine:     [FAILED]")

    # Decoder
    if results.get("decoder_onnx"):
        print(f"  Decoder ONNX:       [OK] {results['decoder_onnx']}")
    elif results.get("decoder_onnx") is False:
        print("  Decoder ONNX:       [FAILED]")

    if results.get("decoder_engine"):
        print(f"  Decoder Engine:     [OK] {results['decoder_engine']}")
    elif results.get("decoder_engine") is False:
        print("  Decoder Engine:     [FAILED]")

    # Errors
    if results.get("errors"):
        print("\nERRORS:")
        for error in results["errors"]:
            print(f"  - {error}")

    print("=" * 60 + "\n")


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
    parser.add_argument(
        "--use-pe",
        action="store_true",
        help="Use Perception Encoder (PE) backbone instead of standard ViT",
    )
    parser.add_argument(
        "--pe-only",
        action="store_true",
        help="Only export PE encoder (implies --use-pe)",
    )
    parser.add_argument(
        "--alignment-tuning",
        action="store_true",
        default=True,
        help="Enable PE alignment tuning (intermediate layer fusion)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on any error (default: continue and report)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing ONNX files, don't export",
    )

    args = parser.parse_args()

    # Handle PE-only flag
    if args.pe_only:
        args.use_pe = True

    # Set precision
    precision = Precision(args.precision)

    # Track export results for summary
    export_results = {
        "model_loaded": False,
        "pe_encoder_onnx": None,
        "pe_encoder_engine": None,
        "encoder_onnx": None,
        "encoder_engine": None,
        "decoder_onnx": None,
        "decoder_engine": None,
        "errors": [],
    }

    # Validate checkpoint exists
    if not args.checkpoint.exists():
        error_msg = f"Checkpoint not found: {args.checkpoint}"
        logger.error(error_msg)
        export_results["errors"].append(error_msg)
        if args.strict:
            logger.error("Aborting due to --strict flag")
            sys.exit(1)
        print_export_summary(export_results)
        return

    # Load SAM3 model
    logger.info(f"Loading SAM3 checkpoint: {args.checkpoint}")
    model = None

    try:
        import torch
        import sys as _sys

        # Add SAM3 to path
        sam3_path = Path(__file__).parent.parent.parent / "sam3"
        _sys.path.insert(0, str(sam3_path))

        from sam3.model_builder import build_sam3_image_model, build_sam3_pe_model

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("CUDA not available - loading on CPU")
            logger.warning("TensorRT engine building requires CUDA")
            if args.strict:
                raise RuntimeError("CUDA required for TensorRT export but not available")

        # Load model - PE or standard
        if args.use_pe:
            logger.info("Loading SAM3 with PE backbone...")
            model = build_sam3_pe_model(
                device=device,
                eval_mode=True,
                checkpoint_path=str(args.checkpoint),
                load_from_HF=False,
                use_alignment_tuning=args.alignment_tuning,
            )
            logger.info(f"PE model loaded successfully on {device}")
        else:
            # Load model directly with checkpoint path (skip HuggingFace download)
            model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                checkpoint_path=str(args.checkpoint),
                load_from_HF=False,
            )
            logger.info(f"Model loaded successfully on {device}")

        export_results["model_loaded"] = True

    except ImportError as e:
        error_msg = f"Missing dependency for model loading: {e}"
        logger.error(error_msg)
        export_results["errors"].append(error_msg)
        if args.strict:
            logger.error("Aborting due to --strict flag")
            sys.exit(1)
    except FileNotFoundError as e:
        error_msg = f"Checkpoint file not found: {e}"
        logger.error(error_msg)
        export_results["errors"].append(error_msg)
        if args.strict:
            logger.error("Aborting due to --strict flag")
            sys.exit(1)
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        export_results["errors"].append(error_msg)
        if args.strict:
            logger.error("Aborting due to --strict flag")
            sys.exit(1)

    # Check if model loaded successfully
    if model is None:
        logger.error("=" * 60)
        logger.error("MODEL LOADING FAILED - Cannot proceed with export")
        logger.error("=" * 60)
        logger.error("Possible causes:")
        logger.error("  1. Checkpoint file is corrupted or incompatible")
        logger.error("  2. Missing SAM3 or PyTorch dependencies")
        logger.error("  3. Insufficient GPU memory")
        logger.error("  4. Wrong model type (standard vs PE backbone)")
        logger.error("")
        logger.error("Try running with --strict to see full error traceback")
        print_export_summary(export_results)
        sys.exit(1)

    output_dir = args.output_dir or Path.home() / ".cache" / "sam3_deepstream" / "engines"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export PE encoder if using PE backbone
    if args.use_pe and model is not None:
        logger.info("Exporting PE encoder to TensorRT...")
        try:
            # Export to ONNX first
            pe_onnx_path = output_dir / "pe_encoder.onnx"
            export_pe_encoder_to_onnx(
                model=model,
                output_path=str(pe_onnx_path),
                input_size=1008,
                export_intermediates=args.alignment_tuning,
            )
            logger.info(f"PE encoder ONNX exported: {pe_onnx_path}")
            export_results["pe_encoder_onnx"] = str(pe_onnx_path)

            # Validate ONNX
            if not validate_onnx(str(pe_onnx_path)):
                logger.warning("PE encoder ONNX validation failed")
                if args.strict:
                    raise ValueError("ONNX validation failed")

            # Build TensorRT engine
            pe_engine_path = output_dir / f"pe_encoder_{args.precision}.engine"
            build_pe_encoder_engine(
                onnx_path=str(pe_onnx_path),
                engine_path=str(pe_engine_path),
                precision=args.precision,
                workspace_mb=int(args.workspace_gb * 1024),
                dla_core=args.dla_core,
            )
            logger.info(f"PE encoder TRT engine exported: {pe_engine_path}")
            export_results["pe_encoder_engine"] = str(pe_engine_path)

        except Exception as e:
            error_msg = f"PE encoder export failed: {e}"
            logger.error(error_msg)
            export_results["errors"].append(error_msg)
            export_results["pe_encoder_onnx"] = export_results.get("pe_encoder_onnx") or False
            export_results["pe_encoder_engine"] = False
            import traceback
            traceback.print_exc()
            if args.strict:
                print_export_summary(export_results)
                sys.exit(1)

        # If PE-only mode, skip standard exports
        if args.pe_only:
            logger.info("PE-only export complete!")
            print_export_summary(export_results)
            return

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
                export_results["encoder_engine"] = str(encoder_path)

                # Validate ONNX if file exists
                onnx_file = find_onnx_file(output_dir, "encoder") or find_onnx_file(output_dir, "vit")
                if onnx_file:
                    export_results["encoder_onnx"] = str(onnx_file)
                    if not validate_onnx(str(onnx_file)):
                        logger.warning("Encoder ONNX validation failed - TRT engine may not work correctly")
                        if args.strict:
                            raise ValueError("Encoder ONNX validation failed")
            except Exception as e:
                error_msg = f"Encoder export failed: {e}"
                logger.error(error_msg)
                export_results["errors"].append(error_msg)
                export_results["encoder_engine"] = False
                import traceback
                traceback.print_exc()
                if args.strict:
                    print_export_summary(export_results)
                    sys.exit(1)
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
                export_results["decoder_engine"] = str(decoder_path)

                # Validate ONNX if file exists
                onnx_file = find_onnx_file(output_dir, "decoder")
                if onnx_file:
                    export_results["decoder_onnx"] = str(onnx_file)
                    if not validate_onnx(str(onnx_file)):
                        logger.warning("Decoder ONNX validation failed - TRT engine may not work correctly")
                        if args.strict:
                            raise ValueError("Decoder ONNX validation failed")
            except Exception as e:
                error_msg = f"Decoder export failed: {e}"
                logger.error(error_msg)
                export_results["errors"].append(error_msg)
                export_results["decoder_engine"] = False
                import traceback
                traceback.print_exc()
                if args.strict:
                    print_export_summary(export_results)
                    sys.exit(1)
        else:
            logger.warning("Skipping decoder export (no model loaded)")

    # Print final summary
    print_export_summary(export_results)

    # Determine exit code
    has_errors = bool(export_results["errors"])
    if has_errors:
        logger.warning("Export completed with errors")
        if args.strict:
            sys.exit(1)
    else:
        logger.info("Export complete!")


if __name__ == "__main__":
    main()
