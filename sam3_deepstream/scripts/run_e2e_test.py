#!/usr/bin/env python3
"""End-to-end test of SAM3 edge inference system.

This script tests the complete SAM3 inference pipeline:
1. TensorRT encoder inference
2. Keyframe-based processing
3. Mask propagation between keyframes
4. Performance measurement

Usage:
    python run_e2e_test.py --video /path/to/video.mp4
    python run_e2e_test.py --synthetic  # Use synthetic test frames
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def get_engine_paths() -> Tuple[Path, Path]:
    """Get paths to TensorRT engines."""
    engine_dir = Path(__file__).parent.parent / "engines"
    return (
        engine_dir / "sam3_encoder.engine",
        engine_dir / "sam3_decoder.engine",
    )


def get_checkpoint_path() -> Path:
    """Get path to SAM3 checkpoint."""
    return Path(__file__).parent.parent.parent / "sam3" / "checkpoints" / "sam3.pt"


def run_synthetic_test(
    num_frames: int = 100,
    frame_size: Tuple[int, int] = (1008, 1008),
    keyframe_interval: int = 5,
    use_vpi: bool = False,
) -> dict:
    """Run synthetic test with generated frames.

    Args:
        num_frames: Number of frames to process
        frame_size: Frame dimensions (H, W)
        keyframe_interval: Keyframe interval
        use_vpi: Whether to use VPI acceleration

    Returns:
        Dictionary with test results
    """
    from sam3_deepstream.inference.keyframe_processor import KeyframeProcessor
    from sam3_deepstream.inference.mask_propagation import MaskPropagator

    encoder_path, decoder_path = get_engine_paths()

    # Check if engines exist
    if not encoder_path.exists():
        logger.warning(f"Encoder engine not found: {encoder_path}")
        logger.info("Running test with dummy encoder (no actual inference)")
        encoder_fn = lambda x: torch.zeros(1, 256, 64, 64).cuda() if CUDA_AVAILABLE else None
    else:
        from sam3_deepstream.inference.trt_runtime import SAM3TRTRuntime
        runtime = SAM3TRTRuntime(encoder_engine_path=encoder_path)
        encoder_fn = runtime.encode_image

    # Initialize processor
    processor = KeyframeProcessor(
        encoder_fn=encoder_fn,
        decoder_fn=None,  # Not using decoder for this test
        keyframe_interval=keyframe_interval,
    )

    # Set up propagation
    propagator = MaskPropagator(use_vpi=use_vpi)
    processor.set_propagator(propagator)

    # Add a dummy tracked object
    initial_mask = np.zeros(frame_size, dtype=np.uint8)
    initial_mask[frame_size[0]//4:frame_size[0]*3//4, frame_size[1]//4:frame_size[1]*3//4] = 1
    processor.add_object(initial_mask, score=0.95)

    # Process frames
    frame_times: List[float] = []
    keyframe_times: List[float] = []
    propagation_times: List[float] = []

    logger.info(f"Processing {num_frames} synthetic frames...")

    for i in range(num_frames):
        # Generate synthetic frame with some motion
        frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)

        start = time.perf_counter()
        result = processor.process_frame(frame)
        elapsed = time.perf_counter() - start

        frame_times.append(elapsed)
        if result.is_keyframe:
            keyframe_times.append(elapsed)
        else:
            propagation_times.append(elapsed)

        if (i + 1) % 20 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-20:])
            logger.info(f"Frame {i+1}/{num_frames}: {avg_fps:.1f} FPS, keyframe={result.is_keyframe}")

    # Calculate statistics
    results = {
        "num_frames": num_frames,
        "frame_size": frame_size,
        "keyframe_interval": keyframe_interval,
        "use_vpi": use_vpi,
        "total_time_ms": sum(frame_times) * 1000,
        "avg_frame_time_ms": np.mean(frame_times) * 1000,
        "avg_fps": 1.0 / np.mean(frame_times),
        "avg_keyframe_time_ms": np.mean(keyframe_times) * 1000 if keyframe_times else 0,
        "avg_propagation_time_ms": np.mean(propagation_times) * 1000 if propagation_times else 0,
        "num_keyframes": len(keyframe_times),
        "num_propagated": len(propagation_times),
    }

    return results


def run_video_test(
    video_path: str,
    max_frames: Optional[int] = None,
    keyframe_interval: int = 5,
    use_vpi: bool = False,
) -> dict:
    """Run test with real video file.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process (None for all)
        keyframe_interval: Keyframe interval
        use_vpi: Whether to use VPI acceleration

    Returns:
        Dictionary with test results
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV not available")
        return {}

    from sam3_deepstream.inference.keyframe_processor import KeyframeProcessor
    from sam3_deepstream.inference.mask_propagation import MaskPropagator

    encoder_path, decoder_path = get_engine_paths()

    # Check if engines exist
    if not encoder_path.exists():
        logger.error(f"Encoder engine not found: {encoder_path}")
        logger.error("Run export_engines.py first to build the engine")
        return {}

    from sam3_deepstream.inference.trt_runtime import SAM3TRTRuntime
    runtime = SAM3TRTRuntime(encoder_engine_path=encoder_path)

    # Initialize processor
    processor = KeyframeProcessor(
        encoder_fn=runtime.encode_image,
        decoder_fn=None,
        keyframe_interval=keyframe_interval,
    )

    propagator = MaskPropagator(use_vpi=use_vpi)
    processor.set_propagator(propagator)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_width}x{video_height} @ {video_fps:.1f} FPS, {total_frames} frames")

    if max_frames:
        total_frames = min(total_frames, max_frames)
        logger.info(f"Processing first {total_frames} frames")

    # Process frames
    frame_times: List[float] = []
    keyframe_times: List[float] = []
    propagation_times: List[float] = []
    frame_count = 0

    logger.info("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_count >= max_frames:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        result = processor.process_frame(frame_rgb)
        elapsed = time.perf_counter() - start

        frame_times.append(elapsed)
        if result.is_keyframe:
            keyframe_times.append(elapsed)
        else:
            propagation_times.append(elapsed)

        frame_count += 1

        if frame_count % 30 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-30:])
            logger.info(f"Frame {frame_count}: {avg_fps:.1f} FPS")

    cap.release()

    # Calculate statistics
    results = {
        "video_path": video_path,
        "video_size": f"{video_width}x{video_height}",
        "video_fps": video_fps,
        "num_frames": frame_count,
        "keyframe_interval": keyframe_interval,
        "use_vpi": use_vpi,
        "total_time_ms": sum(frame_times) * 1000,
        "avg_frame_time_ms": np.mean(frame_times) * 1000,
        "avg_fps": 1.0 / np.mean(frame_times),
        "avg_keyframe_time_ms": np.mean(keyframe_times) * 1000 if keyframe_times else 0,
        "avg_propagation_time_ms": np.mean(propagation_times) * 1000 if propagation_times else 0,
        "num_keyframes": len(keyframe_times),
        "num_propagated": len(propagation_times),
        "realtime_factor": (1.0 / np.mean(frame_times)) / video_fps if video_fps > 0 else 0,
    }

    return results


def print_results(results: dict):
    """Print test results in formatted table."""
    print("\n" + "=" * 60)
    print("SAM3 Edge Inference Test Results")
    print("=" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.2f}")
        else:
            print(f"  {key:30s}: {value}")

    print("=" * 60)

    # Performance assessment
    avg_fps = results.get("avg_fps", 0)
    if avg_fps >= 30:
        print("Performance: EXCELLENT (>=30 FPS)")
    elif avg_fps >= 15:
        print("Performance: GOOD (15-30 FPS)")
    elif avg_fps >= 8:
        print("Performance: ACCEPTABLE (8-15 FPS)")
    else:
        print("Performance: NEEDS OPTIMIZATION (<8 FPS)")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test of SAM3 edge inference system"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file for testing",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic test with generated frames",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=5,
        help="Keyframe interval (default: 5)",
    )
    parser.add_argument(
        "--use-vpi",
        action="store_true",
        help="Use VPI acceleration for optical flow",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        default=[1008, 1008],
        help="Frame size for synthetic test (default: 1008 1008)",
    )

    args = parser.parse_args()

    # Check dependencies
    print("Checking dependencies...")
    print(f"  PyTorch: {'Available' if TORCH_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  CUDA: {'Available' if CUDA_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  OpenCV: {'Available' if CV2_AVAILABLE else 'NOT AVAILABLE'}")

    encoder_path, decoder_path = get_engine_paths()
    print(f"  Encoder engine: {'Found' if encoder_path.exists() else 'NOT FOUND'}")
    print(f"  Decoder engine: {'Found' if decoder_path.exists() else 'NOT FOUND'}")

    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required")
        sys.exit(1)

    # Run test
    if args.video:
        if not Path(args.video).exists():
            logger.error(f"Video file not found: {args.video}")
            sys.exit(1)

        results = run_video_test(
            video_path=args.video,
            max_frames=args.num_frames if args.num_frames != 100 else None,
            keyframe_interval=args.keyframe_interval,
            use_vpi=args.use_vpi,
        )
    else:
        # Default to synthetic test
        results = run_synthetic_test(
            num_frames=args.num_frames,
            frame_size=tuple(args.frame_size),
            keyframe_interval=args.keyframe_interval,
            use_vpi=args.use_vpi,
        )

    if results:
        print_results(results)
    else:
        logger.error("Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
