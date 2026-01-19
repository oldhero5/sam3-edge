#!/usr/bin/env python3
"""Test DeepStream pipeline with SAM3 encoder.

This script tests the DeepStream pipeline integration for SAM3 segmentation.
It requires:
- DeepStream SDK installed
- GStreamer with NVIDIA plugins
- TensorRT engine built (sam3_encoder.engine)

Usage:
    python test_deepstream.py --video /path/to/video.mp4
    python test_deepstream.py --camera 0  # Use camera device 0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Check for GStreamer availability
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstBase', '1.0')
    from gi.repository import Gst, GLib, GstBase
    GST_AVAILABLE = True
except (ImportError, ValueError) as e:
    GST_AVAILABLE = False
    logger.warning(f"GStreamer not available: {e}")

# Check for pyds (DeepStream Python bindings)
try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    PYDS_AVAILABLE = False
    logger.warning("pyds (DeepStream Python) not available")


def get_config_paths():
    """Get paths to configuration files."""
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / "config"

    return {
        "nvinfer": config_dir / "sam3_infer_config.txt",
        "tracker": config_dir / "tracker_NvDCF.yml",
    }


def check_engine_exists():
    """Check if TensorRT engine exists."""
    engine_path = Path(__file__).parent.parent / "engines" / "sam3_encoder.engine"
    if not engine_path.exists():
        logger.error(f"Engine not found: {engine_path}")
        logger.error("Run 'python scripts/export_engines.py' first to build the engine")
        return False
    logger.info(f"Found engine: {engine_path}")
    return True


def build_pipeline_string(
    source: str,
    is_camera: bool = False,
    nvinfer_config: Optional[Path] = None,
    tracker_config: Optional[Path] = None,
    output_sink: str = "display",
) -> str:
    """Build GStreamer pipeline string.

    Args:
        source: Video file path or camera device
        is_camera: True if source is a camera
        nvinfer_config: Path to nvinfer config file
        tracker_config: Path to tracker config file
        output_sink: 'display', 'file', or 'fakesink'

    Returns:
        GStreamer pipeline string
    """
    # Source element
    if is_camera:
        src = f"v4l2src device=/dev/video{source} ! video/x-raw,width=1920,height=1080"
    else:
        # Determine decoder based on file extension
        ext = Path(source).suffix.lower()
        if ext in ['.h265', '.hevc']:
            src = f"filesrc location={source} ! h265parse ! nvv4l2decoder"
        elif ext in ['.h264']:
            src = f"filesrc location={source} ! h264parse ! nvv4l2decoder"
        else:
            # Generic demux for containers
            src = f"filesrc location={source} ! qtdemux ! h264parse ! nvv4l2decoder"

    # Streammux
    mux = "nvstreammux batch-size=1 width=1920 height=1080 batched-push-timeout=40000"

    # Inference
    if nvinfer_config and nvinfer_config.exists():
        infer = f"nvinfer config-file-path={nvinfer_config}"
    else:
        logger.warning("nvinfer config not found, skipping inference")
        infer = ""

    # Tracker
    if tracker_config and tracker_config.exists():
        tracker = f"nvtracker ll-config-file={tracker_config}"
    else:
        tracker = ""

    # Output
    if output_sink == "display":
        sink = "nvsegvisual ! nvvideoconvert ! nveglglessink sync=0"
    elif output_sink == "fakesink":
        sink = "fakesink sync=0"
    else:
        sink = f"nvvideoconvert ! x264enc ! mp4mux ! filesink location={output_sink}"

    # Build pipeline
    elements = [src, mux]
    if infer:
        elements.append(infer)
    if tracker:
        elements.append(tracker)
    elements.append(sink)

    return " ! ".join(elements)


def run_simple_test():
    """Run a simple pipeline test without DeepStream plugins."""
    if not GST_AVAILABLE:
        logger.error("GStreamer not available")
        return False

    logger.info("Running simple GStreamer test...")
    Gst.init(None)

    # Simple test pipeline
    pipeline_str = "videotestsrc num-buffers=30 ! videoconvert ! fakesink"

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        pipeline.set_state(Gst.State.PLAYING)

        # Wait for EOS or error
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            5 * Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )

        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error(f"Pipeline error: {err}, {debug}")
                return False
            elif msg.type == Gst.MessageType.EOS:
                logger.info("Simple test completed successfully")
                return True

        pipeline.set_state(Gst.State.NULL)
        return True

    except Exception as e:
        logger.error(f"Simple test failed: {e}")
        return False


def run_deepstream_test(video_path: str):
    """Run DeepStream pipeline test with video file."""
    if not GST_AVAILABLE:
        logger.error("GStreamer not available")
        return False

    if not PYDS_AVAILABLE:
        logger.warning("pyds not available, running without metadata extraction")

    if not check_engine_exists():
        return False

    configs = get_config_paths()

    logger.info(f"Building DeepStream pipeline for: {video_path}")
    Gst.init(None)

    pipeline_str = build_pipeline_string(
        source=video_path,
        is_camera=False,
        nvinfer_config=configs["nvinfer"],
        tracker_config=configs["tracker"],
        output_sink="fakesink",  # Use fakesink for testing
    )

    logger.info(f"Pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)

        # Start pipeline
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to start pipeline")
            return False

        logger.info("Pipeline started, waiting for completion...")

        # Run for a short time or until EOS
        bus = pipeline.get_bus()

        frame_count = 0
        running = True
        while running:
            msg = bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                Gst.MessageType.EOS | Gst.MessageType.ERROR | Gst.MessageType.STATE_CHANGED
            )

            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    logger.error(f"Pipeline error: {err}")
                    logger.debug(f"Debug info: {debug}")
                    running = False
                elif msg.type == Gst.MessageType.EOS:
                    logger.info("End of stream reached")
                    running = False

            frame_count += 1
            if frame_count > 100:  # Limit for testing
                logger.info("Test frame limit reached")
                running = False

        pipeline.set_state(Gst.State.NULL)
        logger.info("DeepStream test completed")
        return True

    except Exception as e:
        logger.error(f"DeepStream test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test DeepStream pipeline with SAM3")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file for testing",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera device number (e.g., 0 for /dev/video0)",
    )
    parser.add_argument(
        "--simple-test",
        action="store_true",
        help="Run simple GStreamer test without DeepStream",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )

    args = parser.parse_args()

    if args.check_deps:
        print("Dependency check:")
        print(f"  GStreamer: {'Available' if GST_AVAILABLE else 'NOT AVAILABLE'}")
        print(f"  PyDS (DeepStream): {'Available' if PYDS_AVAILABLE else 'NOT AVAILABLE'}")
        print(f"  TRT Engine: {'Found' if check_engine_exists() else 'NOT FOUND'}")

        configs = get_config_paths()
        print(f"  nvinfer config: {'Found' if configs['nvinfer'].exists() else 'NOT FOUND'}")
        print(f"  tracker config: {'Found' if configs['tracker'].exists() else 'NOT FOUND'}")
        return

    if args.simple_test:
        success = run_simple_test()
        sys.exit(0 if success else 1)

    if args.video:
        if not Path(args.video).exists():
            logger.error(f"Video file not found: {args.video}")
            sys.exit(1)
        success = run_deepstream_test(args.video)
        sys.exit(0 if success else 1)

    if args.camera is not None:
        logger.info(f"Camera testing not yet implemented")
        sys.exit(1)

    # Default: run simple test
    logger.info("No source specified, running simple GStreamer test")
    success = run_simple_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
