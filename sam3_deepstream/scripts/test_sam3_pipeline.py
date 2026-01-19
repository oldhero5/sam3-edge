#!/usr/bin/env python3
"""
Test DeepStream pipeline with SAM3 encoder and tensor extraction probe.

This script demonstrates:
1. GStreamer pipeline with nvinfer plugin for SAM3 encoder
2. Pad probe to extract raw encoder features
3. Keyframe-based decoder processing
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required dependencies
MISSING_DEPS = []

try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstBase', '1.0')
    from gi.repository import Gst, GLib, GstBase
    GST_AVAILABLE = True
except (ImportError, ValueError) as e:
    GST_AVAILABLE = False
    MISSING_DEPS.append(f"gi/Gst: {e}")

try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError as e:
    PYDS_AVAILABLE = False
    MISSING_DEPS.append(f"pyds: {e}")

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError as e:
    NP_AVAILABLE = False
    MISSING_DEPS.append(f"numpy: {e}")


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    print("\n=== Dependency Check ===\n")

    deps = {
        'GStreamer (gi.repository.Gst)': GST_AVAILABLE,
        'DeepStream Python bindings (pyds)': PYDS_AVAILABLE,
        'NumPy': NP_AVAILABLE,
    }

    all_ok = True
    for name, available in deps.items():
        status = "✓ OK" if available else "✗ MISSING"
        print(f"  {name}: {status}")
        if not available:
            all_ok = False

    if MISSING_DEPS:
        print("\nMissing dependency details:")
        for dep in MISSING_DEPS:
            print(f"  - {dep}")

    # Check for nvinfer plugin
    if GST_AVAILABLE:
        Gst.init(None)
        registry = Gst.Registry.get()
        nvinfer = registry.find_plugin("nvinfer")
        nvvideo = registry.find_plugin("nvvideo4linux2")

        print(f"\n  nvinfer plugin: {'✓ OK' if nvinfer else '✗ MISSING'}")
        print(f"  nvvideo4linux2 plugin: {'✓ OK' if nvvideo else '✗ MISSING'}")

        if not nvinfer:
            print("\n  Note: nvinfer requires DeepStream SDK installation")
            all_ok = False

    # Check for engine file
    engine_path = Path(__file__).parent.parent / "engines" / "sam3_encoder.engine"
    print(f"\n  SAM3 encoder engine: {'✓ EXISTS' if engine_path.exists() else '✗ NOT FOUND'}")
    if not engine_path.exists():
        print(f"    Expected at: {engine_path}")

    print()
    return all_ok


def run_simple_gstreamer_test() -> bool:
    """Run a simple GStreamer test without DeepStream."""
    if not GST_AVAILABLE:
        logger.error("GStreamer not available")
        return False

    logger.info("Running simple GStreamer test...")

    # Simple test pipeline
    pipeline_str = "videotestsrc num-buffers=10 ! videoconvert ! fakesink"

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        pipeline.set_state(Gst.State.PLAYING)

        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            5 * Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )

        pipeline.set_state(Gst.State.NULL)

        if msg and msg.type == Gst.MessageType.EOS:
            logger.info("Simple GStreamer test PASSED")
            return True
        elif msg and msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error(f"GStreamer error: {err.message}")
            return False
        else:
            logger.warning("GStreamer test timed out")
            return False

    except Exception as e:
        logger.error(f"GStreamer test failed: {e}")
        return False


class SAM3PipelineProbe:
    """
    Probe handler for extracting tensor output from nvinfer.

    Attaches to the src pad of nvinfer to access:
    - NvDsInferTensorMeta with raw encoder features
    - Frame metadata for keyframe decisions
    """

    def __init__(
        self,
        keyframe_interval: int = 5,
        decoder_engine_path: Optional[str] = None,
    ):
        self.keyframe_interval = keyframe_interval
        self.frame_count = 0
        self.keyframe_count = 0
        self.total_inference_time = 0.0

        # Load decoder if available (optional)
        self.decoder = None
        if decoder_engine_path and Path(decoder_engine_path).exists():
            try:
                from sam3_deepstream.inference.trt_runtime import TRTInferenceEngine
                self.decoder = TRTInferenceEngine(decoder_engine_path)
                logger.info(f"Decoder loaded: {decoder_engine_path}")
            except Exception as e:
                logger.warning(f"Could not load decoder: {e}")

    def __call__(self, pad, info):
        """Pad probe callback for tensor extraction."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_num = frame_meta.frame_num
            is_keyframe = (frame_num % self.keyframe_interval == 0)

            self.frame_count += 1

            # Extract tensor metadata
            l_user = frame_meta.frame_user_meta_list
            while l_user:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)

                    # Check for tensor output metadata
                    if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                        if is_keyframe:
                            self._process_keyframe(tensor_meta, frame_num)

                except StopIteration:
                    break

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _process_keyframe(self, tensor_meta, frame_num: int) -> None:
        """Process keyframe with decoder."""
        self.keyframe_count += 1

        # Extract tensor info
        num_layers = tensor_meta.num_output_layers
        logger.debug(f"Keyframe {frame_num}: {num_layers} output layers")

        # Get output tensor
        for i in range(num_layers):
            layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
            if layer:
                logger.debug(f"  Layer {i}: {layer.layerName}, dims={layer.dims}")

                # If we have a decoder, run mask generation
                if self.decoder and layer.layerName == "features":
                    # Note: Actual tensor data extraction requires additional
                    # memory mapping from layer.buffer
                    pass

    @property
    def stats(self) -> dict:
        return {
            'frames_processed': self.frame_count,
            'keyframes': self.keyframe_count,
            'avg_inference_time_ms': self.total_inference_time / max(1, self.keyframe_count),
        }


def create_deepstream_pipeline(
    video_path: str,
    config_path: str,
    output_path: Optional[str] = None,
) -> Optional[Gst.Pipeline]:
    """
    Create DeepStream pipeline for SAM3 inference.

    Pipeline:
    filesrc -> demux -> decode -> nvstreammux -> nvinfer -> nvvideoconvert -> output
    """
    if not GST_AVAILABLE:
        logger.error("GStreamer not available")
        return None

    Gst.init(None)

    # Build pipeline string
    pipeline_str = f"""
        filesrc location="{video_path}" !
        qtdemux ! h264parse !
        nvv4l2decoder !
        m.sink_0
        nvstreammux name=m batch-size=1 width=1920 height=1080 !
        nvinfer config-file-path="{config_path}" !
        nvvideoconvert !
    """

    if output_path:
        pipeline_str += f"""
            nvv4l2h264enc !
            h264parse !
            mp4mux !
            filesink location="{output_path}"
        """
    else:
        pipeline_str += "fakesink"

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return None


def run_deepstream_test(
    video_path: str,
    encoder_engine: str,
    decoder_engine: Optional[str] = None,
    keyframe_interval: int = 5,
    max_frames: int = 100,
) -> bool:
    """Run DeepStream pipeline test with tensor probe."""
    if not GST_AVAILABLE or not PYDS_AVAILABLE:
        logger.error("Missing required dependencies for DeepStream test")
        return False

    config_path = Path(__file__).parent.parent / "config" / "sam3_infer_config.txt"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    if not Path(encoder_engine).exists():
        logger.error(f"Encoder engine not found: {encoder_engine}")
        return False

    logger.info(f"Creating DeepStream pipeline...")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Keyframe interval: {keyframe_interval}")

    pipeline = create_deepstream_pipeline(video_path, str(config_path))
    if not pipeline:
        return False

    # Create probe
    probe = SAM3PipelineProbe(
        keyframe_interval=keyframe_interval,
        decoder_engine_path=decoder_engine,
    )

    # Attach probe to nvinfer src pad
    nvinfer = pipeline.get_by_name("nvinfer0") or pipeline.get_by_name("nvinfer")
    if nvinfer:
        src_pad = nvinfer.get_static_pad("src")
        if src_pad:
            src_pad.add_probe(Gst.PadProbeType.BUFFER, probe)
            logger.info("Probe attached to nvinfer src pad")
        else:
            logger.warning("Could not get nvinfer src pad")
    else:
        logger.warning("Could not find nvinfer element")

    # Run pipeline
    pipeline.set_state(Gst.State.PLAYING)

    bus = pipeline.get_bus()
    start_time = time.time()

    try:
        while True:
            msg = bus.timed_pop(100 * Gst.MSECOND)

            if msg:
                if msg.type == Gst.MessageType.EOS:
                    logger.info("End of stream")
                    break
                elif msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    logger.error(f"Pipeline error: {err.message}")
                    break

            # Check frame limit
            if probe.frame_count >= max_frames:
                logger.info(f"Reached max frames ({max_frames})")
                break

            # Timeout after 60 seconds
            if time.time() - start_time > 60:
                logger.warning("Test timed out")
                break

    finally:
        pipeline.set_state(Gst.State.NULL)

    # Print stats
    stats = probe.stats
    elapsed = time.time() - start_time
    fps = stats['frames_processed'] / max(0.001, elapsed)

    print("\n=== Pipeline Test Results ===")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Keyframes: {stats['keyframes']}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {fps:.1f} FPS")

    return stats['frames_processed'] > 0


def main():
    parser = argparse.ArgumentParser(description='Test SAM3 DeepStream Pipeline')
    parser.add_argument('--check-deps', action='store_true',
                        help='Check dependencies and exit')
    parser.add_argument('--simple-test', action='store_true',
                        help='Run simple GStreamer test')
    parser.add_argument('--video', type=str,
                        help='Input video file path')
    parser.add_argument('--encoder-engine', type=str,
                        help='Path to SAM3 encoder TRT engine')
    parser.add_argument('--decoder-engine', type=str,
                        help='Path to SAM3 decoder TRT engine (optional)')
    parser.add_argument('--keyframe-interval', type=int, default=5,
                        help='Run decoder every N frames')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Maximum frames to process')

    args = parser.parse_args()

    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)

    if args.simple_test:
        success = run_simple_gstreamer_test()
        sys.exit(0 if success else 1)

    # Full pipeline test
    if not args.video:
        # Try to find test video
        test_videos = [
            "/mnt/repos/sam3_test/test_video/test.mp4",
            "/workspace/test.mp4",
            "test.mp4",
        ]
        for v in test_videos:
            if Path(v).exists():
                args.video = v
                break

        if not args.video:
            parser.error("--video is required (no test video found)")

    if not args.encoder_engine:
        default_engine = Path(__file__).parent.parent / "engines" / "sam3_encoder.engine"
        if default_engine.exists():
            args.encoder_engine = str(default_engine)
        else:
            parser.error("--encoder-engine is required")

    success = run_deepstream_test(
        video_path=args.video,
        encoder_engine=args.encoder_engine,
        decoder_engine=args.decoder_engine,
        keyframe_interval=args.keyframe_interval,
        max_frames=args.max_frames,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
