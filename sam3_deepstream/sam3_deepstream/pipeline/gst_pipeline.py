"""GStreamer pipeline builder for SAM3 DeepStream."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

from ..config import get_config, SAM3DeepStreamConfig

logger = logging.getLogger(__name__)

# Check for GStreamer availability
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp, GLib
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False
    logger.warning("GStreamer not available. Install with: apt install python3-gi gstreamer1.0-*")


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    video_path: str
    nvinfer_config: str
    tracker_config: str
    output_width: int = 1920
    output_height: int = 1080
    sync_output: bool = False


class SAM3Pipeline:
    """
    GStreamer pipeline for SAM3 video segmentation.

    Uses DeepStream elements for hardware-accelerated video processing:
    - nvv4l2decoder: Hardware video decode (NVDEC)
    - nvstreammux: Batches frames for inference
    - nvinfer: TensorRT inference (SAM3 encoder)
    - nvtracker: Object tracking (NvDCF)
    - Custom probe: SAM3 decoder + mask generation
    """

    def __init__(
        self,
        config: Optional[SAM3DeepStreamConfig] = None,
    ):
        if not GST_AVAILABLE:
            raise RuntimeError("GStreamer is not available")

        self.config = config or get_config()
        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop: Optional[GLib.MainLoop] = None
        self._frame_callback: Optional[Callable] = None
        self._is_running = False

        # Initialize GStreamer
        Gst.init(None)

    def build_pipeline(
        self,
        video_path: Union[str, Path],
        nvinfer_config_path: Union[str, Path],
        tracker_config_path: Union[str, Path],
        output_sink: str = "appsink",
    ) -> Gst.Pipeline:
        """
        Build the GStreamer pipeline.

        Args:
            video_path: Input video file path
            nvinfer_config_path: Path to nvinfer config file
            tracker_config_path: Path to tracker config file
            output_sink: Output sink type ("appsink", "filesink", "autovideosink")

        Returns:
            Configured GStreamer pipeline
        """
        video_path = str(video_path)
        nvinfer_config_path = str(nvinfer_config_path)
        tracker_config_path = str(tracker_config_path)

        # Determine file type for demuxer selection
        ext = Path(video_path).suffix.lower()
        if ext in (".mp4", ".mov"):
            demux_str = "qtdemux ! h264parse"
        elif ext in (".mkv", ".webm"):
            demux_str = "matroskademux ! h264parse"
        elif ext == ".avi":
            demux_str = "avidemux ! h264parse"
        else:
            demux_str = "decodebin"

        # Build pipeline string
        pipeline_str = f"""
            filesrc location="{video_path}" !
            {demux_str} !
            nvv4l2decoder enable-max-performance=1 !
            nvstreammux name=mux
                batch-size=1
                width={self.config.deepstream.streammux_width}
                height={self.config.deepstream.streammux_height}
                batched-push-timeout={self.config.deepstream.batched_push_timeout} !
            nvinfer name=pgie config-file-path="{nvinfer_config_path}" !
            nvtracker name=tracker ll-config-file="{tracker_config_path}" !
            nvvideoconvert !
            video/x-raw,format=RGBA !
        """

        # Add output sink
        if output_sink == "appsink":
            pipeline_str += """
                appsink name=appsink emit-signals=True max-buffers=1 drop=True
            """
        elif output_sink == "autovideosink":
            pipeline_str += """
                autovideosink sync=False
            """
        elif output_sink == "filesink":
            pipeline_str += """
                nvvideoconvert !
                nvv4l2h264enc !
                h264parse !
                mp4mux !
                filesink location=output.mp4
            """

        # Parse pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)

        if self.pipeline is None:
            raise RuntimeError("Failed to create GStreamer pipeline")

        # Set up appsink callback if using appsink
        if output_sink == "appsink":
            appsink = self.pipeline.get_by_name("appsink")
            if appsink:
                appsink.connect("new-sample", self._on_new_sample)

        # Set up bus for messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("Pipeline built successfully")
        return self.pipeline

    def build_simple_pipeline(
        self,
        video_path: Union[str, Path],
    ) -> Gst.Pipeline:
        """
        Build a simplified pipeline without DeepStream (for testing).

        Args:
            video_path: Input video file path

        Returns:
            Simple GStreamer pipeline
        """
        video_path = str(video_path)

        pipeline_str = f"""
            filesrc location="{video_path}" !
            decodebin !
            videoconvert !
            video/x-raw,format=RGB !
            appsink name=appsink emit-signals=True max-buffers=1 drop=True
        """

        self.pipeline = Gst.parse_launch(pipeline_str)

        if self.pipeline is None:
            raise RuntimeError("Failed to create GStreamer pipeline")

        appsink = self.pipeline.get_by_name("appsink")
        if appsink:
            appsink.connect("new-sample", self._on_new_sample)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        return self.pipeline

    def set_frame_callback(self, callback: Callable) -> None:
        """
        Set callback for frame processing.

        Callback signature: callback(frame_data, width, height, frame_idx)
        """
        self._frame_callback = callback

    def start(self) -> None:
        """Start the pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline first.")

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start pipeline")

        self._is_running = True
        logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self._is_running = False
        logger.info("Pipeline stopped")

    def run_blocking(self) -> None:
        """Run the pipeline in blocking mode."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built")

        self.start()
        self.loop = GLib.MainLoop()

        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """Handle new sample from appsink."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        if self._frame_callback:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)

            width = structure.get_value("width")
            height = structure.get_value("height")

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                try:
                    self._frame_callback(
                        map_info.data,
                        width,
                        height,
                        buffer.pts,
                    )
                finally:
                    buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message) -> None:
        """Handle bus messages."""
        msg_type = message.type

        if msg_type == Gst.MessageType.EOS:
            logger.info("End of stream")
            if self.loop:
                self.loop.quit()

        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Pipeline error: {err.message}")
            logger.debug(f"Debug info: {debug}")
            if self.loop:
                self.loop.quit()

        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"Pipeline warning: {warn.message}")

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running


def create_file_processing_pipeline(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    nvinfer_config: Union[str, Path],
    tracker_config: Union[str, Path],
) -> str:
    """
    Create a GStreamer pipeline string for file-to-file processing.

    Args:
        video_path: Input video path
        output_path: Output video path
        nvinfer_config: Path to nvinfer config
        tracker_config: Path to tracker config

    Returns:
        GStreamer pipeline string
    """
    return f"""
        filesrc location="{video_path}" !
        qtdemux ! h264parse !
        nvv4l2decoder enable-max-performance=1 !
        nvstreammux name=mux batch-size=1 width=1920 height=1080 !
        nvinfer config-file-path="{nvinfer_config}" !
        nvtracker ll-config-file="{tracker_config}" !
        nvsegvisual width=1920 height=1080 !
        nvvideoconvert !
        nvv4l2h264enc !
        h264parse !
        mp4mux !
        filesink location="{output_path}"
    """
