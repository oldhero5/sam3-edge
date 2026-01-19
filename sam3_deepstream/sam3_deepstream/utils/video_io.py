"""Video I/O utilities with NVDEC/NVENC support."""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Video reader with optional hardware acceleration.

    Supports NVDEC acceleration on Jetson platforms.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        use_nvdec: bool = True,
    ):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file
            use_nvdec: Use NVDEC hardware decoder if available
        """
        self.video_path = Path(video_path)
        self.use_nvdec = use_nvdec
        self._cap: Optional[cv2.VideoCapture] = None
        self._opened = False

    def open(self) -> bool:
        """Open video file."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Try NVDEC first if requested
        if self.use_nvdec:
            try:
                self._cap = cv2.VideoCapture(
                    str(self.video_path),
                    cv2.CAP_FFMPEG,
                )
                # Set NVIDIA hardware decode preference
                self._cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                self._cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
            except Exception as e:
                logger.warning(f"NVDEC not available: {e}")
                self._cap = cv2.VideoCapture(str(self.video_path))
        else:
            self._cap = cv2.VideoCapture(str(self.video_path))

        self._opened = self._cap.isOpened()
        return self._opened

    def close(self) -> None:
        """Close video file."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._opened = False

    @property
    def is_opened(self) -> bool:
        """Check if video is opened."""
        return self._opened and self._cap is not None

    @property
    def fps(self) -> float:
        """Get video FPS."""
        if not self._cap:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Get total frame count."""
        if not self._cap:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        """Get frame width."""
        if not self._cap:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Get frame height."""
        if not self._cap:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def size(self) -> Tuple[int, int]:
        """Get frame size (width, height)."""
        return (self.width, self.height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame."""
        if not self._cap:
            return False, None
        return self._cap.read()

    def read_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame in RGB format."""
        ret, frame = self.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame

    def seek(self, frame_idx: int) -> bool:
        """Seek to frame index."""
        if not self._cap:
            return False
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def iter_frames(
        self,
        rgb: bool = True,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate over all frames.

        Args:
            rgb: Convert to RGB format

        Yields:
            Tuple of (frame_idx, frame)
        """
        frame_idx = 0
        while True:
            if rgb:
                ret, frame = self.read_rgb()
            else:
                ret, frame = self.read()

            if not ret or frame is None:
                break

            yield frame_idx, frame
            frame_idx += 1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoWriter:
    """
    Video writer with optional hardware acceleration.

    Supports NVENC acceleration on Jetson platforms.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        size: Tuple[int, int],
        codec: str = "mp4v",
        use_nvenc: bool = True,
    ):
        """
        Initialize video writer.

        Args:
            output_path: Output video path
            fps: Frames per second
            size: Frame size (width, height)
            codec: Video codec (mp4v, avc1, h264)
            use_nvenc: Use NVENC hardware encoder if available
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.size = size
        self.codec = codec
        self.use_nvenc = use_nvenc
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> bool:
        """Open video writer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        # Try NVENC if available
        if self.use_nvenc and self.codec.lower() in ("h264", "hevc"):
            try:
                # Use GStreamer pipeline for NVENC
                gst_str = (
                    f"appsrc ! videoconvert ! nvv4l2h264enc ! "
                    f"h264parse ! mp4mux ! filesink location={self.output_path}"
                )
                self._writer = cv2.VideoWriter(
                    gst_str,
                    cv2.CAP_GSTREAMER,
                    0,
                    self.fps,
                    self.size,
                )
                if self._writer.isOpened():
                    logger.info("Using NVENC hardware encoder")
                    return True
            except Exception as e:
                logger.warning(f"NVENC not available: {e}")

        # Fallback to software encoding
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.size,
        )

        return self._writer.isOpened()

    def close(self) -> None:
        """Close video writer."""
        if self._writer:
            self._writer.release()
            self._writer = None

    def write(self, frame: np.ndarray) -> bool:
        """Write frame (BGR format expected)."""
        if not self._writer:
            return False
        self._writer.write(frame)
        return True

    def write_rgb(self, frame: np.ndarray) -> bool:
        """Write frame in RGB format."""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return self.write(frame_bgr)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_video_info(video_path: Union[str, Path]) -> dict:
    """Get video file information."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    info = {
        "path": str(video_path),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info
