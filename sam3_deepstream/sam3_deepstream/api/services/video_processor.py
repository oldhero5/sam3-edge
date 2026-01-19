"""Video processing service for DeepStream pipeline orchestration."""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ...config import SAM3DeepStreamConfig, get_config
from ...inference.keyframe_processor import FrameResult, KeyframeProcessor
from ...inference.mask_propagation import MaskPropagator
from ...utils.mask_utils import encode_rle, visualize_masks
from ..models.requests import OutputFormat, VideoProcessRequest

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Orchestrates video processing through the SAM3 pipeline.

    Handles both DeepStream-based processing (when available)
    and fallback CPU/GPU processing.
    """

    def __init__(
        self,
        config: Optional[SAM3DeepStreamConfig] = None,
        use_deepstream: bool = True,
    ):
        """
        Initialize video processor.

        Args:
            config: Configuration object
            use_deepstream: Whether to use DeepStream pipeline
        """
        self.config = config or get_config()
        self.use_deepstream = use_deepstream and self._check_deepstream()

        self._processor: Optional[KeyframeProcessor] = None
        self._propagator: Optional[MaskPropagator] = None

    def _check_deepstream(self) -> bool:
        """Check if DeepStream is available."""
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)
            return True
        except:
            return False

    def initialize(
        self,
        encoder_engine,
        decoder_fn=None,
    ) -> None:
        """
        Initialize processing components.

        Args:
            encoder_engine: TensorRT encoder engine
            decoder_fn: Decoder function (optional)
        """
        self._propagator = MaskPropagator(use_vpi=True)

        self._processor = KeyframeProcessor(
            encoder_fn=encoder_engine,
            decoder_fn=decoder_fn,
            keyframe_interval=self.config.inference.keyframe_interval,
            max_objects=self.config.inference.max_objects,
        )
        self._processor.set_propagator(self._propagator)

    async def process_video(
        self,
        video_path: Path,
        request: VideoProcessRequest,
        output_path: Path,
        progress_callback=None,
    ) -> dict:
        """
        Process a video file.

        Args:
            video_path: Path to input video
            request: Processing request parameters
            output_path: Path for output
            progress_callback: Optional callback for progress updates

        Returns:
            Processing results dictionary
        """
        if self.use_deepstream:
            return await self._process_with_deepstream(
                video_path, request, output_path, progress_callback
            )
        else:
            return await self._process_with_opencv(
                video_path, request, output_path, progress_callback
            )

    async def _process_with_deepstream(
        self,
        video_path: Path,
        request: VideoProcessRequest,
        output_path: Path,
        progress_callback=None,
    ) -> dict:
        """Process video using DeepStream pipeline."""
        from ...pipeline.gst_pipeline import SAM3Pipeline
        from ...pipeline.deepstream_config import generate_nvinfer_config
        from ...pipeline.tracker_config import generate_tracker_config

        # Generate config files
        temp_dir = Path(tempfile.mkdtemp())
        nvinfer_config = temp_dir / "nvinfer.txt"
        tracker_config = temp_dir / "tracker.yml"

        encoder_path = self.config.trt.cache_dir / "sam3_encoder.engine"
        generate_nvinfer_config(nvinfer_config, encoder_path)
        generate_tracker_config(tracker_config)

        # Build and run pipeline
        pipeline = SAM3Pipeline(self.config)

        results: List[FrameResult] = []

        def frame_callback(data, width, height, pts):
            # Convert to numpy
            frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Process frame
            if self._processor:
                result = self._processor.process_frame(frame)
                results.append(result)

                if progress_callback:
                    progress = len(results) / 1000  # Estimate
                    progress_callback(progress)

        pipeline.set_frame_callback(frame_callback)
        pipeline.build_pipeline(
            video_path,
            nvinfer_config,
            tracker_config,
        )

        # Run in thread pool
        await asyncio.to_thread(pipeline.run_blocking)

        # Generate output
        return await self._generate_output(
            video_path, results, request, output_path
        )

    async def _process_with_opencv(
        self,
        video_path: Path,
        request: VideoProcessRequest,
        output_path: Path,
        progress_callback=None,
    ) -> dict:
        """Process video using OpenCV (fallback)."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        results: List[FrameResult] = []

        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            if self._processor:
                result = self._processor.process_frame(frame_rgb)
            else:
                result = FrameResult(
                    frame_idx=frame_idx,
                    is_keyframe=False,
                    masks=[],
                    boxes=[],
                    scores=[],
                    object_ids=[],
                )

            results.append(result)
            frame_idx += 1

            if progress_callback and frame_idx % 10 == 0:
                progress = frame_idx / total_frames
                await asyncio.to_thread(progress_callback, progress)

        cap.release()

        # Generate output
        return await self._generate_output(
            video_path, results, request, output_path
        )

    async def _generate_output(
        self,
        video_path: Path,
        results: List[FrameResult],
        request: VideoProcessRequest,
        output_path: Path,
    ) -> dict:
        """Generate output file based on request format."""
        if request.output_format == OutputFormat.VIDEO:
            return await self._generate_video_output(
                video_path, results, output_path
            )
        else:
            return await self._generate_masks_output(
                results, output_path
            )

    async def _generate_video_output(
        self,
        video_path: Path,
        results: List[FrameResult],
        output_path: Path,
    ) -> dict:
        """Generate video with mask overlays."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < len(results):
                result = results[frame_idx]

                # Overlay masks
                if result.masks:
                    frame = visualize_masks(
                        frame,
                        result.masks,
                        alpha=0.4,
                    )

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return {
            "output_path": str(output_path),
            "frames_processed": len(results),
            "format": "video",
        }

    async def _generate_masks_output(
        self,
        results: List[FrameResult],
        output_path: Path,
    ) -> dict:
        """Generate JSON with RLE-encoded masks."""
        output_data = {
            "frames": [],
        }

        for result in results:
            frame_data = {
                "frame_idx": result.frame_idx,
                "is_keyframe": result.is_keyframe,
                "objects": [],
            }

            for i, (mask, box, score, obj_id) in enumerate(
                zip(result.masks, result.boxes, result.scores, result.object_ids)
            ):
                rle = encode_rle(mask)
                frame_data["objects"].append({
                    "object_id": obj_id,
                    "rle": rle,
                    "box": box,
                    "score": score,
                })

            output_data["frames"].append(frame_data)

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(output_data, f)

        return {
            "output_path": str(output_path),
            "frames_processed": len(results),
            "format": "masks",
        }
