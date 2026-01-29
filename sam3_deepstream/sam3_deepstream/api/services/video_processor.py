"""Video processing service for DeepStream pipeline orchestration."""

import asyncio
import hashlib
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ...config import SAM3DeepStreamConfig, get_config
from ...inference.keyframe_processor import FrameResult, KeyframeProcessor
from ...inference.mask_propagation import MaskPropagator
from ...utils.mask_utils import encode_rle, visualize_masks
from ..models.requests import OutputFormat, VideoProcessRequest
from .detection_store import Detection, DetectionStore, VideoMetadata, get_detection_store

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
        self._sam3_processor = None  # Cached SAM3 processor for text prompts

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Process a video file.

        Args:
            video_path: Path to input video
            request: Processing request parameters
            output_path: Path for output
            progress_callback: Optional callback for progress updates
            metadata: Optional metadata (text_prompt, store_masks, store_embeddings)

        Returns:
            Processing results dictionary
        """
        metadata = metadata or {}
        store_detections = metadata.get("store_masks", True) or metadata.get("store_embeddings", True)

        # Use text prompt processing if text_prompt is provided
        if request.text_prompt:
            return await self._process_with_text_prompt(
                video_path, request, output_path, progress_callback, metadata
            )
        elif self.use_deepstream:
            return await self._process_with_deepstream(
                video_path, request, output_path, progress_callback
            )
        else:
            return await self._process_with_opencv(
                video_path, request, output_path, progress_callback
            )

    async def _process_with_text_prompt(
        self,
        video_path: Path,
        request: VideoProcessRequest,
        output_path: Path,
        progress_callback=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Process video using SAM3's text-based segmentation.

        Uses Sam3Processor.set_text_prompt() for natural language object detection.
        """
        metadata = metadata or {}
        store_masks = metadata.get("store_masks", True)
        store_embeddings = metadata.get("store_embeddings", True)

        # Import SAM3 components
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            logger.error(f"SAM3 not available for text prompt processing: {e}")
            # Fallback to standard processing
            return await self._process_with_opencv(
                video_path, request, output_path, progress_callback
            )

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Generate video_id and compute hash
        video_id = str(uuid.uuid4())
        video_hash = self._compute_file_hash(video_path)

        # Store video metadata
        detection_store = await get_detection_store()
        await detection_store.store_video(VideoMetadata(
            video_id=video_id,
            device_id=self.config.federation.device_id,
            filename=video_path.name,
            file_hash=video_hash,
            duration_seconds=total_frames / fps if fps > 0 else None,
            width=width,
            height=height,
            total_frames=total_frames,
        ))

        # Embedding service removed (NLQ feature not implemented)
        prompt_embedding = None

        # Get or create cached SAM3 processor (avoid reloading model for each video)
        if self._sam3_processor is None:
            try:
                from sam3.model_builder import build_sam3_hiera_l

                checkpoint_path = self.config.sam3_checkpoint
                if checkpoint_path is None:
                    # Try common locations
                    for path in [
                        Path("/workspace/checkpoints/sam3.pt"),
                        Path.home() / ".cache" / "sam3" / "sam3.pt",
                    ]:
                        if path.exists():
                            checkpoint_path = path
                            break

                logger.info(f"Loading SAM3 model from {checkpoint_path}")
                model = build_sam3_hiera_l(
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                    device="cuda",
                    eval_mode=True,
                    load_from_HF=False,
                )
                self._sam3_processor = Sam3Processor(
                    model,
                    resolution=1008,
                    device="cuda",
                    confidence_threshold=0.5,
                )
                logger.info("SAM3 model loaded and cached for video processing")
            except Exception as e:
                logger.error(f"Failed to build SAM3 model: {e}")
                cap.release()
                return {"error": str(e), "frames_processed": 0}

        # Update confidence threshold for this request
        processor = self._sam3_processor
        processor.confidence_threshold = request.segmentation_threshold

        results: List[FrameResult] = []
        detections: List[Detection] = []
        detection_count = 0

        # Process frames
        frame_idx = 0
        keyframe_interval = request.keyframe_interval

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = (frame_idx / fps * 1000) if fps > 0 else None

            is_keyframe = (frame_idx % keyframe_interval == 0)

            if is_keyframe:
                # Run full inference with text prompt
                try:
                    from PIL import Image
                    pil_image = Image.fromarray(frame_rgb)

                    state = processor.set_image(pil_image)
                    state = processor.set_text_prompt(request.text_prompt, state)

                    import torch

                    # Debug: log state keys
                    logger.info(f"Frame {frame_idx} SAM3 state keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")

                    # Extract results - match working server endpoint approach
                    masks = state.get("masks", torch.zeros(0, 1, height, width))
                    boxes = state.get("boxes", torch.zeros(0, 4))
                    scores = state.get("scores", torch.zeros(0))

                    # Convert to numpy if tensor
                    if torch.is_tensor(masks):
                        masks = masks.cpu().numpy()
                    if torch.is_tensor(boxes):
                        boxes = boxes.cpu().numpy()
                    if torch.is_tensor(scores):
                        scores = scores.cpu().numpy()

                    logger.info(f"Frame {frame_idx} extracted: {len(masks) if hasattr(masks, '__len__') else 0} masks")

                    frame_masks = []
                    frame_boxes = []
                    frame_scores = []
                    frame_object_ids = []

                    num_detections = len(scores) if hasattr(scores, '__len__') else 0
                    for obj_idx in range(num_detections):
                        score = float(scores[obj_idx])
                        if score < request.segmentation_threshold:
                            continue

                        # Process mask - squeeze and resize to frame size
                        mask = masks[obj_idx].squeeze()
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask.astype(np.float32), (width, height))
                        mask_np = (mask > 0.5).astype(np.uint8)

                        # Get box
                        box = boxes[obj_idx] if obj_idx < len(boxes) else np.array([0, 0, 0, 0])
                        # Normalize box to 0-1
                        box_norm = (
                            float(box[0]) / width,
                            float(box[1]) / height,
                            float(box[2]) / width,
                            float(box[3]) / height,
                        )

                        frame_masks.append(mask_np)
                        frame_boxes.append(box_norm)
                        frame_scores.append(float(score))
                        frame_object_ids.append(obj_idx)

                        # Create detection for storage
                        if store_masks or store_embeddings:
                            rle_dict = encode_rle(mask_np) if store_masks else None
                            # Serialize to JSON string for SQLite storage
                            rle_str = json.dumps(rle_dict) if rle_dict else None
                            detection = Detection(
                                video_id=video_id,
                                device_id=self.config.federation.device_id,
                                frame_idx=frame_idx,
                                timestamp_ms=timestamp_ms,
                                object_id=obj_idx,
                                text_prompt=request.text_prompt,
                                label=request.text_prompt.split()[0] if request.text_prompt else None,
                                confidence=float(score),
                                bbox=box_norm,
                                mask_rle=rle_str,
                            )
                            detections.append(detection)
                            detection_count += 1

                    result = FrameResult(
                        frame_idx=frame_idx,
                        is_keyframe=True,
                        masks=frame_masks,
                        boxes=frame_boxes,
                        scores=frame_scores,
                        object_ids=frame_object_ids,
                    )

                except Exception as e:
                    logger.error(f"Frame {frame_idx} processing error: {e}")
                    result = FrameResult(
                        frame_idx=frame_idx,
                        is_keyframe=True,
                        masks=[],
                        boxes=[],
                        scores=[],
                        object_ids=[],
                    )
            else:
                # Propagate from previous frame (simplified)
                if results and results[-1].masks:
                    result = FrameResult(
                        frame_idx=frame_idx,
                        is_keyframe=False,
                        masks=results[-1].masks,
                        boxes=results[-1].boxes,
                        scores=results[-1].scores,
                        object_ids=results[-1].object_ids,
                    )
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

        # Store detections in batch
        if detections and store_embeddings and prompt_embedding is not None:
            # Use same embedding for all detections from same prompt
            embeddings = np.tile(prompt_embedding, (len(detections), 1))
            await detection_store.store_detections_batch(detections, embeddings)
        elif detections:
            await detection_store.store_detections_batch(detections)

        # Generate output
        output_result = await self._generate_output(
            video_path, results, request, output_path
        )

        output_result["video_id"] = video_id
        output_result["detection_count"] = detection_count
        output_result["text_prompt"] = request.text_prompt

        # Clean up GPU memory after processing
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return output_result

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

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
        """Generate output files - always creates both video AND JSON."""
        # Always generate both video with overlays AND JSON with masks
        video_output_path = output_path.with_suffix(".mp4")
        json_output_path = output_path.with_suffix(".json")

        video_result = await self._generate_video_output(
            video_path, results, video_output_path
        )
        json_result = await self._generate_masks_output(
            results, json_output_path
        )

        # Return video path as main output, include JSON path
        return {
            "output_path": str(video_output_path),
            "json_path": str(json_output_path),
            "frames_processed": video_result["frames_processed"],
            "format": "video+json",
        }

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
