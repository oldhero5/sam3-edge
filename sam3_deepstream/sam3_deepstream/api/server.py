"""
SAM3 Unified API Server

Uses SAM3's native VETextEncoder for text prompt support.
Provides REST API endpoints for image segmentation with text, point, or box prompts.
"""

import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global runtime
_runtime = None
_startup_time = None


# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    backbone_type: Optional[str] = None  # "PE" or "ViT"


class InferenceStats(BaseModel):
    """Inference statistics."""
    total_requests: int
    text_requests: int
    point_requests: int
    avg_inference_time_ms: float
    errors: int


class DetectionResult(BaseModel):
    """Single detection result."""
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] in pixels")
    score: float
    mask_area: int


class SegmentResponse(BaseModel):
    """Response from segmentation."""
    success: bool
    inference_time_ms: float
    num_detections: int
    detections: List[DetectionResult]


# ============================================================================
# SAM3 Runtime (Native Text Support)
# ============================================================================

class SAM3Runtime:
    """
    Runtime using SAM3's native Sam3Processor with VETextEncoder.

    SAM3 has built-in vision-language capabilities - no separate CLIP needed.

    Supports optional PE (Perception Encoder) backbone for improved
    text prompt understanding via alignment tuning.

    Set SAM3_USE_PE_BACKBONE=1 to enable PE backbone.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        resolution: int = 1008,
        use_pe_backbone: Optional[bool] = None,
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.resolution = resolution
        self.processor = None
        self.model = None

        # Check PE configuration from environment
        if use_pe_backbone is None:
            use_pe_backbone = os.environ.get('SAM3_USE_PE_BACKBONE', '0') == '1'
        self.use_pe_backbone = use_pe_backbone
        self.use_alignment_tuning = os.environ.get('SAM3_ALIGNMENT_TUNING', '1') == '1'

        # Stats
        self.stats = {
            'total_requests': 0,
            'text_requests': 0,
            'point_requests': 0,
            'inference_times': [],
            'errors': 0,
            'backbone_type': 'PE' if self.use_pe_backbone else 'ViT',
        }

    def load_model(self):
        """Load SAM3 model with native text encoder or PE backbone."""
        import torch

        if not Path(self.checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {self.checkpoint_path}")
            return False

        try:
            from sam3.model.sam3_image_processor import Sam3Processor

            if self.use_pe_backbone:
                # Load PE-enhanced model
                from sam3.model_builder import build_sam3_pe_model

                logger.info(f"Loading SAM3 with PE backbone from {self.checkpoint_path}...")
                logger.info(f"Alignment tuning: {self.use_alignment_tuning}")

                self.model = build_sam3_pe_model(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    eval_mode=True,
                    load_from_HF=False,
                    use_alignment_tuning=self.use_alignment_tuning,
                )
                backbone_type = "PE (Perception Encoder)"
            else:
                # Load standard model
                from sam3.model_builder import build_sam3_hiera_l

                logger.info(f"Loading SAM3 model from {self.checkpoint_path}...")
                self.model = build_sam3_hiera_l(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    eval_mode=True,
                    load_from_HF=False,
                )
                backbone_type = "ViT (standard)"

            self.processor = Sam3Processor(
                self.model,
                resolution=self.resolution,
                device=self.device,
                confidence_threshold=0.5
            )

            logger.info(f"SAM3 model loaded successfully with {backbone_type} backbone")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            self.stats['errors'] += 1
            return False

    def segment_with_text(self, image: np.ndarray, text_prompt: str) -> Dict:
        """
        Segment image using text prompt via SAM3's native VETextEncoder.

        Args:
            image: RGB image as numpy array (H, W, 3)
            text_prompt: Text description of object to segment

        Returns:
            Dict with masks, boxes, scores
        """
        import torch
        from PIL import Image

        if self.processor is None:
            raise RuntimeError("Model not loaded")

        self.stats['total_requests'] += 1
        self.stats['text_requests'] += 1
        start = time.perf_counter()

        try:
            # Convert numpy to PIL for processor
            pil_image = Image.fromarray(image)

            # SAM3 native text processing
            state = self.processor.set_image(pil_image)
            state = self.processor.set_text_prompt(text_prompt, state)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats['inference_times'].append(elapsed_ms)

            # Extract results
            masks = state.get("masks", torch.zeros(0, 1, image.shape[0], image.shape[1]))
            boxes = state.get("boxes", torch.zeros(0, 4))
            scores = state.get("scores", torch.zeros(0))

            return {
                "masks": masks.cpu().numpy() if torch.is_tensor(masks) else masks,
                "boxes": boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes,
                "scores": scores.cpu().numpy() if torch.is_tensor(scores) else scores,
                "inference_time_ms": elapsed_ms,
            }

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Segmentation error: {e}")
            raise

    def segment_with_point(
        self,
        image: np.ndarray,
        points: List[Tuple[float, float, int]]
    ) -> Dict:
        """
        Segment image using point prompts.

        Args:
            image: RGB image as numpy array (H, W, 3)
            points: List of (x, y, label) where x,y are 0-1 normalized, label is 1=fg/0=bg

        Returns:
            Dict with masks, boxes, scores
        """
        import torch
        from PIL import Image

        if self.processor is None:
            raise RuntimeError("Model not loaded")

        self.stats['total_requests'] += 1
        self.stats['point_requests'] += 1
        start = time.perf_counter()

        try:
            pil_image = Image.fromarray(image)
            h, w = image.shape[:2]

            # Set image first
            state = self.processor.set_image(pil_image)

            # Add points as box prompts (point → small box)
            # SAM3 uses center_x, center_y, width, height format (normalized 0-1)
            for x, y, label in points:
                # Convert point to small box (5% of image size)
                box_size = 0.05
                box = [x, y, box_size, box_size]  # cx, cy, w, h
                state = self.processor.add_geometric_prompt(box, label == 1, state)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats['inference_times'].append(elapsed_ms)

            masks = state.get("masks", torch.zeros(0, 1, h, w))
            boxes = state.get("boxes", torch.zeros(0, 4))
            scores = state.get("scores", torch.zeros(0))

            return {
                "masks": masks.cpu().numpy() if torch.is_tensor(masks) else masks,
                "boxes": boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes,
                "scores": scores.cpu().numpy() if torch.is_tensor(scores) else scores,
                "inference_time_ms": elapsed_ms,
            }

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Point segmentation error: {e}")
            raise

    def get_stats(self) -> Dict:
        """Get inference statistics."""
        times = self.stats['inference_times'][-100:]
        return {
            'total_requests': self.stats['total_requests'],
            'text_requests': self.stats['text_requests'],
            'point_requests': self.stats['point_requests'],
            'avg_inference_time_ms': sum(times) / max(1, len(times)),
            'errors': self.stats['errors'],
        }


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _runtime, _startup_time

    logger.info("Starting SAM3 API Server...")
    _startup_time = time.time()

    # Get checkpoint path from environment
    checkpoint_path = os.environ.get(
        'SAM3_CHECKPOINT',
        '/workspace/checkpoints/sam3.pt'
    )
    # Use "cuda" for device - SAM3 builder only supports "cuda" or "cpu"
    # For multi-GPU, use CUDA_VISIBLE_DEVICES environment variable
    device = "cuda" if os.environ.get('CUDA_DEVICE', '0') != '-1' else "cpu"

    # Initialize runtime
    _runtime = SAM3Runtime(checkpoint_path=checkpoint_path, device=device)

    # Try to load model
    if Path(checkpoint_path).exists():
        try:
            _runtime.load_model()
        except Exception as e:
            logger.warning(f"Could not load model on startup: {e}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Server will start in degraded mode. Provide checkpoint to enable inference.")

    # Create upload directory
    upload_dir = Path("/workspace/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.upload_dir = upload_dir

    yield

    # Cleanup
    logger.info("Shutting down SAM3 API Server...")


app = FastAPI(
    title="SAM3 Inference API",
    description="SAM3 segmentation with native text prompt support via VETextEncoder",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and GPU status."""
    gpu_available = False
    gpu_name = None
    gpu_memory_used = None
    gpu_memory_total = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_used = torch.cuda.memory_allocated(0) / 1024 / 1024
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    except Exception:
        pass

    uptime = time.time() - _startup_time if _startup_time else 0
    model_loaded = _runtime is not None and _runtime.processor is not None
    backbone_type = _runtime.stats.get('backbone_type', 'unknown') if _runtime else None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_memory_used,
        gpu_memory_total_mb=gpu_memory_total,
        backbone_type=backbone_type,
    )


@app.get("/stats", response_model=InferenceStats)
async def get_stats():
    """Get inference statistics."""
    if not _runtime:
        raise HTTPException(status_code=503, detail="Runtime not initialized")

    stats = _runtime.get_stats()
    return InferenceStats(**stats)


@app.post("/api/v1/segment")
async def segment_with_text(
    file: UploadFile = File(...),
    text_prompt: str = Form(..., description="Text description of object to segment"),
    confidence_threshold: float = Form(0.5, description="Minimum confidence 0-1"),
    return_json: bool = Form(False, description="Return JSON instead of image"),
):
    """
    Segment objects using text prompt (SAM3 native VETextEncoder).

    Example:
        curl -X POST http://localhost:8000/api/v1/segment \\
          -F "file=@image.jpg" \\
          -F "text_prompt=red car" \\
          --output result.png
    """
    if not _runtime or not _runtime.processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from PIL import Image
        import cv2

        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Set confidence threshold
        _runtime.processor.confidence_threshold = confidence_threshold

        # Run SAM3 native text segmentation
        result = await asyncio.get_event_loop().run_in_executor(
            None, _runtime.segment_with_text, image_np, text_prompt
        )

        masks = result["masks"]
        boxes = result["boxes"]
        scores = result["scores"]
        inference_time = result["inference_time_ms"]

        num_detections = len(scores) if hasattr(scores, '__len__') else 0

        if return_json:
            # Return JSON response
            detections = []
            for i in range(num_detections):
                mask_area = int(masks[i].sum()) if i < len(masks) else 0
                detections.append(DetectionResult(
                    bbox=boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0],
                    score=float(scores[i]) if i < len(scores) else 0.0,
                    mask_area=mask_area,
                ))

            return SegmentResponse(
                success=True,
                inference_time_ms=inference_time,
                num_detections=num_detections,
                detections=detections,
            )
        else:
            # Create mask overlay image
            overlay = image_np.copy()

            if num_detections > 0:
                # Combine all masks
                combined_mask = np.zeros((h, w), dtype=bool)
                for i in range(num_detections):
                    if i < len(masks):
                        mask = masks[i].squeeze()
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                        combined_mask |= mask

                # Apply green overlay
                overlay[combined_mask] = (
                    overlay[combined_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
                ).astype(np.uint8)

                # Draw bounding boxes
                for i in range(min(num_detections, len(boxes))):
                    box = boxes[i].astype(int)
                    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    if i < len(scores):
                        label = f"{text_prompt}: {scores[i]:.2f}"
                        cv2.putText(overlay, label, (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Return as PNG
            result_image = Image.fromarray(overlay)
            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            return StreamingResponse(
                img_buffer,
                media_type="image/png",
                headers={
                    "X-Inference-Time-Ms": str(inference_time),
                    "X-Num-Detections": str(num_detections),
                    "X-Text-Prompt": text_prompt,
                }
            )

    except Exception as e:
        logger.error(f"Segment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
async def segment_with_point(
    file: UploadFile = File(...),
    points: Optional[str] = Query(None, description="Points as 'x1,y1,l1;x2,y2,l2' (coords 0-1)"),
    boxes: Optional[str] = Query(None, description="Boxes as 'x1,y1,x2,y2;...' (coords 0-1)"),
    return_json: bool = Query(False, description="Return JSON instead of image"),
):
    """
    Segment image with point or box prompts.

    Coordinates are normalized 0-1 (relative to image dimensions).

    Examples:
        - Point at center: points=0.5,0.5,1
        - Box around object: boxes=0.2,0.2,0.8,0.8
    """
    if not _runtime or not _runtime.processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from PIL import Image
        import cv2

        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Parse prompts
        point_list = []
        if points:
            for p in points.split(';'):
                parts = p.strip().split(',')
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    label = int(parts[2]) if len(parts) > 2 else 1
                    point_list.append((x, y, label))

        if not point_list:
            # Default: center point
            point_list = [(0.5, 0.5, 1)]

        # Run segmentation
        result = await asyncio.get_event_loop().run_in_executor(
            None, _runtime.segment_with_point, image_np, point_list
        )

        masks = result["masks"]
        result_boxes = result["boxes"]
        scores = result["scores"]
        inference_time = result["inference_time_ms"]

        num_detections = len(scores) if hasattr(scores, '__len__') else 0

        if return_json:
            detections = []
            for i in range(num_detections):
                mask_area = int(masks[i].sum()) if i < len(masks) else 0
                detections.append(DetectionResult(
                    bbox=result_boxes[i].tolist() if i < len(result_boxes) else [0, 0, 0, 0],
                    score=float(scores[i]) if i < len(scores) else 0.0,
                    mask_area=mask_area,
                ))

            return SegmentResponse(
                success=True,
                inference_time_ms=inference_time,
                num_detections=num_detections,
                detections=detections,
            )
        else:
            # Create mask overlay
            overlay = image_np.copy()

            if num_detections > 0:
                combined_mask = np.zeros((h, w), dtype=bool)
                for i in range(num_detections):
                    if i < len(masks):
                        mask = masks[i].squeeze()
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                        combined_mask |= mask

                overlay[combined_mask] = (
                    overlay[combined_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
                ).astype(np.uint8)

            # Draw input points
            for x, y, label in point_list:
                px, py = int(x * w), int(y * h)
                color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.circle(overlay, (px, py), 10, color, -1)
                cv2.circle(overlay, (px, py), 12, (255, 255, 255), 2)

            # Return as PNG
            result_image = Image.fromarray(overlay)
            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            return StreamingResponse(
                img_buffer,
                media_type="image/png",
                headers={
                    "X-Inference-Time-Ms": str(inference_time),
                    "X-Num-Detections": str(num_detections),
                }
            )

    except Exception as e:
        logger.error(f"Point segment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the FastAPI server."""
    import uvicorn

    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8000'))

    uvicorn.run(
        "sam3_deepstream.api.server:app",
        host=host,
        port=port,
        workers=1,  # Single worker for GPU
        log_level="info",
    )


if __name__ == "__main__":
    main()
