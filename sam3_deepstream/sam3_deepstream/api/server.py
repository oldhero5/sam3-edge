"""
FastAPI server for SAM3 TensorRT inference.

Provides REST API endpoints for:
- Image encoding (extract features)
- Mask decoding (generate segmentation masks)
- Health checks and metrics
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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global inference runtime (initialized on startup)
_runtime = None
_async_runtime = None
_startup_time = None


# ============================================================================
# Pydantic Models
# ============================================================================

class PointPrompt(BaseModel):
    """Point prompt for mask generation."""
    x: float = Field(..., description="X coordinate (0-1 normalized or pixel)")
    y: float = Field(..., description="Y coordinate (0-1 normalized or pixel)")
    label: int = Field(1, description="1 for foreground, 0 for background")


class BoxPrompt(BaseModel):
    """Bounding box prompt for mask generation."""
    x1: float = Field(..., description="Top-left X")
    y1: float = Field(..., description="Top-left Y")
    x2: float = Field(..., description="Bottom-right X")
    y2: float = Field(..., description="Bottom-right Y")


class EncodeRequest(BaseModel):
    """Request for image encoding."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    return_embeddings: bool = Field(False, description="Return raw embeddings")


class EncodeResponse(BaseModel):
    """Response from image encoding."""
    success: bool
    embedding_shape: List[int]
    inference_time_ms: float
    embeddings_base64: Optional[str] = None


class DecodeRequest(BaseModel):
    """Request for mask decoding."""
    points: Optional[List[PointPrompt]] = Field(None, description="Point prompts")
    boxes: Optional[List[BoxPrompt]] = Field(None, description="Box prompts")
    mask_input: Optional[str] = Field(None, description="Base64 encoded mask input")
    multimask_output: bool = Field(True, description="Return multiple masks")


class DecodeResponse(BaseModel):
    """Response from mask decoding."""
    success: bool
    num_masks: int
    mask_shape: List[int]
    iou_predictions: List[float]
    inference_time_ms: float
    masks_base64: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    encoder_loaded: bool
    decoder_loaded: bool
    uptime_seconds: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None


class InferenceStats(BaseModel):
    """Inference statistics."""
    total_requests: int
    encode_requests: int
    decode_requests: int
    avg_encode_time_ms: float
    avg_decode_time_ms: float
    errors: int


# ============================================================================
# Inference Runtime
# ============================================================================

class InferenceRuntime:
    """Manages TRT inference engines and caching."""

    def __init__(
        self,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        device: int = 0,
        use_async: bool = True,
    ):
        self.device = device
        self.use_async = use_async
        self.encoder = None
        self.decoder = None
        self._cached_embeddings = None
        self._cached_image_shape = None

        # Stats
        self.stats = {
            'total_requests': 0,
            'encode_requests': 0,
            'decode_requests': 0,
            'encode_times': [],
            'decode_times': [],
            'errors': 0,
        }

        # Load engines
        if encoder_path and Path(encoder_path).exists():
            self._load_encoder(encoder_path)

        if decoder_path and Path(decoder_path).exists():
            self._load_decoder(decoder_path)

    def _load_encoder(self, path: str) -> None:
        """Load encoder TRT engine."""
        try:
            import torch
            torch.cuda.set_device(self.device)

            if self.use_async:
                from ..inference.async_trt_runtime import AsyncTRTInferenceEngine
                self.encoder = AsyncTRTInferenceEngine(path, device=self.device)
            else:
                from ..inference.trt_runtime import TRTInferenceEngine
                self.encoder = TRTInferenceEngine(path, device=self.device)

            logger.info(f"Encoder loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load encoder: {e}")
            raise

    def _load_decoder(self, path: str) -> None:
        """Load decoder TRT engine."""
        try:
            import torch
            torch.cuda.set_device(self.device)

            from ..inference.trt_runtime import TRTInferenceEngine
            self.decoder = TRTInferenceEngine(path, device=self.device)
            logger.info(f"Decoder loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load decoder: {e}")
            raise

    def encode_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Encode image to embeddings.

        Args:
            image: Input image (H, W, 3) RGB uint8

        Returns:
            Tuple of (embeddings, inference_time_ms)
        """
        import torch
        import torch.nn.functional as F

        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")

        self.stats['total_requests'] += 1
        self.stats['encode_requests'] += 1

        start = time.perf_counter()

        try:
            # Preprocess image
            img_tensor = torch.from_numpy(image).float()
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            img_tensor = img_tensor / 255.0

            # Resize to model input size (1008x1008)
            img_tensor = F.interpolate(
                img_tensor, size=(1008, 1008), mode='bilinear', align_corners=False
            )

            img_tensor = img_tensor.to(f'cuda:{self.device}')

            # Run encoder
            embeddings = self.encoder(img_tensor)

            # Cache for subsequent decode calls
            self._cached_embeddings = embeddings
            self._cached_image_shape = image.shape[:2]

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats['encode_times'].append(elapsed_ms)

            return embeddings.cpu().numpy(), elapsed_ms

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Encode error: {e}")
            raise

    def decode_masks(
        self,
        points: Optional[List[Tuple[float, float, int]]] = None,
        boxes: Optional[List[Tuple[float, float, float, float]]] = None,
        mask_input: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Decode masks from prompts.

        Args:
            points: List of (x, y, label) tuples
            boxes: List of (x1, y1, x2, y2) tuples
            mask_input: Optional previous mask
            embeddings: Optional embeddings (uses cached if not provided)

        Returns:
            Tuple of (masks, iou_predictions, inference_time_ms)
        """
        import torch

        if self.decoder is None:
            raise RuntimeError("Decoder not loaded")

        self.stats['total_requests'] += 1
        self.stats['decode_requests'] += 1

        start = time.perf_counter()

        try:
            # Get embeddings
            if embeddings is not None:
                emb_tensor = torch.from_numpy(embeddings).to(f'cuda:{self.device}')
            elif self._cached_embeddings is not None:
                emb_tensor = self._cached_embeddings
            else:
                raise ValueError("No embeddings provided and none cached")

            # Build sparse prompts from points/boxes
            sparse_prompts = self._build_sparse_prompts(points, boxes)

            # Build dense prompts from mask input
            if mask_input is not None:
                dense_prompts = torch.from_numpy(mask_input).to(f'cuda:{self.device}')
            else:
                # Zero dense prompts
                B, C, H, W = emb_tensor.shape
                dense_prompts = torch.zeros(1, C, H, W, device=emb_tensor.device)

            # Create positional encoding (simplified)
            B, C, H, W = emb_tensor.shape
            image_pe = torch.zeros(1, C, H, W, device=emb_tensor.device)

            # Run decoder
            outputs = self.decoder.infer(
                image_embeddings=emb_tensor,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompts,
                dense_prompt_embeddings=dense_prompts,
            )

            masks = outputs.get('masks', outputs.get(list(outputs.keys())[0]))
            iou_pred = outputs.get('iou_predictions', outputs.get(list(outputs.keys())[-1]))

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats['decode_times'].append(elapsed_ms)

            return masks.cpu().numpy(), iou_pred.cpu().numpy(), elapsed_ms

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Decode error: {e}")
            raise

    def _build_sparse_prompts(
        self,
        points: Optional[List[Tuple[float, float, int]]],
        boxes: Optional[List[Tuple[float, float, float, float]]],
    ):
        """Build sparse prompt tensor from points and boxes."""
        import torch

        prompts = []
        dim = 256  # Transformer dim

        if points:
            for x, y, label in points:
                # Simple positional encoding
                prompt = torch.zeros(dim)
                prompt[0] = x
                prompt[1] = y
                prompt[2] = label
                prompts.append(prompt)

        if boxes:
            for x1, y1, x2, y2 in boxes:
                prompt = torch.zeros(dim)
                prompt[0] = x1
                prompt[1] = y1
                prompt[2] = x2
                prompt[3] = y2
                prompt[4] = 2  # Box indicator
                prompts.append(prompt)

        if not prompts:
            # Default single point at center
            prompt = torch.zeros(dim)
            prompt[0] = 0.5
            prompt[1] = 0.5
            prompt[2] = 1
            prompts.append(prompt)

        sparse = torch.stack(prompts).unsqueeze(0)  # (1, N, dim)
        return sparse.to(f'cuda:{self.device}')

    def get_stats(self) -> Dict:
        """Get inference statistics."""
        encode_times = self.stats['encode_times'][-100:]  # Last 100
        decode_times = self.stats['decode_times'][-100:]

        return {
            'total_requests': self.stats['total_requests'],
            'encode_requests': self.stats['encode_requests'],
            'decode_requests': self.stats['decode_requests'],
            'avg_encode_time_ms': sum(encode_times) / max(1, len(encode_times)),
            'avg_decode_time_ms': sum(decode_times) / max(1, len(decode_times)),
            'errors': self.stats['errors'],
        }

    def clear_cache(self) -> None:
        """Clear cached embeddings."""
        self._cached_embeddings = None
        self._cached_image_shape = None


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _runtime, _startup_time

    logger.info("Starting SAM3 FastAPI server...")
    _startup_time = time.time()

    # Get engine paths from environment
    encoder_path = os.environ.get(
        'SAM3_ENCODER_ENGINE',
        '/workspace/sam3_deepstream/engines/sam3_encoder.engine'
    )
    decoder_path = os.environ.get(
        'SAM3_DECODER_ENGINE',
        '/workspace/sam3_deepstream/engines/sam3_decoder.engine'
    )
    device = int(os.environ.get('CUDA_DEVICE', '0'))
    use_async = os.environ.get('USE_ASYNC', 'true').lower() == 'true'

    # Initialize runtime
    try:
        _runtime = InferenceRuntime(
            encoder_path=encoder_path if Path(encoder_path).exists() else None,
            decoder_path=decoder_path if Path(decoder_path).exists() else None,
            device=device,
            use_async=use_async,
        )
        logger.info("Inference runtime initialized")
    except Exception as e:
        logger.warning(f"Could not initialize full runtime: {e}")
        _runtime = InferenceRuntime(device=device, use_async=use_async)

    yield

    # Cleanup
    logger.info("Shutting down SAM3 FastAPI server...")
    if _runtime:
        _runtime.clear_cache()


app = FastAPI(
    title="SAM3 Inference API",
    description="TensorRT-accelerated SAM3 segmentation inference API",
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
# API Endpoints
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

    return HealthResponse(
        status="healthy" if _runtime else "degraded",
        encoder_loaded=_runtime.encoder is not None if _runtime else False,
        decoder_loaded=_runtime.decoder is not None if _runtime else False,
        uptime_seconds=uptime,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_memory_used,
        gpu_memory_total_mb=gpu_memory_total,
    )


@app.get("/stats", response_model=InferenceStats)
async def get_stats():
    """Get inference statistics."""
    if not _runtime:
        raise HTTPException(status_code=503, detail="Runtime not initialized")

    stats = _runtime.get_stats()
    return InferenceStats(**stats)


@app.post("/encode", response_model=EncodeResponse)
async def encode_image(
    file: Optional[UploadFile] = File(None),
    request: Optional[EncodeRequest] = None,
):
    """
    Encode image to embeddings.

    Upload an image file or provide base64-encoded image data.
    """
    if not _runtime or not _runtime.encoder:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    try:
        from PIL import Image

        # Get image data
        if file:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif request and request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        # Convert to numpy
        image_np = np.array(image)

        # Run encoding
        embeddings, inference_time = await asyncio.get_event_loop().run_in_executor(
            None, _runtime.encode_image, image_np
        )

        response = EncodeResponse(
            success=True,
            embedding_shape=list(embeddings.shape),
            inference_time_ms=inference_time,
        )

        # Optionally return embeddings
        if request and request.return_embeddings:
            emb_bytes = embeddings.astype(np.float16).tobytes()
            response.embeddings_base64 = base64.b64encode(emb_bytes).decode()

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Encode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decode", response_model=DecodeResponse)
async def decode_masks(request: DecodeRequest):
    """
    Decode masks from prompts.

    Requires prior call to /encode or cached embeddings.
    """
    if not _runtime or not _runtime.decoder:
        raise HTTPException(status_code=503, detail="Decoder not loaded")

    if _runtime._cached_embeddings is None:
        raise HTTPException(
            status_code=400,
            detail="No cached embeddings. Call /encode first."
        )

    try:
        # Convert prompts
        points = None
        if request.points:
            points = [(p.x, p.y, p.label) for p in request.points]

        boxes = None
        if request.boxes:
            boxes = [(b.x1, b.y1, b.x2, b.y2) for b in request.boxes]

        # Decode mask input if provided
        mask_input = None
        if request.mask_input:
            mask_bytes = base64.b64decode(request.mask_input)
            mask_input = np.frombuffer(mask_bytes, dtype=np.float32)

        # Run decoding
        masks, iou_pred, inference_time = await asyncio.get_event_loop().run_in_executor(
            None, _runtime.decode_masks, points, boxes, mask_input, None
        )

        # Encode masks as base64
        mask_bytes = masks.astype(np.float16).tobytes()
        masks_base64 = base64.b64encode(mask_bytes).decode()

        return DecodeResponse(
            success=True,
            num_masks=masks.shape[1],
            mask_shape=list(masks.shape),
            iou_predictions=iou_pred.flatten().tolist(),
            inference_time_ms=inference_time,
            masks_base64=masks_base64,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    points: Optional[str] = Query(None, description="Points as 'x1,y1,l1;x2,y2,l2'"),
    boxes: Optional[str] = Query(None, description="Boxes as 'x1,y1,x2,y2;...'"),
    return_mask_image: bool = Query(True, description="Return mask as PNG image"),
):
    """
    One-shot segmentation: encode image and decode masks in single request.

    Returns mask overlay as PNG image or JSON with base64 data.
    """
    if not _runtime:
        raise HTTPException(status_code=503, detail="Runtime not initialized")

    try:
        from PIL import Image

        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)

        # Parse prompts
        point_list = None
        if points:
            point_list = []
            for p in points.split(';'):
                parts = p.split(',')
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    label = int(parts[2]) if len(parts) > 2 else 1
                    point_list.append((x, y, label))

        box_list = None
        if boxes:
            box_list = []
            for b in boxes.split(';'):
                parts = b.split(',')
                if len(parts) >= 4:
                    box_list.append(tuple(float(x) for x in parts[:4]))

        # Encode
        embeddings, encode_time = await asyncio.get_event_loop().run_in_executor(
            None, _runtime.encode_image, image_np
        )

        # Decode
        masks, iou_pred, decode_time = await asyncio.get_event_loop().run_in_executor(
            None, _runtime.decode_masks, point_list, box_list, None, None
        )

        total_time = encode_time + decode_time

        if return_mask_image:
            # Create mask overlay
            import cv2

            # Use best mask (highest IoU)
            best_idx = np.argmax(iou_pred)
            mask = masks[0, best_idx]

            # Resize mask to image size
            mask_resized = cv2.resize(
                mask, (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Apply sigmoid and threshold
            mask_binary = (1 / (1 + np.exp(-mask_resized))) > 0.5

            # Create colored overlay
            overlay = image_np.copy()
            overlay[mask_binary] = overlay[mask_binary] * 0.5 + np.array([0, 255, 0]) * 0.5

            # Convert to PNG
            result_image = Image.fromarray(overlay.astype(np.uint8))
            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            return StreamingResponse(
                img_buffer,
                media_type="image/png",
                headers={
                    "X-Inference-Time-Ms": str(total_time),
                    "X-IoU-Score": str(iou_pred[0, best_idx]),
                }
            )
        else:
            # Return JSON
            mask_bytes = masks.astype(np.float16).tobytes()
            return JSONResponse({
                "success": True,
                "encode_time_ms": encode_time,
                "decode_time_ms": decode_time,
                "total_time_ms": total_time,
                "iou_predictions": iou_pred.flatten().tolist(),
                "mask_shape": list(masks.shape),
                "masks_base64": base64.b64encode(mask_bytes).decode(),
            })

    except Exception as e:
        logger.error(f"Segment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-cache")
async def clear_cache():
    """Clear cached embeddings."""
    if _runtime:
        _runtime.clear_cache()
    return {"status": "cache cleared"}


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the FastAPI server."""
    import uvicorn

    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8000'))
    workers = int(os.environ.get('WORKERS', '1'))

    uvicorn.run(
        "sam3_deepstream.api.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
