"""Health check endpoints."""

import logging

import torch
from fastapi import APIRouter, Request

from ..models.responses import EngineInfoResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Check service health status."""
    from ... import __version__

    # Check GPU availability
    gpu_available = torch.cuda.is_available()

    # Check TensorRT engines
    engine_manager = getattr(request.app.state, "engine_manager", None)
    engines_available = False
    if engine_manager:
        engines_available = engine_manager.are_engines_available()

    # Check DeepStream/GStreamer availability
    deepstream_available = False
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        deepstream_available = True
    except:
        pass

    return HealthResponse(
        status="healthy",
        version=__version__,
        engines_available=engines_available,
        gpu_available=gpu_available,
        deepstream_available=deepstream_available,
    )


@router.get("/api/v1/engines", response_model=EngineInfoResponse)
async def get_engine_info(request: Request) -> EngineInfoResponse:
    """Get information about loaded TensorRT engines."""
    engine_manager = getattr(request.app.state, "engine_manager", None)

    if engine_manager is None:
        return EngineInfoResponse(
            encoder={"exists": False, "error": "Engine manager not initialized"},
            decoder={"exists": False, "error": "Engine manager not initialized"},
        )

    info = engine_manager.get_engine_info()
    return EngineInfoResponse(
        encoder=info.get("encoder", {"exists": False}),
        decoder=info.get("decoder", {"exists": False}),
    )
