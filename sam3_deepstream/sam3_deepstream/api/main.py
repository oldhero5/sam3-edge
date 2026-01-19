"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_config, SAM3DeepStreamConfig
from ..export.engine_manager import EngineManager
from .routes import health, stream, video, nlq
from .services.job_manager import JobManager
from .services.detection_store import DetectionStore, set_detection_store
from .services.embedding_service import EmbeddingService, set_embedding_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    config = get_config()
    logger.info("Starting SAM3 DeepStream API...")

    # Initialize engine manager
    engine_manager = EngineManager(config=config)
    app.state.engine_manager = engine_manager
    app.state.config = config

    # Initialize job manager
    job_manager = JobManager(config=config)
    app.state.job_manager = job_manager

    # Start job processor
    job_manager.start()

    # Initialize detection store (SQLite + FAISS)
    detection_store = DetectionStore(
        db_path=config.database.db_path,
        index_path=config.database.faiss_index_path,
        embedding_dim=config.database.embedding_dim,
        use_gpu=config.database.use_gpu,
    )
    await detection_store.initialize()
    set_detection_store(detection_store)
    app.state.detection_store = detection_store
    logger.info(f"Detection store initialized: {config.database.db_path}")

    # Register this device for federation
    await detection_store.register_device(
        device_id=config.federation.device_id,
        hostname=config.federation.hostname,
    )

    # Initialize embedding service
    embedding_service = EmbeddingService(device="cuda")
    try:
        embedding_service.initialize()
        set_embedding_service(embedding_service)
        app.state.embedding_service = embedding_service
        logger.info("Embedding service initialized")
    except Exception as e:
        logger.warning(f"Embedding service initialization failed: {e}")

    # Try to load engines if available
    if engine_manager.are_engines_available():
        try:
            engine_manager.load_all()
            logger.info("TensorRT engines loaded successfully")

            # Initialize frame processor for WebSocket
            from ..inference.keyframe_processor import KeyframeProcessor
            from ..inference.mask_propagation import MaskPropagator

            processor = KeyframeProcessor(
                encoder_fn=engine_manager.encoder,
                decoder_fn=None,  # Decoder integration TBD
                keyframe_interval=config.inference.keyframe_interval,
                max_objects=config.inference.max_objects,
            )
            processor.set_propagator(MaskPropagator())
            app.state.frame_processor = processor

        except Exception as e:
            logger.warning(f"Failed to load TensorRT engines: {e}")
    else:
        logger.warning("TensorRT engines not found. Run export_engines.py first.")

    logger.info(f"API server ready at http://{config.api.host}:{config.api.port}")

    yield

    # Cleanup
    logger.info("Shutting down...")
    job_manager.stop()
    if hasattr(app.state, "detection_store"):
        app.state.detection_store.close()
    if hasattr(app.state, "engine_manager"):
        app.state.engine_manager.cleanup()


def create_app(config: Optional[SAM3DeepStreamConfig] = None) -> FastAPI:
    """Create FastAPI application."""
    if config:
        from ..config import set_config
        set_config(config)

    app = FastAPI(
        title="SAM3 DeepStream API",
        description="Real-time video segmentation using SAM3 on NVIDIA Jetson",
        version="0.1.0",
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

    # Include routers
    app.include_router(health.router)
    app.include_router(video.router)
    app.include_router(stream.router)
    app.include_router(nlq.router)

    return app


# Default application instance
app = create_app()


def run():
    """Run the API server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "sam3_deepstream.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
        workers=1,  # Single worker for GPU inference
    )


if __name__ == "__main__":
    run()
