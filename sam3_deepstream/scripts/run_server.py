#!/usr/bin/env python3
"""Launch SAM3 DeepStream FastAPI server."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Launch SAM3 DeepStream API server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=None,
        help="Directory containing TensorRT engines",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=5,
        help="Frames between full inference",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    # Configure from args
    import os
    if args.engine_dir:
        os.environ["SAM3_ENGINE_DIR"] = str(args.engine_dir)
    if args.keyframe_interval:
        os.environ["SAM3_KEYFRAME_INTERVAL"] = str(args.keyframe_interval)

    # Import and configure
    from sam3_deepstream.config import get_config, set_config

    config = get_config()
    config.api.host = args.host
    config.api.port = args.port
    set_config(config)

    # Check engines
    from sam3_deepstream.export.engine_manager import EngineManager
    engine_manager = EngineManager(config)

    if engine_manager.are_engines_available():
        logger.info("TensorRT engines found")
    else:
        logger.warning(
            "TensorRT engines not found. "
            "Run 'python scripts/export_engines.py' to create them."
        )

    # Start server
    import uvicorn

    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info("API documentation available at /docs")

    uvicorn.run(
        "sam3_deepstream.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
