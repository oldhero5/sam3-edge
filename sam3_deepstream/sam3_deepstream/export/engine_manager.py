"""TensorRT engine management - loading, caching, and lifecycle."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from sam3.model.trt_export import TRTInferenceEngine

from ..config import get_config, SAM3DeepStreamConfig

logger = logging.getLogger(__name__)


class EngineManager:
    """
    Manages TensorRT engine lifecycle for SAM3 DeepStream.

    Handles loading, caching, and provides unified access to
    encoder and decoder engines.
    """

    def __init__(
        self,
        config: Optional[SAM3DeepStreamConfig] = None,
        device: int = 0,
    ):
        """
        Initialize engine manager.

        Args:
            config: Configuration object. Uses global config if None.
            device: CUDA device index.
        """
        self.config = config or get_config()
        self.device = device

        self._encoder_engine: Optional[TRTInferenceEngine] = None
        self._decoder_engine: Optional[TRTInferenceEngine] = None

    @property
    def encoder(self) -> TRTInferenceEngine:
        """Get or load encoder engine."""
        if self._encoder_engine is None:
            self._encoder_engine = self._load_engine("encoder")
        return self._encoder_engine

    @property
    def decoder(self) -> TRTInferenceEngine:
        """Get or load decoder engine."""
        if self._decoder_engine is None:
            self._decoder_engine = self._load_engine("decoder")
        return self._decoder_engine

    def _load_engine(self, name: str) -> TRTInferenceEngine:
        """Load a TensorRT engine by name."""
        if name == "encoder":
            engine_path = self.config.encoder_engine
            if engine_path is None:
                engine_path = self.config.trt.cache_dir / "sam3_encoder.engine"
        elif name == "decoder":
            engine_path = self.config.decoder_engine
            if engine_path is None:
                engine_path = self.config.trt.cache_dir / "sam3_decoder.engine"
        else:
            raise ValueError(f"Unknown engine name: {name}")

        if not engine_path.exists():
            raise FileNotFoundError(
                f"{name.title()} engine not found at {engine_path}. "
                f"Run export_engines.py to create TensorRT engines."
            )

        logger.info(f"Loading {name} engine from: {engine_path}")
        return TRTInferenceEngine(str(engine_path), device=self.device)

    def load_all(self) -> None:
        """Pre-load all engines."""
        _ = self.encoder
        _ = self.decoder
        logger.info("All TensorRT engines loaded")

    def is_encoder_available(self) -> bool:
        """Check if encoder engine exists."""
        engine_path = self.config.encoder_engine
        if engine_path is None:
            engine_path = self.config.trt.cache_dir / "sam3_encoder.engine"
        return engine_path.exists()

    def is_decoder_available(self) -> bool:
        """Check if decoder engine exists."""
        engine_path = self.config.decoder_engine
        if engine_path is None:
            engine_path = self.config.trt.cache_dir / "sam3_decoder.engine"
        return engine_path.exists()

    def are_engines_available(self) -> bool:
        """Check if both engines are available."""
        return self.is_encoder_available() and self.is_decoder_available()

    def get_engine_info(self) -> Dict[str, dict]:
        """Get information about available engines."""
        info = {}

        for name in ["encoder", "decoder"]:
            if name == "encoder":
                path = self.config.encoder_engine or (
                    self.config.trt.cache_dir / "sam3_encoder.engine"
                )
            else:
                path = self.config.decoder_engine or (
                    self.config.trt.cache_dir / "sam3_decoder.engine"
                )

            if path.exists():
                stat = path.stat()
                info[name] = {
                    "path": str(path),
                    "exists": True,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": stat.st_mtime,
                }
            else:
                info[name] = {
                    "path": str(path),
                    "exists": False,
                }

        return info

    def cleanup(self) -> None:
        """Release engine resources."""
        self._encoder_engine = None
        self._decoder_engine = None
        torch.cuda.empty_cache()
        logger.info("Engine resources released")


# Global engine manager instance
_engine_manager: Optional[EngineManager] = None


def get_engine_manager() -> EngineManager:
    """Get or create global engine manager."""
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager()
    return _engine_manager


def set_engine_manager(manager: EngineManager) -> None:
    """Set the global engine manager."""
    global _engine_manager
    _engine_manager = manager
