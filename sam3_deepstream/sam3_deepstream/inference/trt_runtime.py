"""TensorRT runtime wrapper for unified encoder/decoder inference."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import get_config

logger = logging.getLogger(__name__)


class SAM3TRTRuntime:
    """
    Unified TensorRT runtime for SAM3 encoder and decoder.

    Provides a high-level interface for running SAM3 inference
    using TensorRT engines.
    """

    def __init__(
        self,
        encoder_engine_path: Union[str, Path],
        decoder_engine_path: Optional[Union[str, Path]] = None,
        device: int = 0,
    ):
        """
        Initialize TensorRT runtime.

        Args:
            encoder_engine_path: Path to encoder TRT engine
            decoder_engine_path: Path to decoder TRT engine (optional)
            device: CUDA device index
        """
        self.device = device
        torch.cuda.set_device(device)

        # Load encoder
        logger.info(f"Loading encoder engine: {encoder_engine_path}")
        self.encoder = TRTInferenceEngine(str(encoder_engine_path), device=device)

        # Load decoder if provided
        self.decoder: Optional[TRTInferenceEngine] = None
        if decoder_engine_path:
            logger.info(f"Loading decoder engine: {decoder_engine_path}")
            self.decoder = TRTInferenceEngine(str(decoder_engine_path), device=device)

        # Cache for image embeddings
        self._cached_embeddings: Optional[Tensor] = None
        self._cached_image_size: Optional[Tuple[int, int]] = None

    def encode_image(
        self,
        image: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        """
        Encode image using TensorRT encoder.

        Args:
            image: Input image tensor (B, 3, H, W) in [0, 255] or [0, 1]
            normalize: Whether to normalize the image

        Returns:
            Image embeddings tensor
        """
        # Ensure correct shape
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Normalize if needed
        if normalize and image.max() > 1.0:
            image = image / 255.0

        # Resize to model input size
        config = get_config()
        target_size = config.inference.input_size

        if image.shape[2] != target_size or image.shape[3] != target_size:
            image = F.interpolate(
                image,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        # Move to device
        image = image.to(f"cuda:{self.device}")

        # Run encoder
        embeddings = self.encoder(image)

        # Cache for potential decoder use
        self._cached_embeddings = embeddings
        self._cached_image_size = (image.shape[2], image.shape[3])

        return embeddings

    def decode_masks(
        self,
        image_embeddings: Tensor,
        sparse_prompts: Tensor,
        dense_prompts: Optional[Tensor] = None,
        image_pe: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Decode masks from embeddings and prompts.

        Args:
            image_embeddings: Encoder output
            sparse_prompts: Point/box prompt embeddings
            dense_prompts: Mask prompt embeddings
            image_pe: Positional encodings

        Returns:
            Tuple of (masks, iou_predictions)
        """
        if self.decoder is None:
            raise RuntimeError("Decoder engine not loaded")

        B, C, H, W = image_embeddings.shape

        # Create default dense prompts if not provided
        if dense_prompts is None:
            dense_prompts = torch.zeros(B, C, H, W, device=image_embeddings.device)

        # Create default positional encoding if not provided
        if image_pe is None:
            image_pe = self._get_position_encoding(H, W, C)
            image_pe = image_pe.to(image_embeddings.device)

        # Run decoder - this would need a custom inference method
        # since TRTInferenceEngine expects single input
        # For now, return placeholder
        raise NotImplementedError(
            "Decoder inference requires custom TRT engine with multiple inputs"
        )

    def _get_position_encoding(self, h: int, w: int, dim: int) -> Tensor:
        """Generate sinusoidal position encoding."""
        y_embed = torch.arange(h, dtype=torch.float32).view(1, h, 1, 1)
        x_embed = torch.arange(w, dtype=torch.float32).view(1, 1, w, 1)

        y_embed = y_embed / h * 2 * 3.14159
        x_embed = x_embed / w * 2 * 3.14159

        dim_t = torch.arange(dim // 4, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t

        pos = torch.cat([
            pos_x.sin(), pos_x.cos(),
            pos_y.sin(), pos_y.cos(),
        ], dim=-1)

        return pos.permute(0, 3, 1, 2)

    @property
    def cached_embeddings(self) -> Optional[Tensor]:
        """Get cached image embeddings."""
        return self._cached_embeddings

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cached_embeddings = None
        self._cached_image_size = None
        torch.cuda.empty_cache()

    def create_async_runtime(self, num_streams: int = 3):
        """
        Create async version of this runtime for higher throughput.

        Returns:
            AsyncSAM3Runtime with same encoder/decoder engines
        """
        from .async_trt_runtime import AsyncSAM3Runtime

        # Get engine paths from current engine objects
        # Note: This requires storing paths, which we'll do via a factory
        raise NotImplementedError(
            "Use AsyncSAM3Runtime directly with engine paths for async inference"
        )


class TRTInferenceEngine:
    """
    Extended TensorRT inference engine for multi-input models.

    This extends the base SAM3 TRTInferenceEngine to support
    models with multiple input tensors (like the decoder).
    """

    def __init__(self, engine_path: str, device: int = 0):
        """Load TensorRT engine."""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not found")

        self.device = device
        torch.cuda.set_device(device)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get IO tensor info
        self.inputs: Dict[str, dict] = {}
        self.outputs: Dict[str, dict] = {}
        self._buffers: Dict[str, Tensor] = {}

        self._setup_io()

    def _setup_io(self) -> None:
        """Set up input/output tensor information."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            # Convert TRT dtype to torch
            if dtype.name == "FLOAT":
                torch_dtype = torch.float32
            elif dtype.name == "HALF":
                torch_dtype = torch.float16
            elif dtype.name == "INT32":
                torch_dtype = torch.int32
            else:
                torch_dtype = torch.float32

            info = {
                "shape": tuple(shape),
                "dtype": torch_dtype,
                "trt_dtype": dtype,
            }

            mode = self.engine.get_tensor_mode(name)
            if mode.name == "INPUT":
                self.inputs[name] = info
            else:
                self.outputs[name] = info

    def allocate_buffers(self, **input_shapes) -> None:
        """
        Allocate buffers with specific shapes.

        Args:
            **input_shapes: Mapping of input name to shape tuple
        """
        device_str = f"cuda:{self.device}"

        # Allocate inputs
        for name, info in self.inputs.items():
            shape = input_shapes.get(name, info["shape"])
            self._buffers[name] = torch.empty(
                shape, dtype=info["dtype"], device=device_str
            )
            self.context.set_tensor_address(name, self._buffers[name].data_ptr())

        # Allocate outputs (may depend on input shapes)
        for name, info in self.outputs.items():
            shape = self.context.get_tensor_shape(name)
            self._buffers[name] = torch.empty(
                tuple(shape), dtype=info["dtype"], device=device_str
            )
            self.context.set_tensor_address(name, self._buffers[name].data_ptr())

    def __call__(self, image: Tensor) -> Tensor:
        """
        Run inference with single input tensor (for encoder-style models).

        Args:
            image: Input image tensor

        Returns:
            Output features tensor
        """
        # Ensure buffers are allocated
        if not self._buffers:
            self.allocate_buffers()

        # Get the input tensor name (assume single input)
        input_name = list(self.inputs.keys())[0]
        output_name = list(self.outputs.keys())[0]

        # Ensure buffer matches input shape
        if self._buffers[input_name].shape != image.shape:
            self.allocate_buffers(**{input_name: image.shape})

        # Copy input
        self._buffers[input_name].copy_(image)

        # Run inference
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return self._buffers[output_name].clone()

    def infer(self, **inputs) -> Dict[str, Tensor]:
        """
        Run inference with named inputs.

        Args:
            **inputs: Mapping of input name to tensor

        Returns:
            Dictionary of output name to tensor
        """
        # Copy inputs
        for name, tensor in inputs.items():
            if name in self._buffers:
                self._buffers[name].copy_(tensor)

        # Run inference
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # Return outputs
        return {
            name: self._buffers[name].clone()
            for name in self.outputs
        }
