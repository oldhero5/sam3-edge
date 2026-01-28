"""
TensorRT export for Perception Encoder vision backbone.

This module provides utilities for exporting the PE-enhanced SAM3
vision encoder to TensorRT engines for optimized inference on
NVIDIA Jetson AGX Orin.

Based on PE paper (arXiv:2504.13181) and SAM3 paper (arXiv:2511.16719).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PEEncoderWrapper(nn.Module):
    """
    Wrapper for PE encoder that simplifies ONNX export.

    The PE encoder with alignment tuning has multiple outputs
    (fused features + intermediate features). This wrapper
    packages them for clean ONNX export.
    """

    def __init__(self, pe_encoder, export_intermediates: bool = True):
        super().__init__()
        self.pe_encoder = pe_encoder
        self.export_intermediates = export_intermediates

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass returning fused features and optionally intermediates.

        Args:
            x: (B, 3, H, W) input images

        Returns:
            Tuple of (fused_features, [intermediate_features...])
        """
        if self.export_intermediates:
            features, intermediates = self.pe_encoder.forward_features(x)
            # Return as tuple for ONNX export
            return (features,) + tuple(intermediates)
        else:
            features = self.pe_encoder(x)
            return (features,)


def export_pe_encoder_to_onnx(
    model,
    output_path: str,
    input_size: int = 1008,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    export_intermediates: bool = True,
    simplify: bool = True,
) -> str:
    """
    Export PE vision encoder to ONNX format.

    Args:
        model: SAM3 model with PE backbone
        output_path: Path to save ONNX model
        input_size: Input image size (1008 for SAM3)
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        export_intermediates: Export intermediate layer features
        simplify: Simplify ONNX model with onnxsim

    Returns:
        Path to exported ONNX model
    """
    # Extract PE encoder from model
    try:
        pe_encoder = model.backbone.visual.trunk.pe_encoder
    except AttributeError:
        # Try alternative path
        pe_encoder = model.backbone.visual.trunk

    pe_encoder.eval()

    # Create export wrapper
    wrapper = PEEncoderWrapper(pe_encoder, export_intermediates)
    wrapper.eval()

    # Dummy input
    device = next(pe_encoder.parameters()).device
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # Determine output names
    output_names = ["fused_features"]
    if export_intermediates and hasattr(pe_encoder, "intermediate_layers"):
        num_intermediate = len(pe_encoder.intermediate_layers)
        output_names.extend([f"intermediate_{i}" for i in range(num_intermediate)])

    # Dynamic axes
    dynamic_axes = {"input": {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    # Export to ONNX
    logger.info(f"Exporting PE encoder to ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_batch else None,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    # Optionally simplify
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            logger.info("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx_model_simplified, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(onnx_model_simplified, output_path)
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification check failed, using original")
        except ImportError:
            logger.warning("onnxsim not available, skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")

    logger.info(f"Exported PE encoder to: {output_path}")
    return output_path


def build_pe_encoder_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    workspace_mb: int = 4096,
    dla_core: Optional[int] = None,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
    input_size: int = 1008,
    verbose: bool = False,
) -> str:
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ("fp32", "fp16", "int8")
        workspace_mb: Workspace size in MB
        dla_core: DLA core to use (None for GPU only)
        min_batch: Minimum batch size for optimization
        opt_batch: Optimal batch size for optimization
        max_batch: Maximum batch size for optimization
        input_size: Input image size
        verbose: Enable verbose logging

    Returns:
        Path to TensorRT engine
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT not available. Please install tensorrt package."
        )

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    # Create network with explicit batch
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    logger.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX Parser error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )

    # Precision settings
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")
        else:
            logger.warning("FP16 not supported on this platform")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("Enabled INT8 precision (requires calibration)")
            # Note: INT8 requires a calibrator for accuracy
        else:
            logger.warning("INT8 not supported on this platform")

    # DLA configuration (for Jetson)
    if dla_core is not None:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        logger.info(f"Enabled DLA core {dla_core} with GPU fallback")

    # Optimization profiles for dynamic batch
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(min_batch, 3, input_size, input_size),
        opt=(opt_batch, 3, input_size, input_size),
        max=(max_batch, 3, input_size, input_size),
    )
    config.add_optimization_profile(profile)
    logger.info(f"Set optimization profile: batch [{min_batch}, {opt_batch}, {max_batch}]")

    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    logger.info(f"Built TensorRT engine: {engine_path}")
    return engine_path


class PEEncoderTRTRuntime:
    """
    TensorRT runtime for PE encoder inference.

    Provides efficient inference using TensorRT engine with
    CUDA stream management and memory optimization.
    """

    def __init__(
        self,
        engine_path: str,
        device: int = 0,
    ):
        """
        Initialize TRT runtime.

        Args:
            engine_path: Path to TensorRT engine file
            device: CUDA device ID
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError as e:
            raise ImportError(f"Required package not available: {e}")

        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate input/output buffers."""
        import pycuda.driver as cuda

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            # Convert TRT dtype to numpy dtype
            if dtype == trt.float32:
                np_dtype = np.float32
            elif dtype == trt.float16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32

            # Calculate size
            size = trt.volume(shape)
            if size < 0:
                # Dynamic shape - use max batch
                shape = list(shape)
                shape[0] = 4  # max batch
                size = abs(trt.volume(shape))

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"name": name, "host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"name": name, "host": host_mem, "device": device_mem})

    def infer(self, input_tensor: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """
        Run inference on input tensor.

        Args:
            input_tensor: (B, 3, H, W) input images

        Returns:
            Dict mapping output names to tensors
        """
        import pycuda.driver as cuda
        import numpy as np

        # Copy input to host memory
        input_np = input_tensor.cpu().numpy().ravel()
        np.copyto(self.inputs[0]["host"], input_np)

        # Copy to device
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream,
        )

        # Set input shape for dynamic batch
        batch_size = input_tensor.shape[0]
        self.context.set_input_shape(
            self.inputs[0]["name"],
            input_tensor.shape,
        )

        # Run inference
        self.context.execute_async_v3(self.stream.handle)

        # Copy outputs back
        results = {}
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)

        self.stream.synchronize()

        # Convert to tensors
        for output in self.outputs:
            # Get actual output shape
            shape = self.context.get_tensor_shape(output["name"])
            tensor = torch.from_numpy(
                output["host"][: np.prod(shape)].reshape(shape)
            )
            results[output["name"]] = tensor.to(input_tensor.device)

        return results


def export_pe_model_engines(
    checkpoint_path: str,
    output_dir: str,
    precision: str = "fp16",
    dla_core: Optional[int] = None,
    export_text_encoder: bool = True,
) -> Dict[str, str]:
    """
    Export complete PE-enhanced SAM3 model to TensorRT engines.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save engines
        precision: Precision mode ("fp32", "fp16", "int8")
        dla_core: DLA core to use (None for GPU only)
        export_text_encoder: Also export text encoder to ONNX

    Returns:
        Dict mapping component names to engine paths
    """
    from sam3.model_builder import build_sam3_pe_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engines = {}

    # Build model
    logger.info("Building SAM3 PE model...")
    model = build_sam3_pe_model(
        checkpoint_path=checkpoint_path,
        device="cuda",
        eval_mode=True,
        use_alignment_tuning=True,
    )

    # Export PE encoder
    onnx_path = output_dir / "pe_encoder.onnx"
    engine_path = output_dir / f"pe_encoder_{precision}.engine"

    export_pe_encoder_to_onnx(model, str(onnx_path))
    build_pe_encoder_engine(
        str(onnx_path),
        str(engine_path),
        precision=precision,
        dla_core=dla_core,
    )
    engines["pe_encoder"] = str(engine_path)

    # Export text encoder to ONNX (TRT optional for text)
    if export_text_encoder:
        text_onnx_path = output_dir / "pe_text_encoder.onnx"
        _export_text_encoder_to_onnx(model, str(text_onnx_path))
        engines["pe_text_encoder_onnx"] = str(text_onnx_path)

    logger.info(f"Exported engines: {engines}")
    return engines


def _export_text_encoder_to_onnx(
    model,
    output_path: str,
    opset_version: int = 17,
    max_seq_length: int = 77,
) -> str:
    """
    Export PE text encoder to ONNX.

    Args:
        model: SAM3 model with PE backbone
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        max_seq_length: Maximum token sequence length

    Returns:
        Path to exported ONNX model
    """
    # Get text encoder from model
    try:
        text_encoder = model.backbone.text
    except AttributeError:
        try:
            text_encoder = model.text_encoder
        except AttributeError:
            logger.warning("Could not find text encoder in model, skipping export")
            return output_path

    text_encoder.eval()

    # Wrapper for ONNX-compatible export
    class TextEncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for text encoding.

            Args:
                tokens: (B, L) input token IDs

            Returns:
                (B, D) text embeddings
            """
            # Use the encoder's encode_text if available
            if hasattr(self.encoder, 'encode_text'):
                return self.encoder.encode_text(tokens)

            # Otherwise, manual forward pass
            x = self.encoder.token_embedding(tokens)
            B, L = tokens.shape
            x = x + self.encoder.positional_embedding[:L]

            # Apply transformer
            if hasattr(self.encoder, 'transformer'):
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.encoder.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD

            # Project to embedding space
            if hasattr(self.encoder, 'text_projection'):
                # Take features from EOS token
                x = x[torch.arange(B), tokens.argmax(dim=-1)]
                x = x @ self.encoder.text_projection

            return x

    wrapper = TextEncoderWrapper(text_encoder)
    wrapper.eval()

    # Get device
    device = next(text_encoder.parameters()).device

    # Dummy input - tokenized text
    dummy_tokens = torch.randint(0, 49408, (1, max_seq_length), device=device, dtype=torch.long)

    # Dynamic axes for variable batch and sequence length
    dynamic_axes = {
        "tokens": {0: "batch", 1: "seq_length"},
        "text_embedding": {0: "batch"},
    }

    # Export to ONNX
    logger.info(f"Exporting PE text encoder to ONNX: {output_path}")
    try:
        torch.onnx.export(
            wrapper,
            dummy_tokens,
            output_path,
            input_names=["tokens"],
            output_names=["text_embedding"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )
        logger.info(f"Text encoder ONNX export saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export text encoder: {e}")
        # Create empty placeholder file so we know export was attempted
        Path(output_path).touch()

    return output_path


# Convenience function for external use
def export_pe_text_encoder(
    model,
    output_path: str,
    opset_version: int = 17,
    max_seq_length: int = 77,
) -> str:
    """
    Export PE text encoder to ONNX format.

    This is a public wrapper around _export_text_encoder_to_onnx.

    Args:
        model: SAM3 model with PE backbone
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        max_seq_length: Maximum token sequence length

    Returns:
        Path to exported ONNX model
    """
    return _export_text_encoder_to_onnx(model, output_path, opset_version, max_seq_length)


# Also expose export_pe_encoder as a convenience alias
def export_pe_encoder(
    model,
    output_path: str,
    input_size: int = 1008,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    export_intermediates: bool = True,
    simplify: bool = True,
) -> str:
    """
    Convenience alias for export_pe_encoder_to_onnx.

    Args:
        model: SAM3 model with PE backbone
        output_path: Path to save ONNX model
        input_size: Input image size (1008 for SAM3)
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        export_intermediates: Export intermediate layer features
        simplify: Simplify ONNX model with onnxsim

    Returns:
        Path to exported ONNX model
    """
    return export_pe_encoder_to_onnx(
        model,
        output_path,
        input_size=input_size,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        export_intermediates=export_intermediates,
        simplify=simplify,
    )
