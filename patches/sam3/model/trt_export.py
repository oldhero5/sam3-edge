# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
TensorRT export utilities for SAM3 on NVIDIA Jetson.

This module provides utilities for exporting SAM3 model components to TensorRT
for optimized inference on Jetson AGX Orin and other NVIDIA platforms.

Usage:
    from sam3.model.trt_export import TRTViTWrapper, export_vit_to_onnx, build_trt_engine

    # Wrap the ViT backbone for TRT-compatible export
    vit_wrapper = TRTViTWrapper(vit_model)

    # Export to ONNX
    export_vit_to_onnx(vit_wrapper, "vit_backbone.onnx")

    # Build TensorRT engine
    build_trt_engine("vit_backbone.onnx", "vit_backbone.engine", fp16=True)
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def _compute_rope_frequencies(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    """
    Pre-compute RoPE frequencies as real-valued cos/sin tensors for ONNX compatibility.

    Unlike the original complex-valued implementation, this returns separate
    cos and sin tensors that can be exported to ONNX/TensorRT.

    Args:
        dim: Dimension per head
        end_x: Spatial width
        end_y: Spatial height
        theta: RoPE theta parameter

    Returns:
        Tuple of (cos_freqs, sin_freqs) tensors, each of shape (end_x * end_y, dim // 2)
    """
    # Compute frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    # Create position grid
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()

    # Compute outer products for x and y
    freqs_x = torch.outer(t_x, freqs)
    freqs_y = torch.outer(t_y, freqs)

    # Stack x and y frequencies
    freqs = torch.cat([freqs_x, freqs_y], dim=-1)  # (end_x * end_y, dim // 2)

    # Return cos and sin separately (ONNX-compatible)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_enc_real(
    xq: Tensor,
    xk: Tensor,
    cos_freqs: Tensor,
    sin_freqs: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary position encoding using real-valued operations (ONNX-compatible).

    This replaces torch.view_as_complex/torch.polar which are not supported in ONNX.

    Args:
        xq: Query tensor of shape (B, nHeads, L, dim)
        xk: Key tensor of shape (B, nHeads, L, dim)
        cos_freqs: Cosine frequencies of shape (L, dim // 2)
        sin_freqs: Sine frequencies of shape (L, dim // 2)

    Returns:
        Rotated (xq, xk) tensors
    """
    # Split into pairs for rotation
    # xq shape: (B, nHeads, L, dim) -> (B, nHeads, L, dim // 2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_0, xq_1 = xq_r[..., 0], xq_r[..., 1]

    # Reshape frequencies for broadcasting
    # cos_freqs: (L, dim // 2) -> (1, 1, L, dim // 2)
    cos_f = cos_freqs.view(1, 1, cos_freqs.shape[0], -1)
    sin_f = sin_freqs.view(1, 1, sin_freqs.shape[0], -1)

    # Apply rotation: (x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x0*sin + x1*cos)
    xq_out_0 = xq_0 * cos_f - xq_1 * sin_f
    xq_out_1 = xq_0 * sin_f + xq_1 * cos_f

    # Interleave back
    xq_out = torch.stack([xq_out_0, xq_out_1], dim=-1).flatten(-2)

    if xk.shape[-2] == 0:
        return xq_out.type_as(xq), xk

    # Same for keys
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_0, xk_1 = xk_r[..., 0], xk_r[..., 1]

    xk_out_0 = xk_0 * cos_f - xk_1 * sin_f
    xk_out_1 = xk_0 * sin_f + xk_1 * cos_f

    xk_out = torch.stack([xk_out_0, xk_out_1], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def _get_rel_pos_bias(
    q_size: int,
    k_size: int,
    rel_pos: Tensor,
) -> Tensor:
    """
    Get relative position bias for attention (ONNX-compatible).

    Args:
        q_size: Query spatial size
        k_size: Key spatial size
        rel_pos: Relative position embeddings (L, C)

    Returns:
        Relative position bias of shape (q_size, k_size, C)
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    # Resize if needed
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Compute relative coordinates
    q_coords = torch.arange(q_size, device=rel_pos.device).float()
    k_coords = torch.arange(k_size, device=rel_pos.device).float()

    # Scale coordinates
    q_coords = q_coords * max(k_size / q_size, 1.0)
    k_coords = k_coords * max(q_size / k_size, 1.0)

    # Compute relative positions (q_size, k_size)
    relative_coords = q_coords.unsqueeze(1) - k_coords.unsqueeze(0)
    relative_coords = relative_coords + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


class TRTAttention(nn.Module):
    """
    TensorRT-compatible attention module.

    Replaces F.scaled_dot_product_attention with explicit matrix operations
    that are fully traceable for ONNX export. Supports both RoPE and
    relative position embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        use_rope: bool = False,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rope = use_rope
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        if use_rope and input_size is not None:
            # Pre-compute RoPE frequencies as buffers
            cos_freqs, sin_freqs = _compute_rope_frequencies(
                self.head_dim,
                input_size[0],
                input_size[1],
            )
            self.register_buffer("rope_cos", cos_freqs)
            self.register_buffer("rope_sin", sin_freqs)
        else:
            self.rope_cos = None
            self.rope_sin = None

        # Relative position embeddings (will be set by _copy_block_weights if needed)
        self.rel_pos_h: Optional[Tensor] = None
        self.rel_pos_w: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            B, H, W, _ = x.shape
            L = H * W
            x_flat = x.reshape(B, L, -1)
        else:
            B, L, _ = x.shape
            x_flat = x
            H = W = int(math.sqrt(L))

        # QKV projection
        qkv = self.qkv(x_flat).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, nHeads, L, dim)

        # Apply RoPE if enabled
        if self.use_rope and self.rope_cos is not None:
            q, k = apply_rotary_enc_real(q, k, self.rope_cos, self.rope_sin)

        # Explicit attention computation (ONNX-compatible)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative position bias if enabled
        if self.use_rel_pos and self.rel_pos_h is not None and self.rel_pos_w is not None:
            # Get rel pos bias for height and width
            rel_pos_h_bias = _get_rel_pos_bias(H, H, self.rel_pos_h)  # (H, H, head_dim)
            rel_pos_w_bias = _get_rel_pos_bias(W, W, self.rel_pos_w)  # (W, W, head_dim)

            # Reshape q for einsum: (B, nHeads, H*W, head_dim) -> (B, nHeads, H, W, head_dim)
            q_reshape = q.reshape(B, self.num_heads, H, W, self.head_dim)

            # Compute rel pos attention bias
            # rel_h: (B, nHeads, H, W, H) via einsum over head_dim
            rel_h = torch.einsum("bnhwc,hkc->bnhwk", q_reshape, rel_pos_h_bias)
            # rel_w: (B, nHeads, H, W, W) via einsum over head_dim
            rel_w = torch.einsum("bnhwc,wkc->bnhwk", q_reshape, rel_pos_w_bias)

            # Combine: attn shape is (B, nHeads, H*W, H*W)
            # Need to add spatial biases
            attn_weights = attn_weights.reshape(B, self.num_heads, H, W, H, W)
            attn_weights = attn_weights + rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2)
            attn_weights = attn_weights.reshape(B, self.num_heads, H * W, H * W)

        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape output
        if x.ndim == 4:
            out = out.permute(0, 2, 1, 3).reshape(B, H, W, -1)
        else:
            out = out.permute(0, 2, 1, 3).reshape(B, L, -1)

        return self.proj(out)


class TRTViTBlock(nn.Module):
    """TensorRT-compatible ViT block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TRTAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            use_rel_pos=use_rel_pos,
            input_size=(window_size, window_size) if window_size > 0 else input_size,
        )

        self.norm2 = nn.LayerNorm(dim)

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_fc1 = nn.Linear(dim, hidden_dim)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(hidden_dim, dim)

        self.window_size = window_size

    def _window_partition(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Partition into non-overlapping windows."""
        B, H, W, C = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
        return windows, (Hp, Wp)

    def _window_unpartition(
        self, windows: Tensor, pad_hw: Tuple[int, int], hw: Tuple[int, int]
    ) -> Tensor:
        """Reverse window partition."""
        Hp, Wp = pad_hw
        H, W = hw
        ws = self.window_size
        B = windows.shape[0] // (Hp * Wp // ws // ws)

        x = windows.reshape(B, Hp // ws, Wp // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)

        # Window attention if needed
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self._window_partition(x)

        x = self.attn(x)

        if self.window_size > 0:
            x = self._window_unpartition(x, pad_hw, (H, W))

        x = shortcut + x

        # MLP
        x = x + self.mlp_fc2(self.mlp_act(self.mlp_fc1(self.norm2(x))))

        return x


class TRTViTWrapper(nn.Module):
    """
    TensorRT-exportable wrapper for ViT backbone.

    This wrapper:
    1. Pre-computes RoPE frequencies as static buffers
    2. Replaces complex-valued operations with real equivalents
    3. Uses explicit attention instead of F.scaled_dot_product_attention
    4. Supports fixed input shape for optimal TRT engine building

    Args:
        vit_model: Original ViT model to wrap
        image_size: Input image size (default: 1008)
        patch_size: Patch size (default: 14)
    """

    def __init__(
        self,
        vit_model: nn.Module,
        image_size: int = 1008,
        patch_size: int = 14,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_size = image_size // patch_size  # 72 for 1008/14

        # Copy patch embedding
        self.patch_embed = vit_model.patch_embed

        # Copy position embedding if present
        if hasattr(vit_model, "pos_embed") and vit_model.pos_embed is not None:
            self.register_buffer("pos_embed", vit_model.pos_embed.clone())
        else:
            self.pos_embed = None

        # Copy layer norms
        self.ln_pre = vit_model.ln_pre
        self.ln_post = vit_model.ln_post

        # Build TRT-compatible blocks
        self.blocks = nn.ModuleList()
        self.full_attn_ids = vit_model.full_attn_ids

        for i, orig_block in enumerate(vit_model.blocks):
            # Determine if this is a windowed or global attention block
            window_size = orig_block.window_size

            # Check if block uses relative position embeddings
            use_rel_pos = getattr(orig_block.attn, 'use_rel_pos', False)

            block = TRTViTBlock(
                dim=orig_block.norm1.normalized_shape[0],
                num_heads=orig_block.attn.num_heads,
                mlp_ratio=orig_block.mlp.fc1.out_features
                / orig_block.norm1.normalized_shape[0],
                qkv_bias=orig_block.attn.qkv.bias is not None,
                use_rope=orig_block.attn.use_rope,
                use_rel_pos=use_rel_pos,
                window_size=window_size,
                input_size=(self.feature_size, self.feature_size),
            )

            # Copy weights from original block
            self._copy_block_weights(orig_block, block)
            self.blocks.append(block)

    def _copy_block_weights(self, orig_block: nn.Module, new_block: TRTViTBlock):
        """Copy weights from original block to TRT-compatible block."""
        # Copy norm weights
        new_block.norm1.load_state_dict(orig_block.norm1.state_dict())
        new_block.norm2.load_state_dict(orig_block.norm2.state_dict())

        # Copy attention weights
        new_block.attn.qkv.load_state_dict(orig_block.attn.qkv.state_dict())
        new_block.attn.proj.load_state_dict(orig_block.attn.proj.state_dict())

        # Copy relative position embeddings if present
        if hasattr(orig_block.attn, 'rel_pos_h') and orig_block.attn.rel_pos_h is not None:
            new_block.attn.rel_pos_h = orig_block.attn.rel_pos_h.clone()
        if hasattr(orig_block.attn, 'rel_pos_w') and orig_block.attn.rel_pos_w is not None:
            new_block.attn.rel_pos_w = orig_block.attn.rel_pos_w.clone()

        # Copy MLP weights
        new_block.mlp_fc1.load_state_dict(orig_block.mlp.fc1.state_dict())
        new_block.mlp_fc2.load_state_dict(orig_block.mlp.fc2.state_dict())

    def _get_abs_pos(self, hw: Tuple[int, int]) -> Tensor:
        """Get absolute positional embeddings (simplified for fixed size)."""
        if self.pos_embed is None:
            return None

        h, w = hw
        # Assumes pretrain_use_cls_token=True, retain_cls_token=False
        abs_pos = self.pos_embed[:, 1:]  # Remove cls token position

        xy_num = abs_pos.shape[1]
        size = int(math.sqrt(xy_num))

        if size != h or size != w:
            new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
            new_abs_pos = F.interpolate(
                new_abs_pos,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            return abs_pos.reshape(1, h, w, -1)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass with fixed input shape for TensorRT.

        Args:
            x: Input tensor of shape (B, 3, 1008, 1008)

        Returns:
            List containing single output tensor of shape (B, 1024, 72, 72)
        """
        # Patch embedding: (B, 3, 1008, 1008) -> (B, 72, 72, 1024)
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        # Add position embedding
        if self.pos_embed is not None:
            pos = self._get_abs_pos((h, w))
            if pos is not None:
                x = x + pos

        x = self.ln_pre(x)

        # Process through blocks
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            # Capture output at final global attention block
            if i == self.full_attn_ids[-1]:
                x = self.ln_post(x)
                # (B, H, W, C) -> (B, C, H, W)
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs


def export_vit_to_onnx(
    model: TRTViTWrapper,
    output_path: str,
    opset_version: int = 18,
    batch_size: int = 1,
    export_image_size: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """
    Export TRT-wrapped ViT model to ONNX format.

    Args:
        model: TRTViTWrapper model
        output_path: Path for ONNX file output
        opset_version: ONNX opset version (18+ recommended for transformers)
        batch_size: Fixed batch size for export
        export_image_size: Image size for export (smaller = less memory during tracing).
                          If None, uses 504 (36x36 patches) to avoid OOM.
                          TRT engine can still use different input sizes.
        verbose: Whether to print verbose export info
    """
    model.eval()

    # Use smaller image size for ONNX tracing to avoid OOM crashes
    # Full 1008x1008 (72x72 patches) creates ~40GB+ computation graph
    # 504x504 (36x36 patches) is much more manageable (~10GB)
    if export_image_size is None:
        export_image_size = 504  # Safe default for 32GB Jetson
        logger.info(f"Using reduced image size {export_image_size}x{export_image_size} for ONNX tracing (saves memory)")

    # Create dummy input with reduced shape
    dummy_input = torch.randn(
        batch_size, 3, export_image_size, export_image_size, device=next(model.parameters()).device
    )

    # Export to ONNX with fixed shapes
    # ViT uses pre-computed RoPE frequencies that depend on feature_size
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes=None,  # Fixed shapes - ViT has pre-computed positional embeddings
        do_constant_folding=True,
        verbose=verbose,
    )

    logger.info(f"Exported ViT backbone to ONNX: {output_path}")


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    workspace_size_gb: float = 4.0,
    dla_core: Optional[int] = None,
) -> bool:
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model file
        engine_path: Output path for TRT engine
        fp16: Enable FP16 precision (recommended for Jetson)
        int8: Enable INT8 precision (requires calibration)
        workspace_size_gb: GPU workspace size in GB
        dla_core: DLA core to use (0 or 1 on Jetson Orin, None for GPU only)

    Returns:
        True if successful, False otherwise
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not found. Install with: pip install tensorrt")
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX (use parse_from_file to handle external data files)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # parse_from_file handles external data files (*.onnx.data) automatically
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            logger.error(f"ONNX Parse Error: {parser.get_error(i)}")
        return False

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_size_gb * (1 << 30))
    )

    # Create optimization profile for any dynamic inputs
    # This is required even if we specified fixed shapes in ONNX export
    profile = builder.create_optimization_profile()
    has_dynamic = False

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name
        input_shape = input_tensor.shape

        # Check if any dimension is dynamic (-1)
        if -1 in input_shape:
            has_dynamic = True
            # Replace -1 with reasonable defaults based on input name
            fixed_shape = list(input_shape)
            for j, dim in enumerate(fixed_shape):
                if dim == -1:
                    # Use sensible defaults based on dimension position
                    if j == 0:
                        fixed_shape[j] = 1  # batch size
                    elif j == 1:
                        fixed_shape[j] = 3 if "image" in input_name.lower() else 256
                    else:
                        fixed_shape[j] = 504 if "image" in input_name.lower() else 72

            fixed_shape = tuple(fixed_shape)
            logger.info(f"Setting optimization profile for {input_name}: {fixed_shape}")
            profile.set_shape(input_name, fixed_shape, fixed_shape, fixed_shape)

    if has_dynamic:
        config.add_optimization_profile(profile)
        logger.info("Added optimization profile for dynamic inputs")

    # Precision settings
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 precision")

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("Enabled INT8 precision (requires calibration)")

    # DLA settings for Jetson
    if dla_core is not None:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        logger.info(f"Configured for DLA core {dla_core} with GPU fallback")

    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return False

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved to: {engine_path}")
    return True


class TRTInferenceEngine:
    """
    TensorRT inference engine wrapper for SAM3.

    Provides a simple interface for running inference with TRT engines.
    """

    def __init__(self, engine_path: str, device: int = 0):
        """
        Load TensorRT engine.

        Args:
            engine_path: Path to serialized TRT engine
            device: CUDA device index
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not found. Install with: pip install tensorrt")

        self.device = device
        torch.cuda.set_device(device)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs: Dict[str, torch.Tensor] = {}
        self.outputs: Dict[str, torch.Tensor] = {}
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate input/output buffers based on engine bindings."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            # Convert TRT dtype to torch dtype
            if dtype == 1:  # trt.float32
                torch_dtype = torch.float32
            elif dtype == 2:  # trt.float16
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            tensor = torch.empty(
                tuple(shape), dtype=torch_dtype, device=f"cuda:{self.device}"
            )

            if self.engine.get_tensor_mode(name) == 0:  # Input
                self.inputs[name] = tensor
            else:  # Output
                self.outputs[name] = tensor

            self.context.set_tensor_address(name, tensor.data_ptr())

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run inference.

        Args:
            image: Input tensor of shape (B, 3, H, W)

        Returns:
            Output features tensor
        """
        # Copy input
        self.inputs["image"].copy_(image)

        # Run inference
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return self.outputs["features"].clone()


def convert_sam3_vit_to_trt(
    sam3_model: nn.Module,
    output_dir: str,
    fp16: bool = True,
    dla_core: Optional[int] = None,
) -> str:
    """
    Convert SAM3 ViT backbone to TensorRT.

    This is a high-level convenience function that:
    1. Wraps the ViT backbone for TRT compatibility
    2. Exports to ONNX
    3. Builds TRT engine

    Args:
        sam3_model: SAM3 model (Sam3Image or Sam3VideoInference)
        output_dir: Directory for output files
        fp16: Enable FP16 precision
        dla_core: DLA core to use (None for GPU only)

    Returns:
        Path to the TRT engine file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract ViT backbone
    # SAM3 model structure: detector.backbone.vision_backbone.trunk
    vit = None
    if hasattr(sam3_model, "backbone"):
        backbone = sam3_model.backbone
        # SAM3VLBackbone has vision_backbone.trunk
        if hasattr(backbone, "vision_backbone"):
            if hasattr(backbone.vision_backbone, "trunk"):
                vit = backbone.vision_backbone.trunk
        # Alternative: direct visual.trunk path
        elif hasattr(backbone, "visual"):
            vit = backbone.visual.trunk
        # Fallback: backbone itself might be the ViT
        elif hasattr(backbone, "patch_embed"):
            vit = backbone

    if vit is None:
        raise ValueError(
            "Could not find ViT backbone in model. "
            f"Model type: {type(sam3_model)}, "
            f"backbone type: {type(getattr(sam3_model, 'backbone', None))}"
        )

    # Wrap for TRT export with reduced image size to avoid OOM during tracing
    # 504x504 (36x36 patches) is much more memory-efficient than 1008x1008 (72x72 patches)
    # The tracing computation graph scales quadratically with patch count
    export_image_size = 504
    logger.info(f"Wrapping ViT backbone for TensorRT export (image_size={export_image_size})...")
    wrapped_vit = TRTViTWrapper(vit, image_size=export_image_size)

    # Use CPU for ONNX tracing to avoid GPU memory exhaustion on Jetson
    # TensorRT engine building will still use GPU
    logger.info("Using CPU for ONNX export (avoids GPU memory issues)...")
    wrapped_vit = wrapped_vit.cpu().eval()

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "sam3_vit.onnx")
    logger.info(f"Exporting to ONNX: {onnx_path}")
    export_vit_to_onnx(wrapped_vit, onnx_path, export_image_size=export_image_size)

    # Build TRT engine
    engine_path = os.path.join(output_dir, "sam3_vit.engine")
    logger.info(f"Building TensorRT engine: {engine_path}")
    success = build_trt_engine(
        onnx_path,
        engine_path,
        fp16=fp16,
        dla_core=dla_core,
    )

    if not success:
        raise RuntimeError("Failed to build TensorRT engine")

    return engine_path
