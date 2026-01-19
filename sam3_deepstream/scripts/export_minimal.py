#!/usr/bin/env python3
"""
Minimal TRT export script that avoids importing the full sam3 package.
This script directly loads checkpoint weights and exports to TensorRT.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from sam3.model.trt_export import (
    TRTViTWrapper,
    export_vit_to_onnx,
    build_trt_engine,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_vit_from_checkpoint(checkpoint_path: str) -> nn.Module:
    """
    Load ViT backbone directly from checkpoint without full model init.

    This avoids importing the full sam3 package with all its dependencies.
    Infers all model parameters directly from checkpoint tensor shapes.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Find ViT-related keys to understand structure
    vit_keys = [k for k in state_dict.keys() if 'backbone' in k or 'trunk' in k or 'patch_embed' in k]
    logger.info(f"Found {len(vit_keys)} ViT-related keys")

    # Determine the prefix for ViT weights
    prefix = None
    for key in state_dict.keys():
        if 'patch_embed' in key:
            prefix = key.split('patch_embed')[0]
            break

    if prefix is None:
        raise ValueError("Could not find patch_embed in checkpoint to determine ViT prefix")

    logger.info(f"ViT weight prefix: '{prefix}'")

    # Extract ViT weights with prefix removed
    vit_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            vit_state_dict[new_key] = value

    logger.info(f"Extracted {len(vit_state_dict)} ViT weights")

    # Infer model structure from weight shapes
    # Get embed_dim from patch_embed
    if 'patch_embed.proj.weight' in vit_state_dict:
        patch_embed_weight = vit_state_dict['patch_embed.proj.weight']
        embed_dim = patch_embed_weight.shape[0]
        patch_size = patch_embed_weight.shape[2]  # Kernel size
    else:
        raise ValueError("Could not determine embed_dim from checkpoint")

    # Count blocks
    block_ids = set()
    for key in vit_state_dict.keys():
        if key.startswith('blocks.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    num_blocks = max(block_ids) + 1 if block_ids else 0

    # Infer MLP ratio from fc1 weight shape
    if 'blocks.0.mlp.fc1.weight' in vit_state_dict:
        mlp_hidden = vit_state_dict['blocks.0.mlp.fc1.weight'].shape[0]
        mlp_ratio = mlp_hidden / embed_dim
    else:
        mlp_ratio = 4.0

    # Infer num_heads from rel_pos or qkv shape
    # rel_pos_h shape is (2*win-1, head_dim), head_dim = embed_dim / num_heads
    num_heads = 16  # Default
    if 'blocks.0.attn.rel_pos_h' in vit_state_dict:
        rel_pos_shape = vit_state_dict['blocks.0.attn.rel_pos_h'].shape
        head_dim = rel_pos_shape[1]
        num_heads = embed_dim // head_dim

    # Infer image size from pos_embed shape
    if 'pos_embed' in vit_state_dict:
        pos_embed_shape = vit_state_dict['pos_embed'].shape
        num_patches = pos_embed_shape[1] - 1  # Subtract cls token
        grid_size = int(num_patches ** 0.5)
        image_size = grid_size * patch_size
    else:
        image_size = 1008  # Default

    # Check for features
    has_rel_pos = any('rel_pos' in k for k in vit_state_dict.keys())
    has_rope = any('rope' in k.lower() for k in vit_state_dict.keys())

    # Determine window sizes - check for window_size attribute or infer from structure
    # SAM2/3 typically uses window_size=14 for local attention blocks
    window_sizes = []
    for i in range(num_blocks):
        # Default: every 4th block is global attention
        if (i + 1) % 4 == 0 or i == num_blocks - 1:
            window_sizes.append(0)  # Global attention
        else:
            window_sizes.append(14)  # Local windowed attention

    # Full attention block IDs
    full_attn_ids = [i for i in range(num_blocks) if window_sizes[i] == 0]

    logger.info(f"Inferred config: embed_dim={embed_dim}, num_blocks={num_blocks}, "
                f"num_heads={num_heads}, mlp_ratio={mlp_ratio:.2f}, "
                f"patch_size={patch_size}, image_size={image_size}")
    logger.info(f"Features: rel_pos={has_rel_pos}, rope={has_rope}")
    logger.info(f"Full attention block IDs: {full_attn_ids}")

    # Build ViT with inferred parameters
    vit = MinimalViT(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_rel_pos=has_rel_pos,
        use_rope=has_rope,
        window_sizes=window_sizes,
        full_attn_ids=full_attn_ids,
        image_size=image_size,
        patch_size=patch_size,
        vit_state_dict=vit_state_dict,  # Pass for dynamic initialization
    )

    # Load weights (should now match)
    missing, unexpected = vit.load_state_dict(vit_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing {len(missing)} keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys: {unexpected[:5]}...")

    return vit


class MinimalViT(nn.Module):
    """Minimal ViT implementation for weight loading - TRTViTWrapper will do the actual work."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_rel_pos: bool = True,
        use_rope: bool = False,
        window_sizes: list = None,
        full_attn_ids: list = None,
        image_size: int = 1008,
        patch_size: int = 14,
        vit_state_dict: dict = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size

        # Patch embedding - use PatchEmbed class to match state dict keys
        self.patch_embed = PatchEmbed(embed_dim, patch_size)

        # Position embedding - infer shape from state dict if available
        if vit_state_dict and 'pos_embed' in vit_state_dict:
            pos_shape = vit_state_dict['pos_embed'].shape
            self.pos_embed = nn.Parameter(torch.zeros(pos_shape))
        else:
            num_patches = (image_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Layer norms
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

        # Blocks
        self.blocks = nn.ModuleList()

        if window_sizes is None:
            window_sizes = [14 if i % 2 == 0 else 0 for i in range(num_blocks)]

        if full_attn_ids is None:
            full_attn_ids = [num_blocks - 1]

        self.full_attn_ids = full_attn_ids

        for i in range(num_blocks):
            # Infer mlp hidden dim for this block from state dict
            block_mlp_hidden = None
            if vit_state_dict:
                fc1_key = f'blocks.{i}.mlp.fc1.weight'
                if fc1_key in vit_state_dict:
                    block_mlp_hidden = vit_state_dict[fc1_key].shape[0]

            # Infer rel_pos shape for this block
            rel_pos_shape = None
            if vit_state_dict:
                rel_h_key = f'blocks.{i}.attn.rel_pos_h'
                if rel_h_key in vit_state_dict:
                    rel_pos_shape = vit_state_dict[rel_h_key].shape

            block = MinimalViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_hidden=block_mlp_hidden,
                use_rel_pos=use_rel_pos,
                use_rope=use_rope,
                window_size=window_sizes[i] if i not in full_attn_ids else 0,
                rel_pos_shape=rel_pos_shape,
            )
            self.blocks.append(block)

    def forward(self, x):
        # This won't be called - TRTViTWrapper handles forward
        raise NotImplementedError("Use TRTViTWrapper for inference")


class PatchEmbed(nn.Module):
    """Patch embedding to match sam3's structure with proj submodule."""

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)  # BCHW -> BHWC


class MinimalViTBlock(nn.Module):
    """Minimal ViT block for weight loading."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_hidden: int = None,
        use_rel_pos: bool = True,
        use_rope: bool = False,
        window_size: int = 0,
        rel_pos_shape: tuple = None,
    ):
        super().__init__()
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MinimalAttention(
            dim=dim,
            num_heads=num_heads,
            use_rel_pos=use_rel_pos,
            use_rope=use_rope,
            rel_pos_shape=rel_pos_shape,
        )
        self.norm2 = nn.LayerNorm(dim)

        # Use exact hidden dim if provided, otherwise compute from ratio
        hidden_dim = mlp_hidden if mlp_hidden else int(dim * mlp_ratio)
        self.mlp = MinimalMLP(dim, hidden_dim)

    def forward(self, x):
        raise NotImplementedError()


class MinimalAttention(nn.Module):
    """Minimal attention for weight loading."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_rel_pos: bool = True,
        use_rope: bool = False,
        rel_pos_shape: tuple = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        if use_rel_pos:
            # Use exact shape from checkpoint if provided
            if rel_pos_shape:
                self.rel_pos_h = nn.Parameter(torch.zeros(rel_pos_shape))
                self.rel_pos_w = nn.Parameter(torch.zeros(rel_pos_shape))
            else:
                # Default shape
                self.rel_pos_h = nn.Parameter(torch.zeros(127, dim // num_heads))
                self.rel_pos_w = nn.Parameter(torch.zeros(127, dim // num_heads))

    def forward(self, x):
        raise NotImplementedError()


class MinimalMLP(nn.Module):
    """Minimal MLP for weight loading."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        raise NotImplementedError()


def main():
    parser = argparse.ArgumentParser(description='Export SAM3 ViT to TensorRT')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to SAM3 checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for engine files')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                       help='TensorRT precision mode')
    parser.add_argument('--workspace-gb', type=float, default=4.0,
                       help='TensorRT workspace size in GB')
    parser.add_argument('--dla-core', type=int, default=None,
                       help='DLA core to use (0 or 1 on Jetson)')
    parser.add_argument('--skip-onnx', action='store_true',
                       help='Skip ONNX export, use existing file')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    onnx_path = os.path.join(args.output_dir, 'sam3_encoder.onnx')
    engine_path = os.path.join(args.output_dir, 'sam3_encoder.engine')

    if not args.skip_onnx:
        # Load ViT from checkpoint
        vit = load_vit_from_checkpoint(args.checkpoint)

        # Wrap for TRT export
        logger.info("Wrapping ViT for TensorRT export...")
        wrapped_vit = TRTViTWrapper(vit)
        wrapped_vit = wrapped_vit.cuda().eval()

        # Export to ONNX
        logger.info(f"Exporting to ONNX: {onnx_path}")
        with torch.no_grad():
            export_vit_to_onnx(wrapped_vit, onnx_path)

        logger.info("ONNX export complete")
    else:
        logger.info(f"Skipping ONNX export, using existing: {onnx_path}")

    # Build TRT engine
    logger.info(f"Building TensorRT engine: {engine_path}")
    fp16 = args.precision in ('fp16', 'int8')
    int8 = args.precision == 'int8'

    success = build_trt_engine(
        onnx_path,
        engine_path,
        fp16=fp16,
        int8=int8,
        workspace_size_gb=args.workspace_gb,
        dla_core=args.dla_core,
    )

    if success:
        logger.info(f"Successfully built TensorRT engine: {engine_path}")
        # Print engine size
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        logger.info(f"Engine size: {engine_size:.1f} MB")
    else:
        logger.error("Failed to build TensorRT engine")
        sys.exit(1)


if __name__ == '__main__':
    main()
