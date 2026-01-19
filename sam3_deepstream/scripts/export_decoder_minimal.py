#!/usr/bin/env python3
"""
Minimal TRT export for SAM3 MaskDecoder that avoids importing the full sam3 package.
This script directly loads checkpoint weights and exports to TensorRT.

Key TRT compatibility fixes:
1. torch.repeat_interleave → torch.tile (ONNX/TRT incompatible)
2. F.scaled_dot_product_attention → explicit matmul (for TRT optimization)
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TRT-Compatible Attention (explicit matmul instead of SDPA)
# ============================================================================

class TRTAttention(nn.Module):
    """TensorRT-compatible attention using explicit matmul."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.head_dim = self.internal_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads: (B, N, C) -> (B, num_heads, N, head_dim)
        B, N_q, _ = q.shape
        _, N_k, _ = k.shape

        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Explicit attention computation (TRT-friendly)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Recombine heads: (B, num_heads, N, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N_q, self.internal_dim)
        out = self.out_proj(out)

        return out


# ============================================================================
# TRT-Compatible TwoWayTransformer
# ============================================================================

class TRTTwoWayAttentionBlock(nn.Module):
    """TRT-compatible two-way attention block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe

        self.self_attn = TRTAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = TRTAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = TRTAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention: tokens attending to image
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention: image attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TRTTwoWayTransformer(nn.Module):
    """TRT-compatible two-way transformer for mask decoder."""

    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            TRTTwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            for i in range(depth)
        ])

        self.final_attn_token_to_image = TRTAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Final attention from points to image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


# ============================================================================
# TRT-Compatible MaskDecoder
# ============================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs (BCHW format)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Simple MLP with optional sigmoid output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class TRTMaskDecoder(nn.Module):
    """
    TRT-compatible MaskDecoder with torch.tile instead of repeat_interleave.

    Key fix: Uses torch.tile for batch expansion (TRT/ONNX compatible)
    instead of torch.repeat_interleave (not supported).
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        pred_obj_scores: bool = False,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        # Transformer
        self.transformer = TRTTwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            num_heads=8,
            mlp_dim=2048,
        )

        # Output tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)

        # Output upscaling
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1)

        # Hypernetwork MLPs for mask prediction
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        TRT-compatible forward pass.

        Args:
            image_embeddings: (1, C, H, W) - encoder output (batch=1 for TRT)
            image_pe: (1, C, H, W) - positional encoding
            sparse_prompt_embeddings: (B, N, C) - point/box prompts
            dense_prompt_embeddings: (B, C, H, W) - mask prompts

        Returns:
            masks: (B, num_masks, H*4, W*4)
            iou_pred: (B, num_masks)
        """
        B = sparse_prompt_embeddings.shape[0]

        # Prepare output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat([
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ], dim=0)
            s = 1
        else:
            output_tokens = torch.cat([
                self.iou_token.weight,
                self.mask_tokens.weight,
            ], dim=0)

        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # ================================================================
        # KEY FIX: Use torch.tile instead of torch.repeat_interleave
        # Original: src = torch.repeat_interleave(image_embeddings, B, dim=0)
        # TRT-compatible: src = torch.tile(image_embeddings, (B, 1, 1, 1))
        # ================================================================
        src = torch.tile(image_embeddings, (B, 1, 1, 1))
        src = src + dense_prompt_embeddings

        pos_src = torch.tile(image_pe, (B, 1, 1, 1))

        b, c, h, w = src.shape

        # Run transformer
        hs, src_out = self.transformer(src, pos_src, tokens)

        # Extract IoU and mask tokens
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : s + 1 + self.num_mask_tokens, :]

        # Upscale mask embeddings
        src_out = src_out.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src_out)

        # Generate masks via hypernetwork MLPs
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate IoU predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# ============================================================================
# Weight Loading
# ============================================================================

def load_decoder_from_checkpoint(checkpoint_path: str) -> TRTMaskDecoder:
    """Load TRT-compatible decoder from SAM3 checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Find decoder prefix
    decoder_prefix = None
    for key in state_dict.keys():
        if 'mask_decoder' in key or 'sam_mask_decoder' in key:
            if 'transformer' in key:
                # Extract prefix: e.g., "sam_mask_decoder." or "detector.mask_decoder."
                idx = key.find('mask_decoder')
                decoder_prefix = key[:idx + len('mask_decoder') + 1]
                break

    if decoder_prefix is None:
        raise ValueError("Could not find mask_decoder in checkpoint")

    logger.info(f"Decoder weight prefix: '{decoder_prefix}'")

    # Extract decoder weights
    decoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(decoder_prefix):
            new_key = key[len(decoder_prefix):]
            decoder_state_dict[new_key] = value

    logger.info(f"Extracted {len(decoder_state_dict)} decoder weights")

    # Infer decoder config from weights
    transformer_dim = decoder_state_dict['iou_token.weight'].shape[1]
    num_mask_tokens = decoder_state_dict['mask_tokens.weight'].shape[0]
    num_multimask_outputs = num_mask_tokens - 1
    pred_obj_scores = 'obj_score_token.weight' in decoder_state_dict
    use_high_res_features = 'conv_s0.weight' in decoder_state_dict

    logger.info(f"Inferred config: transformer_dim={transformer_dim}, "
                f"num_multimask_outputs={num_multimask_outputs}, "
                f"pred_obj_scores={pred_obj_scores}, "
                f"use_high_res_features={use_high_res_features}")

    # Create TRT decoder
    decoder = TRTMaskDecoder(
        transformer_dim=transformer_dim,
        num_multimask_outputs=num_multimask_outputs,
        pred_obj_scores=pred_obj_scores,
        use_high_res_features=use_high_res_features,
    )

    # Map original weights to TRT decoder structure
    mapped_state_dict = map_decoder_weights(decoder_state_dict, decoder)

    # Load weights
    missing, unexpected = decoder.load_state_dict(mapped_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing {len(missing)} keys: {missing[:10]}...")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys: {unexpected[:10]}...")

    return decoder


def map_decoder_weights(src_dict: Dict, trt_decoder: TRTMaskDecoder) -> Dict:
    """Map original decoder weights to TRT decoder structure."""
    mapped = {}

    # Direct mappings (same structure)
    direct_keys = [
        'iou_token.weight',
        'mask_tokens.weight',
        'iou_prediction_head.layers.0.weight',
        'iou_prediction_head.layers.0.bias',
        'iou_prediction_head.layers.1.weight',
        'iou_prediction_head.layers.1.bias',
        'iou_prediction_head.layers.2.weight',
        'iou_prediction_head.layers.2.bias',
    ]

    for key in direct_keys:
        if key in src_dict:
            mapped[key] = src_dict[key]

    # Output upscaling
    upscaling_map = {
        'output_upscaling.0.weight': 'output_upscaling.0.weight',
        'output_upscaling.0.bias': 'output_upscaling.0.bias',
        'output_upscaling.1.weight': 'output_upscaling.1.weight',
        'output_upscaling.1.bias': 'output_upscaling.1.bias',
        'output_upscaling.3.weight': 'output_upscaling.3.weight',
        'output_upscaling.3.bias': 'output_upscaling.3.bias',
    }
    for src_key, dst_key in upscaling_map.items():
        if src_key in src_dict:
            mapped[dst_key] = src_dict[src_key]

    # Hypernetwork MLPs
    for i in range(trt_decoder.num_mask_tokens):
        for j in range(3):
            src_w = f'output_hypernetworks_mlps.{i}.layers.{j}.weight'
            src_b = f'output_hypernetworks_mlps.{i}.layers.{j}.bias'
            if src_w in src_dict:
                mapped[src_w] = src_dict[src_w]
            if src_b in src_dict:
                mapped[src_b] = src_dict[src_b]

    # Transformer weights - need to map attention structure
    # Original: transformer.layers.{i}.{module}.{proj}
    # TRT: transformer.layers.{i}.{module}.{proj}
    for i in range(2):  # 2 layers
        layer_prefix = f'transformer.layers.{i}'

        # Map attention modules
        attn_modules = [
            'self_attn',
            'cross_attn_token_to_image',
            'cross_attn_image_to_token',
        ]

        for attn_name in attn_modules:
            for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                for param in ['weight', 'bias']:
                    src_key = f'{layer_prefix}.{attn_name}.{proj}.{param}'
                    if src_key in src_dict:
                        mapped[src_key] = src_dict[src_key]

        # Map layer norms
        for norm_name in ['norm1', 'norm2', 'norm3', 'norm4']:
            for param in ['weight', 'bias']:
                src_key = f'{layer_prefix}.{norm_name}.{param}'
                if src_key in src_dict:
                    mapped[src_key] = src_dict[src_key]

        # Map MLP - original structure: mlp.lin1, mlp.lin2
        # TRT structure: mlp.0, mlp.2
        mlp_map = {
            f'{layer_prefix}.mlp.lin1.weight': f'{layer_prefix}.mlp.0.weight',
            f'{layer_prefix}.mlp.lin1.bias': f'{layer_prefix}.mlp.0.bias',
            f'{layer_prefix}.mlp.lin2.weight': f'{layer_prefix}.mlp.2.weight',
            f'{layer_prefix}.mlp.lin2.bias': f'{layer_prefix}.mlp.2.bias',
        }
        for src_key, dst_key in mlp_map.items():
            if src_key in src_dict:
                mapped[dst_key] = src_dict[src_key]

    # Final attention
    for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        for param in ['weight', 'bias']:
            src_key = f'transformer.final_attn_token_to_image.{proj}.{param}'
            if src_key in src_dict:
                mapped[src_key] = src_dict[src_key]

    # Final norm
    for param in ['weight', 'bias']:
        src_key = f'transformer.norm_final_attn.{param}'
        if src_key in src_dict:
            mapped[src_key] = src_dict[src_key]

    # Object score token if present
    if 'obj_score_token.weight' in src_dict:
        mapped['obj_score_token.weight'] = src_dict['obj_score_token.weight']
        if 'pred_obj_score_head.weight' in src_dict:
            mapped['pred_obj_score_head.weight'] = src_dict['pred_obj_score_head.weight']
            mapped['pred_obj_score_head.bias'] = src_dict['pred_obj_score_head.bias']

    # High-res feature convs if present
    if 'conv_s0.weight' in src_dict:
        mapped['conv_s0.weight'] = src_dict['conv_s0.weight']
        mapped['conv_s0.bias'] = src_dict['conv_s0.bias']
        mapped['conv_s1.weight'] = src_dict['conv_s1.weight']
        mapped['conv_s1.bias'] = src_dict['conv_s1.bias']

    return mapped


# ============================================================================
# Export Functions
# ============================================================================

def export_decoder_to_onnx(
    model: TRTMaskDecoder,
    output_path: str,
    feature_size: int = 64,
    max_prompts: int = 10,
    opset_version: int = 17,
):
    """Export TRT decoder to ONNX format."""
    model.eval()
    device = next(model.parameters()).device
    dim = model.transformer_dim

    # Create dummy inputs
    dummy_image_embeddings = torch.randn(1, dim, feature_size, feature_size, device=device)
    dummy_image_pe = torch.randn(1, dim, feature_size, feature_size, device=device)
    dummy_sparse_prompts = torch.randn(1, max_prompts, dim, device=device)
    dummy_dense_prompts = torch.randn(1, dim, feature_size, feature_size, device=device)

    # Export
    torch.onnx.export(
        model,
        (dummy_image_embeddings, dummy_image_pe, dummy_sparse_prompts, dummy_dense_prompts),
        output_path,
        opset_version=opset_version,
        input_names=['image_embeddings', 'image_pe', 'sparse_prompt_embeddings', 'dense_prompt_embeddings'],
        output_names=['masks', 'iou_predictions'],
        dynamic_axes={
            'sparse_prompt_embeddings': {0: 'batch', 1: 'num_prompts'},
            'dense_prompt_embeddings': {0: 'batch'},
            'masks': {0: 'batch'},
            'iou_predictions': {0: 'batch'},
        },
        do_constant_folding=True,
    )
    logger.info(f"Exported decoder to ONNX: {output_path}")


def validate_onnx(onnx_path: str) -> bool:
    """Validate ONNX model for TRT compatibility."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Check for problematic ops
        problematic_ops = ['Complex', 'RepeatInterleave']
        for node in model.graph.node:
            for op in problematic_ops:
                if op in node.op_type:
                    logger.error(f"Found TRT-incompatible op: {node.op_type}")
                    return False

        logger.info("ONNX validation passed")
        return True
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_size_gb: float = 4.0,
) -> bool:
    """Build TensorRT engine from ONNX model."""
    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                return False

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size_gb * 1024 ** 3))

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Set optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()

        # sparse_prompt_embeddings: (batch, num_prompts, dim)
        # batch: 1-4, num_prompts: 1-20
        profile.set_shape(
            'sparse_prompt_embeddings',
            min=(1, 1, 256),
            opt=(1, 5, 256),
            max=(4, 20, 256),
        )
        profile.set_shape(
            'dense_prompt_embeddings',
            min=(1, 256, 64, 64),
            opt=(1, 256, 64, 64),
            max=(4, 256, 64, 64),
        )

        config.add_optimization_profile(profile)

        # Build engine
        logger.info("Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        logger.info(f"Saved TensorRT engine: {engine_path}")
        return True

    except Exception as e:
        logger.error(f"TRT build failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export SAM3 MaskDecoder to TensorRT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SAM3 checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for engine files')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16',
                        help='TensorRT precision mode')
    parser.add_argument('--workspace-gb', type=float, default=4.0,
                        help='TensorRT workspace size in GB')
    parser.add_argument('--feature-size', type=int, default=64,
                        help='Encoder output feature map size')
    parser.add_argument('--skip-onnx', action='store_true',
                        help='Skip ONNX export, use existing file')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing ONNX file')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    onnx_path = os.path.join(args.output_dir, 'sam3_decoder.onnx')
    engine_path = os.path.join(args.output_dir, 'sam3_decoder.engine')

    if args.validate_only:
        validate_onnx(onnx_path)
        return

    if not args.skip_onnx:
        # Load decoder from checkpoint
        decoder = load_decoder_from_checkpoint(args.checkpoint)
        decoder = decoder.cuda().eval()

        # Test forward pass
        logger.info("Testing forward pass...")
        with torch.no_grad():
            dummy_img = torch.randn(1, 256, args.feature_size, args.feature_size, device='cuda')
            dummy_pe = torch.randn(1, 256, args.feature_size, args.feature_size, device='cuda')
            dummy_sparse = torch.randn(1, 5, 256, device='cuda')
            dummy_dense = torch.randn(1, 256, args.feature_size, args.feature_size, device='cuda')

            masks, iou = decoder(dummy_img, dummy_pe, dummy_sparse, dummy_dense)
            logger.info(f"Test output - masks: {masks.shape}, iou: {iou.shape}")

        # Export to ONNX
        logger.info(f"Exporting to ONNX: {onnx_path}")
        with torch.no_grad():
            export_decoder_to_onnx(decoder, onnx_path, feature_size=args.feature_size)

        # Validate ONNX
        if not validate_onnx(onnx_path):
            logger.error("ONNX validation failed, aborting TRT build")
            sys.exit(1)
    else:
        logger.info(f"Skipping ONNX export, using existing: {onnx_path}")

    # Build TRT engine
    logger.info(f"Building TensorRT engine: {engine_path}")
    fp16 = args.precision == 'fp16'

    success = build_trt_engine(
        onnx_path,
        engine_path,
        fp16=fp16,
        workspace_size_gb=args.workspace_gb,
    )

    if success:
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        logger.info(f"Successfully built decoder engine: {engine_path}")
        logger.info(f"Engine size: {engine_size:.1f} MB")
    else:
        logger.error("Failed to build TensorRT engine")
        sys.exit(1)


if __name__ == '__main__':
    main()
