"""Decoder TensorRT export for SAM3 UniversalSegmentationHead."""

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sam3.model.trt_export import build_trt_engine

from ..config import get_config, Precision

logger = logging.getLogger(__name__)


class TRTPixelDecoder(nn.Module):
    """TensorRT-compatible PixelDecoder wrapper."""

    def __init__(self, pixel_decoder: nn.Module):
        super().__init__()
        self.hidden_dim = pixel_decoder.hidden_dim
        self.num_upsampling_stages = pixel_decoder.num_upsampling_stages
        self.interpolation_mode = pixel_decoder.interpolation_mode
        self.conv_layers = pixel_decoder.conv_layers
        self.norms = pixel_decoder.norms
        self.shared_conv = pixel_decoder.shared_conv

    def forward(self, backbone_feats: List[Tensor]) -> Tensor:
        """Forward pass matching original PixelDecoder."""
        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]

        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            prev_fpn = curr_fpn + F.interpolate(
                prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode
            )
            if self.shared_conv:
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn


class TRTMaskPredictor(nn.Module):
    """TensorRT-compatible MaskPredictor wrapper."""

    def __init__(self, mask_predictor: nn.Module):
        super().__init__()
        self.mask_embed = mask_predictor.mask_embed

    def forward(self, obj_queries: Tensor, pixel_embed: Tensor) -> Tensor:
        """
        Predict masks from object queries and pixel embeddings.

        Args:
            obj_queries: (B, Q, C) object query embeddings
            pixel_embed: (B, C, H, W) pixel embeddings

        Returns:
            mask_preds: (B, Q, H, W) predicted masks
        """
        # Apply mask embedding MLP
        query_embed = self.mask_embed(obj_queries)  # (B, Q, C)

        # Einsum for mask prediction: (B, Q, C) x (B, C, H, W) -> (B, Q, H, W)
        B, Q, C = query_embed.shape
        _, _, H, W = pixel_embed.shape

        # Explicit matmul instead of einsum for better TRT compatibility
        query_flat = query_embed.view(B, Q, C)
        pixel_flat = pixel_embed.view(B, C, H * W)
        mask_preds = torch.bmm(query_flat, pixel_flat).view(B, Q, H, W)

        return mask_preds


class TRTSegmentationHeadWrapper(nn.Module):
    """
    TensorRT-compatible wrapper for SAM3 UniversalSegmentationHead.

    This wrapper prepares the segmentation head for ONNX/TensorRT export by:
    1. Simplifying the forward pass for fixed input shapes
    2. Using explicit operations instead of dynamic indexing
    3. Removing optional components not needed for inference

    Args:
        seg_head: Original UniversalSegmentationHead module
        hidden_dim: Hidden dimension (default: 256)
    """

    def __init__(
        self,
        seg_head: nn.Module,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Core components
        self.pixel_decoder = TRTPixelDecoder(seg_head.pixel_decoder)
        self.mask_predictor = TRTMaskPredictor(seg_head.mask_predictor)
        self.instance_seg_head = seg_head.instance_seg_head
        self.semantic_seg_head = seg_head.semantic_seg_head

        # Optional cross-attention for prompts
        self.has_cross_attend = seg_head.cross_attend_prompt is not None
        if self.has_cross_attend:
            self.cross_attend_prompt = seg_head.cross_attend_prompt
            self.cross_attn_norm = seg_head.cross_attn_norm

        # Optional presence head
        self.has_presence_head = seg_head.presence_head is not None
        if self.has_presence_head:
            self.presence_head = seg_head.presence_head

        # Check for no_dec mode
        self.no_dec = seg_head.no_dec

    def forward(
        self,
        backbone_feats: List[Tensor],
        obj_queries: Tensor,
        encoder_hidden_states: Tensor,
        prompt: Optional[Tensor] = None,
        prompt_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Forward pass for TensorRT inference.

        Args:
            backbone_feats: List of backbone feature maps at different scales
            obj_queries: (L, B, Q, C) object queries from transformer decoder
            encoder_hidden_states: (S, B, C) encoder hidden states
            prompt: Optional (B, P, C) prompt embeddings
            prompt_mask: Optional (B, P) prompt mask

        Returns:
            pred_masks: (B, Q, H, W) predicted instance masks
            semantic_seg: (B, 1, H, W) semantic segmentation
            presence_logit: Optional (B, 1) presence logit
        """
        bs = encoder_hidden_states.shape[1]

        # Cross-attend prompts if available
        if self.has_cross_attend and prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        # Presence logit
        presence_logit = None
        if self.has_presence_head and prompt is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = self.presence_head(
                pooled_enc.view(1, bs, 1, self.hidden_dim),
                prompt=prompt,
                prompt_mask=prompt_mask,
            ).squeeze(0).squeeze(1)

        # Pixel embedding via pixel decoder
        pixel_embed = self.pixel_decoder(backbone_feats)

        # Instance segmentation head
        instance_embeds = self.instance_seg_head(pixel_embed)

        # Mask prediction
        if self.no_dec:
            mask_pred = self.mask_predictor.mask_embed(instance_embeds)
        else:
            # Use last layer of obj_queries
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        # Semantic segmentation
        semantic_seg = self.semantic_seg_head(pixel_embed)

        return mask_pred, semantic_seg, presence_logit


class TRTSegmentationHeadSimple(nn.Module):
    """
    Simplified TRT wrapper for inference without prompts.

    This is optimized for the common case where we just need mask predictions
    from backbone features and object queries.
    """

    def __init__(self, seg_head: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Core components only
        self.pixel_decoder = TRTPixelDecoder(seg_head.pixel_decoder)
        self.mask_predictor = TRTMaskPredictor(seg_head.mask_predictor)
        self.instance_seg_head = seg_head.instance_seg_head

    def forward(
        self,
        backbone_feat: Tensor,
        obj_queries: Tensor,
    ) -> Tensor:
        """
        Simplified forward for basic inference.

        Args:
            backbone_feat: (B, C, H, W) single-scale backbone features
            obj_queries: (B, Q, C) object queries

        Returns:
            mask_preds: (B, Q, H, W) predicted masks
        """
        # Pixel decoder expects list of features
        pixel_embed = self.pixel_decoder([backbone_feat])
        instance_embeds = self.instance_seg_head(pixel_embed)
        mask_preds = self.mask_predictor(obj_queries, instance_embeds)
        return mask_preds


def export_decoder_to_onnx(
    model: nn.Module,
    output_path: str,
    hidden_dim: int = 256,
    feature_size: int = 72,
    num_queries: int = 100,
    opset_version: int = 17,
    verbose: bool = False,
) -> None:
    """
    Export TRT-wrapped segmentation head to ONNX format.

    Args:
        model: TRTSegmentationHeadSimple model
        output_path: Path for ONNX file output
        hidden_dim: Hidden dimension
        feature_size: Feature map height/width
        num_queries: Number of object queries
        opset_version: ONNX opset version
        verbose: Whether to print verbose export info
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy inputs
    batch_size = 1
    dummy_backbone_feat = torch.randn(
        batch_size, hidden_dim, feature_size, feature_size, device=device
    )
    dummy_obj_queries = torch.randn(
        batch_size, num_queries, hidden_dim, device=device
    )

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_backbone_feat, dummy_obj_queries),
        output_path,
        opset_version=opset_version,
        input_names=["backbone_features", "object_queries"],
        output_names=["mask_predictions"],
        dynamic_axes={
            "object_queries": {1: "num_queries"},
            "mask_predictions": {1: "num_queries"},
        },
        do_constant_folding=True,
        verbose=verbose,
    )

    logger.info(f"Exported segmentation head to ONNX: {output_path}")


def export_decoder_to_tensorrt(
    sam3_model: nn.Module,
    output_dir: Optional[Union[str, Path]] = None,
    precision: Precision = Precision.FP16,
    workspace_size_gb: float = 4.0,
) -> Path:
    """
    Export SAM3 segmentation head to TensorRT.

    Args:
        sam3_model: SAM3 model containing segmentation_head
        output_dir: Directory for output files
        precision: TensorRT precision mode
        workspace_size_gb: GPU workspace size

    Returns:
        Path to the TensorRT engine file
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.trt.cache_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract segmentation head from SAM3 model
    seg_head = _extract_segmentation_head(sam3_model)

    if seg_head is None:
        raise ValueError(
            "Could not find segmentation_head in SAM3 model. "
            f"Model type: {type(sam3_model).__name__}"
        )

    # Get hidden dimension
    hidden_dim = getattr(seg_head, 'd_model', 256)

    # Wrap for TRT export (use simplified version)
    logger.info("Wrapping segmentation head for TensorRT export...")
    wrapped_decoder = TRTSegmentationHeadSimple(seg_head, hidden_dim=hidden_dim)

    # Use CPU for ONNX export to avoid GPU memory issues
    logger.info("Using CPU for ONNX export (avoids GPU memory issues)...")
    wrapped_decoder = wrapped_decoder.cpu().eval()

    # Export to ONNX
    onnx_path = output_dir / "sam3_decoder.onnx"
    logger.info(f"Exporting to ONNX: {onnx_path}")
    export_decoder_to_onnx(wrapped_decoder, str(onnx_path), hidden_dim=hidden_dim)

    # Build TRT engine
    engine_path = output_dir / "sam3_decoder.engine"
    logger.info(f"Building TensorRT engine: {engine_path}")

    fp16 = precision in (Precision.FP16, Precision.INT8)
    success = build_trt_engine(
        str(onnx_path),
        str(engine_path),
        fp16=fp16,
        workspace_size_gb=workspace_size_gb,
    )

    if not success:
        raise RuntimeError("Failed to build decoder TensorRT engine")

    logger.info(f"Decoder engine saved to: {engine_path}")
    return engine_path


def _extract_segmentation_head(sam3_model: nn.Module) -> Optional[nn.Module]:
    """Extract UniversalSegmentationHead from SAM3 model."""
    # SAM3 Image model has segmentation_head directly
    if hasattr(sam3_model, "segmentation_head"):
        seg_head = sam3_model.segmentation_head
        # Verify it has the expected components
        if hasattr(seg_head, "pixel_decoder") and hasattr(seg_head, "mask_predictor"):
            logger.info(f"Found segmentation_head: {type(seg_head).__name__}")
            return seg_head

    # Try other common paths
    paths = [
        ("detector", "segmentation_head"),
        ("tracker", "segmentation_head"),
    ]

    for path in paths:
        obj = sam3_model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if obj is not None and hasattr(obj, "pixel_decoder"):
                logger.info(f"Found segmentation_head at {'.'.join(path)}")
                return obj
        except AttributeError:
            continue

    return None


def get_decoder_engine_path() -> Optional[Path]:
    """Get path to existing decoder engine if available."""
    config = get_config()

    if config.decoder_engine and config.decoder_engine.exists():
        return config.decoder_engine

    # Check default cache location
    default_path = config.trt.cache_dir / "sam3_decoder.engine"
    if default_path.exists():
        return default_path

    return None
