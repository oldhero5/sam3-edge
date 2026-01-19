"""Decoder TensorRT export - NEW export for SAM3 MaskDecoder."""

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


class TRTMaskDecoderWrapper(nn.Module):
    """
    TensorRT-compatible wrapper for SAM3 MaskDecoder.

    This wrapper prepares the MaskDecoder for ONNX/TensorRT export by:
    1. Using explicit operations instead of complex ops
    2. Handling fixed input shapes for optimal TRT performance
    3. Pre-allocating token embeddings as buffers

    Args:
        mask_decoder: Original MaskDecoder module
        transformer_dim: Transformer dimension (default: 256)
    """

    def __init__(
        self,
        mask_decoder: nn.Module,
        transformer_dim: int = 256,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim

        # Copy core components
        self.transformer = mask_decoder.transformer
        self.iou_token = mask_decoder.iou_token
        self.mask_tokens = mask_decoder.mask_tokens
        self.num_mask_tokens = mask_decoder.num_mask_tokens

        # Output layers
        self.output_upscaling = mask_decoder.output_upscaling
        self.output_hypernetworks_mlps = mask_decoder.output_hypernetworks_mlps
        self.iou_prediction_head = mask_decoder.iou_prediction_head

        # Optional high-res features
        self.use_high_res_features = mask_decoder.use_high_res_features
        if self.use_high_res_features:
            self.conv_s0 = mask_decoder.conv_s0
            self.conv_s1 = mask_decoder.conv_s1

        # Optional object score prediction
        self.pred_obj_scores = mask_decoder.pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = mask_decoder.obj_score_token
            self.pred_obj_score_head = mask_decoder.pred_obj_score_head

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for TensorRT inference.

        Args:
            image_embeddings: (B, C, H, W) image encoder output
            image_pe: (B, C, H, W) positional encodings
            sparse_prompt_embeddings: (B, N, C) point/box prompt embeddings
            dense_prompt_embeddings: (B, C, H, W) mask prompt embeddings

        Returns:
            masks: (B, num_masks, H*4, W*4) predicted masks
            iou_pred: (B, num_masks) predicted IoU scores
        """
        masks, iou_pred = self._predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        return masks, iou_pred

    def _predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Core mask prediction logic."""
        B = image_embeddings.shape[0]

        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)

        # Add object score token if present
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [self.obj_score_token.weight.unsqueeze(0).expand(B, -1, -1), output_tokens],
                dim=1,
            )

        # Combine with sparse prompts
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Expand image embeddings for batch
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Extract IoU token output
        iou_token_out = hs[:, 1 if self.pred_obj_scores else 0, :]

        # Extract mask tokens output
        mask_start = 2 if self.pred_obj_scores else 1
        mask_tokens_out = hs[:, mask_start : mask_start + self.num_mask_tokens, :]

        # Upscale mask embeddings
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # Generate mask predictions via hypernetwork MLPs
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


def export_decoder_to_onnx(
    model: TRTMaskDecoderWrapper,
    output_path: str,
    image_size: int = 64,  # Feature map size (1008/14 = 72, but use 64 for power of 2)
    transformer_dim: int = 256,
    max_prompts: int = 10,
    opset_version: int = 17,
    verbose: bool = False,
) -> None:
    """
    Export TRT-wrapped MaskDecoder to ONNX format.

    Args:
        model: TRTMaskDecoderWrapper model
        output_path: Path for ONNX file output
        image_size: Feature map height/width
        transformer_dim: Transformer channel dimension
        max_prompts: Maximum number of prompt points
        opset_version: ONNX opset version
        verbose: Whether to print verbose export info
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy inputs
    batch_size = 1
    dummy_image_embeddings = torch.randn(
        batch_size, transformer_dim, image_size, image_size, device=device
    )
    dummy_image_pe = torch.randn(
        batch_size, transformer_dim, image_size, image_size, device=device
    )
    dummy_sparse_prompts = torch.randn(
        batch_size, max_prompts, transformer_dim, device=device
    )
    dummy_dense_prompts = torch.randn(
        batch_size, transformer_dim, image_size, image_size, device=device
    )

    # Export to ONNX with dynamic axes for prompt count
    torch.onnx.export(
        model,
        (dummy_image_embeddings, dummy_image_pe, dummy_sparse_prompts, dummy_dense_prompts),
        output_path,
        opset_version=opset_version,
        input_names=[
            "image_embeddings",
            "image_pe",
            "sparse_prompt_embeddings",
            "dense_prompt_embeddings",
        ],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "sparse_prompt_embeddings": {1: "num_prompts"},
        },
        do_constant_folding=True,
        verbose=verbose,
    )

    logger.info(f"Exported MaskDecoder to ONNX: {output_path}")


def export_decoder_to_tensorrt(
    sam3_model: nn.Module,
    output_dir: Optional[Union[str, Path]] = None,
    precision: Precision = Precision.FP16,
    workspace_size_gb: float = 4.0,
) -> Path:
    """
    Export SAM3 MaskDecoder to TensorRT.

    Args:
        sam3_model: SAM3 model containing mask decoder
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

    # Extract MaskDecoder from SAM3 model
    mask_decoder = _extract_mask_decoder(sam3_model)

    if mask_decoder is None:
        raise ValueError("Could not find MaskDecoder in SAM3 model")

    # Wrap for TRT export
    logger.info("Wrapping MaskDecoder for TensorRT export...")
    wrapped_decoder = TRTMaskDecoderWrapper(mask_decoder)
    wrapped_decoder = wrapped_decoder.cuda().eval()

    # Export to ONNX
    onnx_path = output_dir / "sam3_decoder.onnx"
    logger.info(f"Exporting to ONNX: {onnx_path}")
    export_decoder_to_onnx(wrapped_decoder, str(onnx_path))

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


def _extract_mask_decoder(sam3_model: nn.Module) -> Optional[nn.Module]:
    """Extract MaskDecoder from various SAM3 model structures."""
    # Try common paths
    paths = [
        # Video inference models
        ("sam_mask_decoder",),
        ("mask_decoder",),
        # Image models
        ("detector", "sam_mask_decoder"),
        ("detector", "mask_decoder"),
        # Tracker models
        ("tracker", "sam_mask_decoder"),
    ]

    for path in paths:
        obj = sam3_model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if obj is not None and hasattr(obj, "transformer"):
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
