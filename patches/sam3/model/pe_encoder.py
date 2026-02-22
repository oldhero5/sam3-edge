# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Perception Encoder backbone wrapper for SAM3.

Based on the PE paper (arXiv:2504.13181):
- "Best visual embeddings are NOT at the output of the network"
- Intermediate layers contain task-specific features
- Alignment tuning lifts intermediate features to output space

This module wraps a ViT backbone with alignment tuning for dense prediction,
following PE_spatial's approach for segmentation tasks.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentTuningLayer(nn.Module):
    """
    Alignment tuning layer that projects intermediate ViT features.

    From PE paper Section 4.3: "We lift the intermediate features from
    layers {8, 16, 24, 31} and project them to the output dimension."
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 256,
        use_layer_norm: bool = True,
        use_gelu: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, output_dim))
        if use_gelu:
            layers.append(nn.GELU())

        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features from intermediate layer.

        Args:
            x: (B, N, input_dim) features from intermediate ViT layer

        Returns:
            (B, N, output_dim) aligned features
        """
        return self.proj(x)


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuses features from multiple intermediate layers.

    Uses a 1x1 convolution to combine multi-scale aligned features
    into a single feature map.
    """

    def __init__(
        self,
        num_layers: int,
        feature_dim: int,
        output_dim: int,
        use_learned_weights: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # 1x1 conv for feature fusion
        self.fusion_conv = nn.Conv2d(
            num_layers * feature_dim,
            output_dim,
            kernel_size=1,
            bias=True,
        )

        # Optional learned weights for each layer
        if use_learned_weights:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            self.register_buffer(
                "layer_weights", torch.ones(num_layers) / num_layers
            )

    def forward(
        self,
        features: List[torch.Tensor],
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Fuse multi-scale features.

        Args:
            features: List of (B, N, D) aligned features from each layer
            spatial_size: (H, W) spatial dimensions, inferred if None

        Returns:
            (B, output_dim, H, W) fused feature map
        """
        B, N, D = features[0].shape

        # Infer spatial size
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        # Normalize layer weights
        weights = F.softmax(self.layer_weights, dim=0)

        # Reshape and weight features
        weighted_features = []
        for i, feat in enumerate(features):
            # (B, N, D) -> (B, D, H, W)
            feat_2d = feat.transpose(1, 2).reshape(B, D, H, W)
            weighted_features.append(feat_2d * weights[i])

        # Concatenate along channel dimension
        stacked = torch.cat(weighted_features, dim=1)  # (B, num_layers*D, H, W)

        # Fuse with 1x1 conv
        fused = self.fusion_conv(stacked)  # (B, output_dim, H, W)

        return fused


class PEVisionEncoder(nn.Module):
    """
    Perception Encoder vision backbone for SAM3.

    This wraps the existing ViT with alignment tuning that extracts
    features from multiple intermediate layers and fuses them for
    improved dense prediction performance.

    Key insight from PE paper: The best features for different tasks
    are found at different depths in the network:
    - Layer 8: Low-level features (edges, textures)
    - Layer 16: Mid-level features (parts, patterns)
    - Layer 24: High-level features (objects, scenes)
    - Layer 31: Final features (semantic concepts)

    Args:
        img_size: Input image size
        patch_size: Patch size for ViT
        embed_dim: ViT embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        intermediate_layers: Which layers to extract features from
        output_dim: Output feature dimension (256 for SAM3)
        use_alignment_tuning: Whether to use intermediate layer fusion
        compile_mode: Torch compile mode (None, "default", etc.)
    """

    def __init__(
        self,
        img_size: int = 1008,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.625,
        intermediate_layers: Tuple[int, ...] = (7, 15, 23, 31),
        output_dim: int = 256,
        use_alignment_tuning: bool = True,
        compile_mode: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.intermediate_layers = intermediate_layers
        self.output_dim = output_dim
        self.use_alignment_tuning = use_alignment_tuning

        # Create ViT backbone
        self.backbone = self._create_vit_backbone(
            img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, compile_mode
        )

        if use_alignment_tuning:
            # Create alignment layers for each intermediate layer
            self.alignment_layers = nn.ModuleList(
                [
                    AlignmentTuningLayer(
                        input_dim=embed_dim,
                        output_dim=output_dim,
                        use_layer_norm=True,
                        use_gelu=True,
                    )
                    for _ in intermediate_layers
                ]
            )

            # Feature fusion module
            self.feature_fusion = MultiScaleFeatureFusion(
                num_layers=len(intermediate_layers),
                feature_dim=output_dim,
                output_dim=output_dim,
                use_learned_weights=True,
            )
        else:
            # Single projection from final layer
            self.final_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, output_dim),
            )

        # Calculate spatial dimensions
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

    def _create_vit_backbone(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        compile_mode: Optional[str],
    ):
        """Create the PE-compatible ViT backbone."""
        from sam3.model.vitdet import ViT

        return ViT(
            img_size=img_size,
            pretrain_img_size=336,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer="LayerNorm",
            drop_path_rate=0.1,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            # PE-specific: global attention at layers 7, 15, 23, 31
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            # Enable intermediate feature extraction
            return_interm_layers=self.use_alignment_tuning,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features with intermediate layer outputs.

        Args:
            x: (B, 3, H, W) input images

        Returns:
            features: (B, N, output_dim) main output features
            intermediate_features: List of (B, N, output_dim) aligned intermediate features
        """
        if self.use_alignment_tuning:
            # Get intermediate features from backbone
            intermediate_outputs = self.backbone.forward_intermediate(
                x, self.intermediate_layers
            )

            # Apply alignment tuning to each layer
            aligned_features = []
            for feat, align_layer in zip(intermediate_outputs, self.alignment_layers):
                aligned_features.append(align_layer(feat))

            # Fuse features
            H = W = self.num_patches_per_side
            fused = self.feature_fusion(aligned_features, (H, W))

            # Convert back to sequence format: (B, D, H, W) -> (B, H*W, D)
            features = fused.flatten(2).transpose(1, 2)

            return features, aligned_features
        else:
            # Standard forward without alignment tuning
            backbone_output = self.backbone(x)
            features = self.final_proj(backbone_output)
            return features, [features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning main features.

        Args:
            x: (B, 3, H, W) input images

        Returns:
            (B, N, output_dim) output features
        """
        features, _ = self.forward_features(x)
        return features

    def get_intermediate_features(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get aligned intermediate features for multi-scale processing.

        Args:
            x: (B, 3, H, W) input images

        Returns:
            List of (B, N, output_dim) aligned features from each intermediate layer
        """
        _, aligned_features = self.forward_features(x)
        return aligned_features


class PEViTNeckAdapter(nn.Module):
    """
    Adapter to make PEVisionEncoder compatible with Sam3DualViTDetNeck.

    The SAM3 neck expects a ViT with a specific interface:
    - channel_list: List of output channel dimensions
    - forward(tensor_list) -> List[Tensor] in (B, C, H, W) format

    This adapter wraps PEVisionEncoder to provide that interface while
    preserving the alignment tuning capabilities. The alignment tuning
    features are accessible via get_aligned_features() for downstream use.
    """

    def __init__(self, pe_encoder: PEVisionEncoder):
        super().__init__()
        self.pe_encoder = pe_encoder

        # Expose backbone properties expected by neck
        self.embed_dim = pe_encoder.embed_dim
        self.num_patches = pe_encoder.num_patches

        # channel_list is required by Sam3DualViTDetNeck
        # The neck uses channel_list[-1] to determine conv dimensions
        # We expose the backbone's channel_list directly
        self._channel_list = pe_encoder.backbone.channel_list

    @property
    def channel_list(self) -> List[int]:
        """Return output channel dimensions expected by neck."""
        return self._channel_list

    def forward(self, tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass compatible with neck expectations.

        The neck calls trunk(tensor_list) expecting:
        - Input: List of image tensors
        - Output: List of feature tensors in (B, C, H, W) format

        We pass through the underlying ViT backbone directly to maintain
        compatibility with the neck's expected dimensions (1024-dim).

        Args:
            tensor_list: List containing batched images (B, 3, H, W)

        Returns:
            List of feature tensors in (B, embed_dim, H, W) format
        """
        # The neck passes a list but typically uses only the first element
        x = tensor_list[0] if isinstance(tensor_list, list) else tensor_list

        # Process through the underlying ViT backbone directly
        # This returns features in (B, C, H, W) format as expected by neck
        outputs = self.pe_encoder.backbone(x)

        return outputs

    def get_aligned_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Get alignment-tuned features from PE encoder.

        This provides access to the PE alignment tuning features for
        downstream use (e.g., in text-vision fusion).

        Args:
            x: (B, 3, H, W) input images

        Returns:
            features: (B, N, output_dim) fused aligned features
            intermediate: List of aligned intermediate features
        """
        return self.pe_encoder.forward_features(x)

    @property
    def pos_embed(self):
        """Expose position embedding from backbone."""
        return self.pe_encoder.backbone.pos_embed

    def get_pos_embed(self, H: int, W: int):
        """Get interpolated position embedding."""
        return self.pe_encoder.backbone.get_pos_embed(H, W)
