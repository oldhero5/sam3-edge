# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
PE-aligned text encoder for SAM3.

Based on the PE paper (arXiv:2504.13181):
- Uses a causal transformer for text encoding
- Trained with contrastive vision-language alignment
- Produces embeddings aligned with PE vision encoder

This module provides a drop-in replacement for VETextEncoder that uses
PE's text representation approach for better vision-language understanding.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerBlock(nn.Module):
    """
    Single transformer block with causal (left-to-right) attention.

    PE uses a causal text encoder, meaning each token can only attend
    to previous tokens, similar to GPT-style language models.
    """

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with causal attention.

        Args:
            x: (B, L, D) input tokens
            attn_mask: (L, L) causal attention mask
            key_padding_mask: (B, L) padding mask

        Returns:
            (B, L, D) output tokens
        """
        if self.pre_norm:
            # Pre-norm: LN -> Attn -> Residual
            x2 = self.norm1(x)
            x2, _ = self.self_attn(
                x2, x2, x2,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout1(x2)

            # Pre-norm: LN -> FFN -> Residual
            x2 = self.norm2(x)
            x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
            x = x + self.dropout3(x2)
        else:
            # Post-norm: Attn -> Residual -> LN
            x2, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm1(x + self.dropout1(x2))

            # Post-norm: FFN -> Residual -> LN
            x2 = self.linear2(self.dropout2(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout3(x2))

        return x


class PETextEncoder(nn.Module):
    """
    PE-aligned text encoder for SAM3.

    This encoder is designed to produce text embeddings that are aligned
    with PE's vision encoder through contrastive pretraining. It can be
    used as a drop-in replacement for VETextEncoder.

    Architecture follows PE paper:
    - Causal transformer (24 layers by default)
    - 1024-dim width, 16 heads
    - Projects to 256-dim for SAM3 compatibility
    - Uses BPE tokenization (same as CLIP/VETextEncoder)

    Args:
        tokenizer: Tokenizer instance (SimpleTokenizer from SAM3)
        d_model: Output dimension for SAM3 compatibility (256)
        width: Transformer hidden dimension (1024)
        heads: Number of attention heads (16)
        layers: Number of transformer layers (24)
        max_seq_len: Maximum sequence length (32 for PE)
        dropout: Dropout rate
    """

    def __init__(
        self,
        tokenizer,
        d_model: int = 256,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.width = width
        self.layers = layers
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        vocab_size = getattr(tokenizer, "vocab_size", 49408)  # CLIP vocab size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(
            torch.zeros(max_seq_len, width)
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=width * 4,
                    dropout=dropout,
                    activation="gelu",
                    pre_norm=True,
                )
                for _ in range(layers)
            ]
        )

        # Final layer norm (pre-norm architecture)
        self.ln_final = nn.LayerNorm(width)

        # Projection to SAM3 dimension
        self.text_projection = nn.Linear(width, d_model, bias=False)

        # Initialize causal mask buffer
        self._init_causal_mask()

        # Initialize weights
        self._init_weights()

    def _init_causal_mask(self):
        """Initialize causal attention mask."""
        mask = torch.triu(
            torch.ones(self.max_seq_len, self.max_seq_len),
            diagonal=1,
        ).bool()
        self.register_buffer("causal_mask", mask)

    def _init_weights(self):
        """Initialize weights following PE/CLIP conventions."""
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Position embedding
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Projection
        nn.init.normal_(self.text_projection.weight, std=self.width ** -0.5)

    def encode_text(
        self,
        text: Union[str, List[str]],
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            text: Single string or list of strings
            device: Device to place tensors on

        Returns:
            (B, d_model) text embeddings
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize
        tokens = self.tokenizer(text)  # (B, L)
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        else:
            tokens = torch.tensor(tokens, device=device)

        # Pad/truncate to max_seq_len
        B, L = tokens.shape
        if L > self.max_seq_len:
            tokens = tokens[:, : self.max_seq_len]
            L = self.max_seq_len

        # Get embeddings
        x = self.token_embedding(tokens)  # (B, L, width)
        x = x + self.positional_embedding[:L]

        # Create masks
        causal_mask = self.causal_mask[:L, :L]
        padding_mask = tokens == 0  # Padding token

        # Apply transformer
        for block in self.transformer_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)

        # Final layer norm
        x = self.ln_final(x)

        # Pool: use EOS token (last non-padding token) following CLIP/PE
        # Find last valid token position for each sequence
        seq_lens = (~padding_mask).sum(dim=1) - 1  # (B,)
        seq_lens = seq_lens.clamp(min=0)

        # Gather EOS token embeddings
        pooled = x[torch.arange(B, device=device), seq_lens]  # (B, width)

        # Project to output dimension
        text_embeds = self.text_projection(pooled)  # (B, d_model)

        return text_embeds

    def forward(
        self,
        texts: List[str],
        input_boxes: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with VETextEncoder interface.

        This method maintains API compatibility with the existing
        VETextEncoder so it can be used as a drop-in replacement.

        Args:
            texts: List of text prompts
            input_boxes: Optional boxes (unused, for API compatibility)
            device: Device for inference

        Returns:
            text_mask: (B, L) attention mask
            text_memory: (L, B, d_model) encoded text features
            text_embeds: (B, d_model) pooled text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        tokens = self.tokenizer(texts)
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        else:
            tokens = torch.tensor(tokens, device=device)

        B, L = tokens.shape
        if L > self.max_seq_len:
            tokens = tokens[:, : self.max_seq_len]
            L = self.max_seq_len

        # Create attention mask
        text_mask = tokens != 0  # (B, L) - True for valid tokens

        # Get token embeddings
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding[:L]

        # Causal mask and padding mask
        causal_mask = self.causal_mask[:L, :L]
        padding_mask = ~text_mask

        # Transform
        for block in self.transformer_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)

        x = self.ln_final(x)

        # Project to output dimension
        text_memory = self.text_projection(x)  # (B, L, d_model)

        # Transpose for SAM3 compatibility: (B, L, D) -> (L, B, D)
        text_memory = text_memory.transpose(0, 1)

        # Pool for embeddings (EOS token)
        seq_lens = text_mask.sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        text_embeds = text_memory[seq_lens, torch.arange(B, device=device)]

        return text_mask, text_memory, text_embeds

    def encode_for_retrieval(
        self,
        texts: List[str],
        device: str = "cuda",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode texts for retrieval/similarity computation.

        Returns L2-normalized embeddings suitable for cosine similarity.

        Args:
            texts: List of text prompts
            device: Device for inference
            normalize: Whether to L2-normalize embeddings

        Returns:
            (B, d_model) normalized text embeddings
        """
        embeds = self.encode_text(texts, device=device)

        if normalize:
            embeds = F.normalize(embeds, p=2, dim=-1)

        return embeds


def create_pe_text_encoder(
    bpe_path: Optional[str] = None,
    d_model: int = 256,
    width: int = 1024,
    heads: int = 16,
    layers: int = 24,
    device: str = "cuda",
) -> PETextEncoder:
    """
    Factory function to create PE text encoder.

    Args:
        bpe_path: Path to BPE vocabulary file
        d_model: Output dimension (256 for SAM3)
        width: Transformer hidden dimension
        heads: Number of attention heads
        layers: Number of transformer layers
        device: Device to place model on

    Returns:
        Initialized PETextEncoder
    """
    from sam3.model.tokenizer_ve import SimpleTokenizer
    import pkg_resources

    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    tokenizer = SimpleTokenizer(bpe_path=bpe_path)

    encoder = PETextEncoder(
        tokenizer=tokenizer,
        d_model=d_model,
        width=width,
        heads=heads,
        layers=layers,
    )

    return encoder.to(device)
