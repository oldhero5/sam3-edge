"""Embedding service for generating text embeddings using SAM3's VETextEncoder."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates text embeddings using SAM3's VETextEncoder.

    The VETextEncoder is a 24-layer transformer with CLIP-based tokenization
    that produces 256-dimensional embeddings for text prompts.
    """

    def __init__(
        self,
        bpe_path: Optional[str] = None,
        device: str = "cuda",
        embedding_dim: int = 256,
    ):
        """
        Initialize embedding service.

        Args:
            bpe_path: Path to BPE vocabulary file. If None, uses SAM3 default.
            device: Device for inference ("cuda" or "cpu")
            embedding_dim: Output embedding dimension (256 from VETextEncoder)
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self._encoder = None
        self._tokenizer = None
        self._bpe_path = bpe_path
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the text encoder."""
        if self._initialized:
            return

        try:
            from sam3.model.text_encoder_ve import VETextEncoder
            from sam3.model.tokenizer_ve import SimpleTokenizer
            import pkg_resources

            # Get BPE path
            if self._bpe_path is None:
                try:
                    self._bpe_path = pkg_resources.resource_filename(
                        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
                    )
                except Exception:
                    # Fallback path
                    sam3_path = Path(__file__).parent.parent.parent.parent.parent / "sam3"
                    self._bpe_path = str(sam3_path / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

            # Initialize tokenizer and encoder
            self._tokenizer = SimpleTokenizer(bpe_path=self._bpe_path)
            self._encoder = VETextEncoder(
                tokenizer=self._tokenizer,
                d_model=self.embedding_dim,
                width=1024,
                heads=16,
                layers=24,
            )
            self._encoder = self._encoder.to(self.device).eval()

            self._initialized = True
            logger.info(f"EmbeddingService initialized on {self.device}")

        except ImportError as e:
            logger.warning(f"SAM3 text encoder not available: {e}")
            logger.warning("Using fallback random embeddings for testing")
            self._initialized = True  # Allow operation with random embeddings

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text prompt to embedding.

        Args:
            text: Text prompt (e.g., "green traffic lights")

        Returns:
            256-dimensional embedding as numpy array
        """
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple text prompts to embeddings.

        Args:
            texts: List of text prompts

        Returns:
            Array of shape (N, 256) with embeddings
        """
        if not self._initialized:
            self.initialize()

        if self._encoder is None:
            # Fallback: random embeddings for testing
            logger.warning("Using random embeddings (encoder not available)")
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms

        with torch.inference_mode():
            # Encode texts
            text_mask, text_memory, text_embeds = self._encoder(
                texts, input_boxes=None, device=self.device
            )

            # text_memory shape: [seq_len, batch, 256]
            # Use mean pooling over sequence for fixed-size embedding
            embeddings = text_memory.mean(dim=0)  # [batch, 256]

            return embeddings.cpu().numpy()

    def similarity(
        self,
        query_embedding: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and targets.

        Args:
            query_embedding: Single embedding (256,)
            target_embeddings: Array of embeddings (N, 256)

        Returns:
            Array of similarity scores (N,)
        """
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        target_norms = target_embeddings / np.linalg.norm(
            target_embeddings, axis=1, keepdims=True
        )

        # Cosine similarity via dot product
        return np.dot(target_norms, query_norm)

    @property
    def is_available(self) -> bool:
        """Check if text encoder is available."""
        if not self._initialized:
            self.initialize()
        return self._encoder is not None


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        _embedding_service.initialize()
    return _embedding_service


def set_embedding_service(service: EmbeddingService) -> None:
    """Set global embedding service."""
    global _embedding_service
    _embedding_service = service
