import numpy as np
from typing import List

from dasa.config import DASAConfig


class EmbeddingEngine:
    """
    Generates dense vector embeddings using sentence-transformers.

    Designed for hardware-constrained environments:
    - Runs entirely on CPU (no CUDA required).
    - Model is loaded lazily on first use.
    - Normalized embeddings allow cosine similarity via a simple dot product.

    Typical RAM usage with 'all-MiniLM-L6-v2': ~200 MB model + data.
    """

    def __init__(self, config: DASAConfig) -> None:
        self.config = config
        self._model = None

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string to a normalized embedding vector."""
        self._ensure_loaded()
        return self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings in a single batched forward pass.

        Returns an array of shape (len(texts), embedding_dim).
        All vectors are L2-normalized so cosine similarity equals dot product.
        """
        self._ensure_loaded()
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

    @staticmethod
    def cosine_similarity_batch(
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between one query vector and N corpus vectors.

        Because vectors are pre-normalized, this is a simple matrix-vector dot
        product — no division needed.  Runs in O(N·dim) with NumPy BLAS.

        Args:
            query_vec:   Shape (dim,).
            corpus_vecs: Shape (N, dim).

        Returns:
            Similarity scores, shape (N,), in range [-1, 1].
        """
        return np.dot(corpus_vecs, query_vec)

    # ── Private ────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(
            self.config.embedding_model,
            device=self.config.device,
        )
