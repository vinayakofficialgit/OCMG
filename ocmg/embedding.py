"""Embedding backends.

Two are provided. The hashing embedder has no dependencies and lets the whole
pipeline run anywhere, which matters for reproducibility and for CI. The
sentence-transformers backend is the one used for reported numbers.

Whichever is selected is held constant across every experimental condition, so
that measured differences are attributable to governance rather than to
representation quality (Section 7.2).
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Deterministic bag-of-n-grams projected into a fixed-width vector.

    Not competitive with a trained encoder, but it is stable, fast and free of
    downloads, which makes it the right default for pipeline validation.
    """

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self.name = f"hashing-{dim}"

    def _hash(self, token: str) -> int:
        h = hashlib.blake2b(
            token.encode("utf8"), digest_size=8, key=str(self.seed).encode()
        )
        return int.from_bytes(h.digest(), "little") % self.dim

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        if not toks:
            return vec
        for t in toks:
            vec[self._hash(t)] += 1.0
        # Character trigrams add robustness to morphology and typos.
        squeezed = " ".join(toks)
        for i in range(len(squeezed) - 2):
            vec[self._hash("###" + squeezed[i : i + 3])] += 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.encode_one(t) for t in texts])


class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers encoder (bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def build_embedder(kind: str = "auto", dim: int = 256, st_model: str = "BAAI/bge-small-en-v1.5", seed: int = 0):
    """Return the requested embedder, falling back to hashing if unavailable."""
    if kind in ("auto", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder(st_model)
        except Exception:
            if kind != "auto":
                raise
    return HashingEmbedder(dim=dim, seed=seed)


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against a stack of vectors."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    qn = np.linalg.norm(query)
    if qn == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    mn = np.linalg.norm(matrix, axis=1)
    mn[mn == 0] = 1e-9
    return (matrix @ query) / (mn * qn)
