from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from papersearch.config import Settings, get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any = None


def _load_model(settings: Settings) -> Any:
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s …", settings.embedding_model)
            _model = SentenceTransformer(
                settings.embedding_model,
                trust_remote_code=False,
            )
        return _model


def embed_texts(texts: list[str], settings: Settings | None = None) -> np.ndarray:
    """Return float32 matrix (n, dim) L2-normalized for cosine-friendly geometry."""
    cfg = settings or get_settings()
    model = _load_model(cfg)
    vectors = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32, copy=False)
    if vectors.shape[1] != cfg.embedding_dimension:
        raise ValueError(
            f"Model output dim {vectors.shape[1]} != configured {cfg.embedding_dimension}"
        )
    return vectors


def embed_query(text: str, settings: Settings | None = None) -> np.ndarray:
    return embed_texts([text], settings=settings)[0]
