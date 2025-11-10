from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    provider: str = "gemini"
    model_name: str = "models/gemini-embedding-001"
    max_batch_size: int = 32
    local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingClient:
    def __init__(self, api_key: Optional[str], config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.api_key = api_key
        self._local_model = None
        self._gemini_model: Optional[str] = None

        if self.config.model_name.lower() == "local":
            self.config.provider = "sentence-transformers"
            self.config.model_name = self.config.local_model_name

        if self.config.provider == "sentence-transformers":
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. Install with `pip install sentence-transformers`."
                )
            self._local_model = SentenceTransformer(self.config.local_model_name)
        elif self.config.provider == "gemini":
            if genai is None:
                raise RuntimeError(
                    "google-generativeai is required for Gemini embeddings. Install with `pip install google-generativeai`."
                )
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings.")
            genai.configure(api_key=api_key)
            self._gemini_model = self.config.model_name or "models/gemini-embedding-001"
        else:
            raise ValueError(f"Unsupported embedding provider {self.config.provider}")

    @retry(wait=wait_exponential_jitter(initial=1, max=10), stop=stop_after_attempt(3))
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        batched_vectors: List[List[float]] = []
        batch: List[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= self.config.max_batch_size:
                batched_vectors.extend(self._embed_batch(batch))
                batch = []
        if batch:
            batched_vectors.extend(self._embed_batch(batch))
        return batched_vectors

    def _ensure_local_model(self) -> None:
        if self._local_model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. Install with `pip install sentence-transformers`."
                )
            LOGGER.warning("Falling back to local sentence-transformers embeddings (%s).", self.config.local_model_name)
            self._local_model = SentenceTransformer(self.config.local_model_name)
        self.config.provider = "sentence-transformers"

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        if self.config.provider == "gemini":
            if genai is None or self._gemini_model is None:
                raise RuntimeError("Gemini embedding provider not properly initialized.")
            embeddings: List[List[float]] = []
            try:
                for text in batch:
                    result = genai.embed_content(model=self._gemini_model, content=text)
                    embeddings.append(result["embedding"])
                return embeddings
            except Exception as exc:  # pragma: no cover - remote fallback
                LOGGER.error("Gemini embedding request failed (%s). Switching to local embeddings.", exc)
                self._ensure_local_model()

        if self.config.provider != "sentence-transformers":
            self._ensure_local_model()

        assert self._local_model is not None
        vectors = self._local_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        if isinstance(vectors, np.ndarray):
            return vectors.tolist()
        return [vec.tolist() for vec in vectors]

