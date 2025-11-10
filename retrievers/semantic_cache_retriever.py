from __future__ import annotations

import logging
import time
from typing import Dict, List

import chromadb
import numpy as np

from retrievers.base import BaseRetriever
from utils.embedding_utils import EmbeddingClient
from utils.gemini_helpers import GeminiService
from utils.redis_utils import RedisManager, SemanticCache, SemanticCacheConfig
from utils.types import (
    CorpusArtifacts,
    DocumentChunk,
    QueryMetrics,
    RetrievalHit,
    RetrievalResult,
    SetupBreakdown,
    SetupMetrics,
)

LOGGER = logging.getLogger(__name__)


class SemanticCacheRetriever(BaseRetriever):
    def __init__(
        self,
        storage_root,
        embedding_client: EmbeddingClient,
        gemini_service: GeminiService,
        redis_manager: RedisManager,
        index_name: str,
        distance_threshold: float,
        top_k: int = 5,
    ) -> None:
        super().__init__("semantic_cache", storage_root)
        self.embedding_client = embedding_client
        self.gemini_service = gemini_service
        self.redis_manager = redis_manager
        self.index_name = index_name
        self.distance_threshold = distance_threshold
        self.top_k = top_k
        self._vector_client = chromadb.PersistentClient(path=str(self.storage_root / "chroma"))
        self._collection_name = "semantic_cache_vectors"
        self._collection = None
        self._chunks: Dict[str, DocumentChunk] = {}
        self._semantic_cache: SemanticCache | None = None
        self._vector_size: int = 0

    def build(self, artifacts: CorpusArtifacts) -> SetupMetrics:
        build_start = time.perf_counter()
        texts = [chunk.text for chunk in artifacts.chunks]
        chunk_ids = [chunk.chunk_id for chunk in artifacts.chunks]
        metadata = [chunk.metadata for chunk in artifacts.chunks]

        emb_start = time.perf_counter()
        embeddings = self.embedding_client.embed_texts(texts)
        emb_seconds = time.perf_counter() - emb_start
        self._vector_size = len(embeddings[0]) if embeddings else 0

        if self._collection is not None:
            try:
                self._vector_client.delete_collection(self._collection.name)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to delete existing semantic cache collection %s: %s", self._collection.name, exc)
        else:
            try:
                self._vector_client.delete_collection(self._collection_name)
            except Exception:
                pass
        self._collection = self._vector_client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        index_start = time.perf_counter()
        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
        )
        index_seconds = time.perf_counter() - index_start
        total_seconds = time.perf_counter() - build_start

        self._chunks = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
        if self._vector_size == 0:
            raise RuntimeError("Semantic cache retriever requires non-zero embedding dimension")
        self._semantic_cache = SemanticCache(
            self.redis_manager,
            SemanticCacheConfig(
                index_name=self.index_name,
                distance_threshold=self.distance_threshold,
                vector_size=self._vector_size,
            ),
        )

        metrics = SetupMetrics(
            retriever_name=self.name,
            total_build_seconds=total_seconds,
            memory_peak_mb=0.0,
            storage_mb=0.0,
            breakdown=SetupBreakdown(
                total_seconds=total_seconds,
                chunking_seconds=artifacts.chunk_seconds,
                embedding_seconds=emb_seconds,
                indexing_seconds=index_seconds,
            ),
            extra={
                "num_chunks": len(chunk_ids),
                "embedding_dim": self._vector_size,
            },
        )
        return metrics

    def _vector_search(self, query_vector: List[float]) -> List[RetrievalHit]:
        search = self._collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
        )
        ids = search["ids"][0] if search["ids"] else []
        scores = search["distances"][0] if search["distances"] else []
        hits: List[RetrievalHit] = []
        for chunk_id, score in zip(ids, scores):
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=float(score)))
        return hits

    def retrieve(self, query: str, query_id: int, **kwargs) -> tuple[RetrievalResult, QueryMetrics]:
        if self._semantic_cache is None:
            raise RuntimeError("Semantic cache not initialized")

        embed_start = time.perf_counter()
        query_vector = self.embedding_client.embed_texts([query])[0]
        embed_latency_ms = (time.perf_counter() - embed_start) * 1000
        is_repeat = kwargs.get("is_repeat", False)

        cache_start = time.perf_counter()
        cached = self._semantic_cache.get(query_vector)
        cache_latency_ms = (time.perf_counter() - cache_start) * 1000
        if cached:
            hits = []
            for chunk_id in cached["metadata"].get("chunk_ids", []):
                chunk = self._chunks.get(chunk_id)
                if chunk:
                    hits.append(RetrievalHit(chunk=chunk, score=1.0 - cached.get("distance", 0.0)))

            result = RetrievalResult(
                hits=hits,
                combined_context=cached["metadata"].get("context", ""),
                generator_output=cached.get("answer"),
                metadata={
                    "cache_hit": True,
                    "distance": cached.get("distance"),
                },
            )

            metrics = QueryMetrics(
                retriever_name=self.name,
                query_id=query_id,
                query_text=query,
                retrieval_latency_ms=embed_latency_ms + cache_latency_ms,
                end_to_end_latency_ms=embed_latency_ms + cache_latency_ms,
                context_relevance=cached["metadata"].get("context_relevance", 0.0),
                answer_relevance=cached["metadata"].get("answer_relevance", 0.0),
                tokens_consumed=0,
                extra={
                    "cache_hit": True,
                    "cache_latency_ms": cache_latency_ms,
                    "distance": cached.get("distance"),
                    "is_repeat": is_repeat,
                },
            )
            return result, metrics

        retrieval_start = time.perf_counter()
        hits = self._vector_search(query_vector)
        retrieval_latency_ms = embed_latency_ms + (time.perf_counter() - retrieval_start) * 1000
        combined_context = "\n\n".join(hit.chunk.text for hit in hits)

        generation = self.gemini_service.generate_answer(query, combined_context)
        evaluation = self.gemini_service.evaluate_relevance(query, combined_context, generation.text)

        metadata = {
            "context": combined_context,
            "chunk_ids": [hit.chunk.chunk_id for hit in hits],
            "context_relevance": evaluation.context_relevance,
            "answer_relevance": evaluation.answer_relevance,
        }
        self._semantic_cache.set(query, query_vector, generation.text, metadata)

        result = RetrievalResult(
            hits=hits,
            combined_context=combined_context,
            generator_output=generation.text,
            metadata={"cache_hit": False, "query_vector": np.array(query_vector).tolist()},
        )

        end_to_end_ms = retrieval_latency_ms + generation.latency_seconds * 1000
        tokens_consumed = (
            generation.prompt_tokens
            + generation.completion_tokens
            + evaluation.prompt_tokens
            + evaluation.completion_tokens
        )

        metrics = QueryMetrics(
            retriever_name=self.name,
            query_id=query_id,
            query_text=query,
            retrieval_latency_ms=retrieval_latency_ms,
            end_to_end_latency_ms=end_to_end_ms,
            context_relevance=evaluation.context_relevance,
            answer_relevance=evaluation.answer_relevance,
            tokens_consumed=tokens_consumed,
            extra={
                "cache_hit": False,
                "cache_latency_ms": cache_latency_ms,
                "generation_latency_ms": generation.latency_seconds * 1000,
                "is_repeat": is_repeat,
            },
        )
        return result, metrics

