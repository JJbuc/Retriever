from __future__ import annotations

import logging
import time
from typing import Dict, List

import chromadb
import numpy as np

from retrievers.base import BaseRetriever
from utils.embedding_utils import EmbeddingClient
from utils.gemini_helpers import GeminiService
from utils.metrics import directory_size_mb, wait_for_filesystem_flush
from utils.redis_utils import ExactMatchCache, RedisManager
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


class KVCacheRetriever(BaseRetriever):
    def __init__(
        self,
        storage_root,
        embedding_client: EmbeddingClient,
        gemini_service: GeminiService,
        redis_manager: RedisManager,
        ttl_seconds: int,
        top_k: int = 5,
    ) -> None:
        super().__init__("kv_cache", storage_root)
        self.embedding_client = embedding_client
        self.gemini_service = gemini_service
        self.redis_cache = ExactMatchCache(redis_manager, ttl_seconds=ttl_seconds)
        self.top_k = top_k
        self._vector_client = chromadb.PersistentClient(path=str(self.storage_root / "chroma"))
        self._collection_name = "kv_cache_vectors"
        self._collection = None
        self._chunks: Dict[str, DocumentChunk] = {}

    def _build(self, artifacts: CorpusArtifacts) -> SetupMetrics:
        build_start = time.perf_counter()
        texts = [chunk.text for chunk in artifacts.chunks]
        chunk_ids = [chunk.chunk_id for chunk in artifacts.chunks]
        metadata = [chunk.metadata for chunk in artifacts.chunks]

        emb_start = time.perf_counter()
        embeddings = self.embedding_client.embed_texts(texts)
        emb_seconds = time.perf_counter() - emb_start

        if self._collection is not None:
            try:
                self._vector_client.delete_collection(self._collection.name)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to delete existing kv cache collection %s: %s", self._collection.name, exc)
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
        self.redis_cache.clear()

        wait_for_filesystem_flush()
        size_mb = directory_size_mb(self.storage_paths())
        metrics = SetupMetrics(
            retriever_name=self.name,
            total_build_seconds=total_seconds,
            memory_peak_mb=0.0,
            storage_mb=size_mb,
            breakdown=SetupBreakdown(
                total_seconds=total_seconds,
                chunking_seconds=artifacts.chunk_seconds,
                embedding_seconds=emb_seconds,
                indexing_seconds=index_seconds,
            ),
            extra={
                "num_chunks": len(chunk_ids),
                "cache_cleared": True,
            },
        )
        return metrics

    def _on_build_success(self, artifacts: CorpusArtifacts) -> None:
        self._save_chunk_cache(artifacts.chunks)

    def _load_cache(self) -> None:
        cached_chunks = self._load_chunk_cache()
        if cached_chunks:
            self._chunks = {chunk.chunk_id: chunk for chunk in cached_chunks}
        else:
            self._chunks = {}
        self._collection = self._vector_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
        is_repeat = kwargs.get("is_repeat", False)
        cache_start = time.perf_counter()
        cached = self.redis_cache.get(query)
        cache_latency_ms = (time.perf_counter() - cache_start) * 1000
        if cached:
            hits = []
            for chunk_id in cached.get("chunk_ids", []):
                chunk = self._chunks.get(chunk_id)
                if chunk:
                    hits.append(RetrievalHit(chunk=chunk, score=1.0))
            result = RetrievalResult(
                hits=hits,
                combined_context=cached["context"],
                generator_output=cached["answer"],
                metadata={"cache_hit": True},
            )
            self.redis_cache.stats.record_exact_match(True)
            metrics = QueryMetrics(
                retriever_name=self.name,
                query_id=query_id,
                query_text=query,
                answer_text=cached.get("answer"),
                retrieval_latency_ms=cache_latency_ms,
                end_to_end_latency_ms=cache_latency_ms,
                context_relevance=cached.get("context_relevance", 0.0),
                answer_relevance=cached.get("answer_relevance", 0.0),
                tokens_consumed=0,
                extra={
                    "cache_hit": True,
                    "cache_latency_ms": cache_latency_ms,
                    "ttl_seconds_remaining": cached.get("ttl"),
                    "hit_rate": self.redis_cache.stats.hit_rate,
                    "is_repeat": is_repeat,
                },
            )
            return result, metrics

        retrieval_start = time.perf_counter()
        query_vector = self.embedding_client.embed_texts([query])[0]
        hits = self._vector_search(query_vector)
        combined_context = "\n\n".join(hit.chunk.text for hit in hits)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000 + cache_latency_ms

        generation = self.gemini_service.generate_answer(query, combined_context)
        evaluation = self.gemini_service.evaluate_relevance(query, combined_context, generation.text)

        payload = {
            "answer": generation.text,
            "context": combined_context,
            "chunk_ids": [hit.chunk.chunk_id for hit in hits],
            "context_relevance": evaluation.context_relevance,
            "answer_relevance": evaluation.answer_relevance,
            "tokens": {
                "prompt": generation.prompt_tokens,
                "completion": generation.completion_tokens,
                "judge_prompt": evaluation.prompt_tokens,
                "judge_completion": evaluation.completion_tokens,
            },
            "timestamp": time.time(),
        }
        self.redis_cache.set(query, payload)

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
            answer_text=generation.text,
            retrieval_latency_ms=retrieval_latency_ms,
            end_to_end_latency_ms=end_to_end_ms,
            context_relevance=evaluation.context_relevance,
            answer_relevance=evaluation.answer_relevance,
            tokens_consumed=tokens_consumed,
            extra={
                "cache_hit": False,
                "cache_latency_ms": cache_latency_ms,
                "generation_latency_ms": generation.latency_seconds * 1000,
                "hit_rate": self.redis_cache.stats.hit_rate,
                "is_repeat": is_repeat,
            },
        )
        return result, metrics

