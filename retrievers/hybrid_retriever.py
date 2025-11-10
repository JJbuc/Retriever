from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from retrievers.base import BaseRetriever
from utils.embedding_utils import EmbeddingClient
from utils.gemini_helpers import GeminiService
from utils.metrics import ResourceMonitor, directory_size_mb, wait_for_filesystem_flush
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


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        storage_root,
        embedding_client: EmbeddingClient,
        gemini_service: GeminiService,
        top_k: int = 5,
        fusion_k: int = 10,
        k_fusion: float = 60.0,
    ) -> None:
        super().__init__("hybrid", storage_root)
        self.embedding_client = embedding_client
        self.gemini_service = gemini_service
        self.top_k = top_k
        self.fusion_k = fusion_k
        self.k_fusion = k_fusion
        self._vector_client = chromadb.PersistentClient(path=str(self.storage_root / "chroma"))
        self._collection_name = "hybrid_vectors"
        self._collection = None
        self._bm25: BM25Okapi | None = None
        self._chunks: Dict[str, DocumentChunk] = {}

    def _build(self, artifacts: CorpusArtifacts) -> SetupMetrics:
        monitor = ResourceMonitor()
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
                LOGGER.warning("Failed to delete existing hybrid collection %s: %s", self._collection.name, exc)
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

        bm25_start = time.perf_counter()
        tokenized_corpus = [text.split(" ") for text in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        bm25_seconds = time.perf_counter() - bm25_start
        total_seconds = time.perf_counter() - build_start

        self._chunks = {chunk.chunk_id: chunk for chunk in artifacts.chunks}

        wait_for_filesystem_flush()
        size_mb = directory_size_mb(self.storage_paths())
        metrics = SetupMetrics(
            retriever_name=self.name,
            total_build_seconds=total_seconds,
            memory_peak_mb=monitor.peak_memory_mb(),
            storage_mb=size_mb,
            breakdown=SetupBreakdown(
                total_seconds=total_seconds,
                chunking_seconds=artifacts.chunk_seconds,
                embedding_seconds=emb_seconds,
                indexing_seconds=index_seconds + bm25_seconds,
            ),
            extra={
                "num_chunks": len(chunk_ids),
                "bm25_seconds": bm25_seconds,
            },
        )
        return metrics

    def _on_build_success(self, artifacts: CorpusArtifacts) -> None:
        self._save_chunk_cache(artifacts.chunks)

    def _load_cache(self) -> None:
        cached_chunks = self._load_chunk_cache()
        if cached_chunks:
            self._chunks = {chunk.chunk_id: chunk for chunk in cached_chunks}
            tokenized_corpus = [chunk.text.split(" ") for chunk in cached_chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._chunks = {}
            self._bm25 = None
        self._collection = self._vector_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _bm25_scores(self, query: str) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Hybrid retriever not built")
        scores = self._bm25.get_scores(query.split())
        ranked = sorted(zip(self._chunks.keys(), scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.fusion_k]

    def retrieve(self, query: str, query_id: int, **kwargs) -> tuple[RetrievalResult, QueryMetrics]:
        is_repeat = kwargs.get("is_repeat", False)
        start = time.perf_counter()
        vector = self.embedding_client.embed_texts([query])[0]
        vector_results = self._collection.query(
            query_embeddings=[vector],
            n_results=self.fusion_k,
        )
        vector_ids = vector_results["ids"][0] if vector_results["ids"] else []
        vector_scores = vector_results["distances"][0] if vector_results["distances"] else []

        bm25_ranked = self._bm25_scores(query)

        fused_scores: Dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(bm25_ranked):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.k_fusion + rank + 1)
        for rank, (chunk_id, score) in enumerate(zip(vector_ids, vector_scores)):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.k_fusion + rank + 1)

        sorted_hits = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        retrieval_latency = (time.perf_counter() - start) * 1000

        hits: List[RetrievalHit] = []
        combined_context_parts: List[str] = []
        for chunk_id, fused_score in sorted_hits:
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=fused_score))
            combined_context_parts.append(chunk.text)

        combined_context = "\n\n".join(combined_context_parts)

        generation = self.gemini_service.generate_answer(query, combined_context)
        evaluation = self.gemini_service.evaluate_relevance(query, combined_context, generation.text)

        end_to_end = retrieval_latency + generation.latency_seconds * 1000

        result = RetrievalResult(
            hits=hits,
            combined_context=combined_context,
            generator_output=generation.text,
        )

        metrics = QueryMetrics(
            retriever_name=self.name,
            query_id=query_id,
            query_text=query,
            answer_text=generation.text,
            retrieval_latency_ms=retrieval_latency,
            end_to_end_latency_ms=end_to_end,
            context_relevance=evaluation.context_relevance,
            answer_relevance=evaluation.answer_relevance,
            tokens_consumed=(
                generation.prompt_tokens
                + generation.completion_tokens
                + evaluation.prompt_tokens
                + evaluation.completion_tokens
            ),
            extra={
                "generation_latency_ms": generation.latency_seconds * 1000,
                "evaluation_rationale": evaluation.rationale,
                "is_repeat": is_repeat,
            },
        )
        return result, metrics

