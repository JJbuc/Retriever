from __future__ import annotations

import logging
import time
from typing import Dict, List

import chromadb
import numpy as np

from retrievers.base import BaseRetriever
from utils.embedding_utils import EmbeddingClient
from utils.gemini_helpers import GeminiService
from utils.metrics import ResourceMonitor
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


class VectorDBRetriever(BaseRetriever):
    def __init__(
        self,
        storage_root,
        embedding_client: EmbeddingClient,
        gemini_service: GeminiService,
        top_k: int = 5,
    ) -> None:
        super().__init__("vector_db", storage_root)
        self.embedding_client = embedding_client
        self.gemini_service = gemini_service
        self.top_k = top_k
        self._client = chromadb.PersistentClient(path=str(self.storage_root / "chroma"))
        self._collection_name = "benchmark_vectors"
        self._collection = None
        self._chunks: Dict[str, DocumentChunk] = {}

    def build(self, artifacts: CorpusArtifacts) -> SetupMetrics:
        monitor = ResourceMonitor()
        build_start = time.perf_counter()

        chunk_ids = [chunk.chunk_id for chunk in artifacts.chunks]
        texts = [chunk.text for chunk in artifacts.chunks]
        metadata = [chunk.metadata for chunk in artifacts.chunks]

        emb_start = time.perf_counter()
        embeddings = self.embedding_client.embed_texts(texts)
        emb_seconds = time.perf_counter() - emb_start

        index_start = time.perf_counter()
        if self._collection is not None:
            try:
                self._client.delete_collection(self._collection.name)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to delete existing Chroma collection %s: %s", self._collection.name, exc)
        else:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=metadata,
            documents=texts,
        )
        index_seconds = time.perf_counter() - index_start
        total_seconds = time.perf_counter() - build_start

        self._chunks = {chunk.chunk_id: chunk for chunk in artifacts.chunks}

        metrics = SetupMetrics(
            retriever_name=self.name,
            total_build_seconds=total_seconds,
            memory_peak_mb=monitor.peak_memory_mb(),
            storage_mb=0.0,  # updated later by benchmark runner
            breakdown=SetupBreakdown(
                total_seconds=total_seconds,
                chunking_seconds=artifacts.chunk_seconds,
                embedding_seconds=emb_seconds,
                indexing_seconds=index_seconds,
            ),
            extra={
                "num_chunks": len(chunk_ids),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
            },
        )
        return metrics

    def retrieve(self, query: str, query_id: int, **kwargs) -> tuple[RetrievalResult, QueryMetrics]:
        is_repeat = kwargs.get("is_repeat", False)
        retrieval_start = time.perf_counter()
        query_vec = self.embedding_client.embed_texts([query])[0]
        search = self._collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k,
        )
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        ids = search["ids"][0] if search["ids"] else []
        scores = search["distances"][0] if search["distances"] else []
        hits: List[RetrievalHit] = []
        combined_context_parts: List[str] = []
        for chunk_id, score in zip(ids, scores):
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue
            combined_context_parts.append(chunk.text)
            hits.append(RetrievalHit(chunk=chunk, score=float(score)))

        combined_context = "\n\n".join(combined_context_parts)

        generation = self.gemini_service.generate_answer(query, combined_context)
        evaluation = self.gemini_service.evaluate_relevance(query, combined_context, generation.text)

        end_to_end = retrieval_latency + generation.latency_seconds * 1000

        result = RetrievalResult(
            hits=hits,
            combined_context=combined_context,
            generator_output=generation.text,
            metadata={"query_vector": np.array(query_vec).tolist()},
        )

        metrics = QueryMetrics(
            retriever_name=self.name,
            query_id=query_id,
            query_text=query,
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

