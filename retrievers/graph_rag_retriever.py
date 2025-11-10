from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from tqdm.auto import tqdm
from neo4j import GraphDatabase

from retrievers.base import BaseRetriever
from utils.gemini_helpers import GeminiService, Triple
from utils.graph_utils import load_graph, save_graph
from utils.types import CorpusArtifacts, DocumentChunk, QueryMetrics, RetrievalHit, RetrievalResult, SetupBreakdown, SetupMetrics

LOGGER = logging.getLogger(__name__)


class GraphRAGRetriever(BaseRetriever):
    def __init__(
        self,
        storage_root,
        gemini_service: GeminiService,
        neo4j_uri: Optional[str],
        neo4j_user: Optional[str],
        neo4j_password: Optional[str],
        neo4j_database: Optional[str] = None,
        max_hops: int = 2,
        top_k: int = 8,
    ) -> None:
        super().__init__("graph_rag", storage_root)
        self.gemini_service = gemini_service
        self.max_hops = max_hops
        self.top_k = top_k
        self._graph = nx.MultiDiGraph()
        self._chunk_lookup: Dict[str, DocumentChunk] = {}
        self._graph_path = self.storage_root / "knowledge_graph.json"
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._neo4j_database = neo4j_database or "neo4j"

    def _build(self, artifacts: CorpusArtifacts) -> SetupMetrics:
        build_start = time.perf_counter()
        self._chunk_lookup = {chunk.chunk_id: chunk for chunk in artifacts.chunks}

        triple_records: List[Triple] = []
        extraction_tokens = 0
        graph_start = time.perf_counter()
        for chunk in tqdm(artifacts.chunks, desc="GraphRAG triple extraction", unit="chunk"):
            triples, metadata = self.gemini_service.extract_triples(chunk.chunk_id, chunk.text)
            usage_info = metadata.get("usage", {})
            if not isinstance(usage_info, dict):
                usage_info = {}
            extraction_tokens += usage_info.get("prompt_token_count", 0)
            extraction_tokens += usage_info.get("candidates_token_count", 0)
            for triple in triples:
                if not triple.subject or not triple.object:
                    continue
                triple_records.append(triple)
                self._graph.add_node(
                    triple.subject,
                    label="entity",
                )
                self._graph.add_node(
                    triple.object,
                    label="entity",
                )
                self._graph.add_edge(
                    triple.subject,
                    triple.object,
                    key=f"{triple.source_chunk_id}:{triple.relation}",
                    relation=triple.relation,
                    chunk_id=triple.source_chunk_id,
                    confidence=triple.confidence,
                    metadata=json.dumps(triple.metadata),
                )
            # Link chunk to entity nodes to preserve source context
            self._graph.add_node(chunk.chunk_id, label="chunk")
            for triple in triples:
                self._graph.add_edge(
                    triple.subject,
                    chunk.chunk_id,
                    relation="MENTIONED_IN",
                )
                self._graph.add_edge(
                    chunk.chunk_id,
                    triple.object,
                    relation="MENTIONS",
                )

        graph_seconds = time.perf_counter() - graph_start
        total_seconds = time.perf_counter() - build_start

        save_graph(self._graph, self._graph_path)

        if self._neo4j_uri and self._neo4j_user and self._neo4j_password:
            try:
                self._push_to_neo4j(triple_records)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Neo4j push failed: %s", exc)

        metrics = SetupMetrics(
            retriever_name=self.name,
            total_build_seconds=total_seconds,
            memory_peak_mb=0.0,
            storage_mb=self._graph_path.stat().st_size / (1024 * 1024) if self._graph_path.exists() else 0.0,
            breakdown=SetupBreakdown(
                total_seconds=total_seconds,
                chunking_seconds=artifacts.chunk_seconds,
                embedding_seconds=0.0,
                indexing_seconds=0.0,
                graph_seconds=graph_seconds,
            ),
            extra={
                "num_nodes": self._graph.number_of_nodes(),
                "num_edges": self._graph.number_of_edges(),
                "extraction_tokens": extraction_tokens,
            },
        )
        return metrics

    def _on_build_success(self, artifacts: CorpusArtifacts) -> None:
        self._save_chunk_cache(artifacts.chunks)

    def _load_cache(self) -> None:
        cached_chunks = self._load_chunk_cache()
        if cached_chunks:
            self._chunk_lookup = {chunk.chunk_id: chunk for chunk in cached_chunks}
        if self._graph_path.exists():
            try:
                self._graph = load_graph(self._graph_path)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to load cached knowledge graph: %s", exc)

    def _push_to_neo4j(self, triples: Iterable[Triple]) -> None:
        driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
        cypher = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATION {type: $relation, chunk_id: $chunk_id}]->(o)
        SET r.confidence = $confidence,
            r.metadata = $metadata
        """
        with driver.session(database=self._neo4j_database) as session:
            for triple in triples:
                session.run(
                    cypher,
                    subject=triple.subject,
                    relation=triple.relation,
                    object=triple.object,
                    chunk_id=triple.source_chunk_id,
                    confidence=triple.confidence,
                    metadata=json.dumps(triple.metadata),
                )
        driver.close()

    def _expand_entities(self, seeds: Set[str]) -> List[str]:
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque([(seed, 0) for seed in seeds])
        relevant_chunks: Set[str] = set()
        while queue:
            node, depth = queue.popleft()
            if node in visited or depth > self.max_hops:
                continue
            visited.add(node)
            for _, neighbor, edge_data in self._graph.out_edges(node, data=True):
                if edge_data.get("relation") == "HAS_CHUNK":
                    relevant_chunks.add(neighbor)
                else:
                    queue.append((neighbor, depth + 1))
            for neighbor, _, edge_data in self._graph.in_edges(node, data=True):
                if edge_data.get("relation") == "HAS_CHUNK":
                    relevant_chunks.add(neighbor)
                else:
                    queue.append((neighbor, depth + 1))
        return list(relevant_chunks)

    def _collect_context(self, chunk_ids: Iterable[str]) -> Tuple[List[RetrievalHit], str]:
        hits: List[RetrievalHit] = []
        context_parts: List[str] = []
        for chunk_id in chunk_ids:
            chunk = self._chunk_lookup.get(chunk_id)
            if not chunk:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=1.0))
            context_parts.append(chunk.text)
        combined_context = "\n\n".join(context_parts[: self.top_k])
        return hits, combined_context

    def retrieve(self, query: str, query_id: int, **kwargs) -> tuple[RetrievalResult, QueryMetrics]:
        is_repeat = kwargs.get("is_repeat", False)
        start = time.perf_counter()
        triples, metadata = self.gemini_service.extract_triples(f"query-{query_id}", query)
        seed_entities = {triple.subject for triple in triples if triple.subject} | {
            triple.object for triple in triples if triple.object
        }
        if not seed_entities and query:
            seed_entities = {query}
        chunk_candidates = self._expand_entities(seed_entities)
        hits, combined_context = self._collect_context(chunk_candidates)
        retrieval_latency_ms = (time.perf_counter() - start) * 1000

        generation = self.gemini_service.generate_answer(query, combined_context)
        evaluation = self.gemini_service.evaluate_relevance(query, combined_context, generation.text)

        end_to_end_ms = retrieval_latency_ms + generation.latency_seconds * 1000
        tokens_consumed = (
            generation.prompt_tokens
            + generation.completion_tokens
            + evaluation.prompt_tokens
            + evaluation.completion_tokens
            + metadata.get("usage", {}).get("prompt_tokens", 0)
            + metadata.get("usage", {}).get("completion_tokens", 0)
        )

        result = RetrievalResult(
            hits=hits,
            combined_context=combined_context,
            generator_output=generation.text,
            metadata={
                "seed_entities": list(seed_entities),
                "num_candidate_chunks": len(chunk_candidates),
            },
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
                "num_seed_entities": len(seed_entities),
                "candidate_chunks": len(chunk_candidates),
                "generation_latency_ms": generation.latency_seconds * 1000,
                "is_repeat": is_repeat,
            },
        )
        return result, metrics

