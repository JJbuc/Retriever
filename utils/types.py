from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    source_path: Path
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusArtifacts:
    documents: List[Document]
    chunks: List[DocumentChunk]
    load_seconds: float
    chunk_seconds: float


@dataclass
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    location: Optional[str] = None


@dataclass
class RetrievalResult:
    hits: List[RetrievalHit]
    combined_context: str
    generator_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetupBreakdown:
    total_seconds: float
    chunking_seconds: float
    embedding_seconds: float
    indexing_seconds: float
    graph_seconds: float = 0.0


@dataclass
class SetupMetrics:
    retriever_name: str
    total_build_seconds: float
    memory_peak_mb: float
    storage_mb: float
    breakdown: SetupBreakdown
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetrics:
    retriever_name: str
    query_id: int
    query_text: str
    retrieval_latency_ms: float
    end_to_end_latency_ms: float
    context_relevance: float
    answer_relevance: float
    tokens_consumed: int
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


