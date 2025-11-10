from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List

from utils.metrics import ResourceMonitor, directory_size_mb, wait_for_filesystem_flush
from utils.types import CorpusArtifacts, DocumentChunk, QueryMetrics, RetrievalResult, SetupBreakdown, SetupMetrics

LOGGER = logging.getLogger(__name__)


class BaseRetriever(ABC):
    def __init__(self, name: str, storage_root: Path) -> None:
        self._name = name
        self.storage_root = storage_root / name
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.storage_root / "metadata.json"
        self._chunk_cache_path = self.storage_root / "chunks.json"
        self._current_dataset_hash: str | None = None

    @property
    def name(self) -> str:
        return self._name

    def build(self, artifacts: CorpusArtifacts, dataset_hash: str) -> SetupMetrics:
        self._current_dataset_hash = dataset_hash
        if self._is_cache_valid(dataset_hash):
            LOGGER.info("Reusing cached %s artifacts.", self.name)
            self._load_cache()
            self._save_metadata(dataset_hash)
            return self._skip_metrics(artifacts)
        metrics = self._build(artifacts)
        self._on_build_success(artifacts)
        self._save_metadata(dataset_hash)
        return metrics

    @abstractmethod
    def _build(self, artifacts: CorpusArtifacts) -> SetupMetrics:
        """Subclass implementation that performs the actual build process."""

    @abstractmethod
    def retrieve(self, query: str, query_id: int, **kwargs) -> tuple[RetrievalResult, QueryMetrics]:
        """Execute retrieval and return results alongside metrics."""

    def _load_cache(self) -> None:
        """Hook for subclasses to hydrate in-memory state from disk."""
        return None

    def _on_build_success(self, artifacts: CorpusArtifacts) -> None:
        """Hook for subclasses to persist additional metadata."""
        return None

    def _measure_setup(
        self,
        artefacts: CorpusArtifacts,
        build_callable,
        *,
        additional_paths: Iterable[Path] | None = None,
    ) -> SetupMetrics:
        monitor = ResourceMonitor()
        result: SetupMetrics = build_callable(monitor)
        wait_for_filesystem_flush()
        storage_paths = list(additional_paths or []) + [self.storage_root]
        size_mb = directory_size_mb(storage_paths)

        breakdown = SetupBreakdown(
            total_seconds=result.breakdown.total_seconds,
            chunking_seconds=artefacts.chunk_seconds,
            embedding_seconds=result.breakdown.embedding_seconds,
            indexing_seconds=result.breakdown.indexing_seconds,
            graph_seconds=result.breakdown.graph_seconds,
        )
        metrics = SetupMetrics(
            retriever_name=self.name,
            total_build_seconds=result.total_build_seconds,
            memory_peak_mb=monitor.peak_memory_mb(),
            storage_mb=size_mb,
            breakdown=breakdown,
            extra=result.extra,
        )
        return metrics

    def _skip_metrics(self, artefacts: CorpusArtifacts) -> SetupMetrics:
        wait_for_filesystem_flush()
        size_mb = directory_size_mb(self.storage_paths())
        breakdown = SetupBreakdown(
            total_seconds=0.0,
            chunking_seconds=artefacts.chunk_seconds,
            embedding_seconds=0.0,
            indexing_seconds=0.0,
            graph_seconds=0.0,
        )
        return SetupMetrics(
            retriever_name=self.name,
            total_build_seconds=0.0,
            memory_peak_mb=0.0,
            storage_mb=size_mb,
            breakdown=breakdown,
            extra={"cache_hit": True},
        )

    def _is_cache_valid(self, dataset_hash: str) -> bool:
        metadata = self._load_metadata()
        return metadata.get("dataset_hash") == dataset_hash

    def _load_metadata(self) -> dict:
        if not self._metadata_path.exists():
            return {}
        try:
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_metadata(self, dataset_hash: str) -> None:
        metadata = {"dataset_hash": dataset_hash, "name": self.name}
        self._metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _save_chunk_cache(self, chunks: Iterable[DocumentChunk]) -> None:
        data = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        self._chunk_cache_path.write_text(json.dumps(data), encoding="utf-8")

    def _load_chunk_cache(self) -> List[DocumentChunk]:
        if not self._chunk_cache_path.exists():
            return []
        raw = json.loads(self._chunk_cache_path.read_text(encoding="utf-8"))
        return [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in raw
        ]

    def storage_paths(self) -> List[Path]:
        return [self.storage_root]

