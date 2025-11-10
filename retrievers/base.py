from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List

from utils.metrics import ResourceMonitor, directory_size_mb, wait_for_filesystem_flush
from utils.types import CorpusArtifacts, QueryMetrics, RetrievalResult, SetupBreakdown, SetupMetrics

LOGGER = logging.getLogger(__name__)


class BaseRetriever(ABC):
    def __init__(self, name: str, storage_root: Path) -> None:
        self._name = name
        self.storage_root = storage_root / name
        self.storage_root.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def build(self, artifacts: CorpusArtifacts) -> SetupMetrics:
        """Build indexes/artifacts needed for retrieval."""

    @abstractmethod
    def retrieve(self, query: str, query_id: int, **kwargs) -> tuple[RetrievalResult, QueryMetrics]:
        """Execute retrieval and return results alongside metrics."""

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

    def storage_paths(self) -> List[Path]:
        return [self.storage_root]

