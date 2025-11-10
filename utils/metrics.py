from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import psutil

from .types import QueryMetrics, SetupMetrics


def format_metrics_table(metrics: Sequence[SetupMetrics | QueryMetrics]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for metric in metrics:
        data = asdict(metric)
        # Expand nested dataclasses into dictionaries
        for key, value in list(data.items()):
            if hasattr(value, "__dict__"):
                data[key] = asdict(value)
        table.append(data)
    return table


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def directory_size_mb(paths: Iterable[Path]) -> float:
    total_bytes = 0
    for path in paths:
        if path.is_file():
            total_bytes += path.stat().st_size
        elif path.is_dir():
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    total_bytes += (Path(dirpath) / filename).stat().st_size
    return total_bytes / (1024 * 1024)


class ResourceMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self._peak_memory = self.process.memory_info().rss

    def snapshot_memory(self) -> float:
        rss = self.process.memory_info().rss
        self._peak_memory = max(self._peak_memory, rss)
        return rss / (1024 * 1024)

    def peak_memory_mb(self) -> float:
        self.snapshot_memory()
        return self._peak_memory / (1024 * 1024)


def wait_for_filesystem_flush(delay: float = 0.1) -> None:
    """Allow filesystem stats to settle."""
    time.sleep(delay)

