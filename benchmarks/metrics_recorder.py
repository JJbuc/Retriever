from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from utils.metrics import directory_size_mb
from utils.types import QueryMetrics, SetupMetrics


class MetricsRecorder:
    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.setup_rows: List[Dict[str, object]] = []
        self.query_rows: List[Dict[str, object]] = []
        self.update_rows: List[Dict[str, object]] = []

    def add_setup(self, metrics: SetupMetrics) -> None:
        row = metrics.__dict__.copy()
        breakdown = row.pop("breakdown")
        if breakdown:
            for key, value in breakdown.__dict__.items():
                row[f"breakdown_{key}"] = value
        extra = row.pop("extra", {})
        for key, value in extra.items():
            row[f"extra_{key}"] = value
        self.setup_rows.append(row)

    def add_query(self, metrics: QueryMetrics) -> None:
        row = metrics.__dict__.copy()
        extra = row.pop("extra", {})
        for key, value in extra.items():
            row[f"extra_{key}"] = value
        self.query_rows.append(row)

    def add_update(self, retriever_name: str, timings: Dict[str, float], storage_paths: Sequence[Path]) -> None:
        row: Dict[str, object] = {"retriever_name": retriever_name}
        row.update({f"time_{k}": v for k, v in timings.items()})
        row["storage_mb"] = directory_size_mb(storage_paths)
        self.update_rows.append(row)

    def write_csv(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.setup_rows:
            pd.DataFrame(self.setup_rows).to_csv(self.reports_dir / "setup_metrics.csv", index=False)
        if self.query_rows:
            pd.DataFrame(self.query_rows).to_csv(self.reports_dir / "query_metrics.csv", index=False)
        if self.update_rows:
            pd.DataFrame(self.update_rows).to_csv(self.reports_dir / "update_metrics.csv", index=False)

