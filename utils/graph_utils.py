from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx

LOGGER = logging.getLogger(__name__)


@dataclass
class GraphArtifacts:
    graph: nx.MultiDiGraph
    triples: List[Dict[str, str]]
    build_seconds: float


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_graph(path: Path) -> nx.MultiDiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True, multigraph=True)


def graph_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)

