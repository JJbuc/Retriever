from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

LOGGER = logging.getLogger(__name__)


class RedisManager:
    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self.url, decode_responses=True)
        return self._client


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    exact_match_success: int = 0
    exact_match_total: int = 0

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def record_exact_match(self, success: bool) -> None:
        self.exact_match_total += 1
        if success:
            self.exact_match_success += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def exact_match_accuracy(self) -> float:
        total = self.exact_match_total
        return self.exact_match_success / total if total else 0.0


class ExactMatchCache:
    def __init__(self, manager: RedisManager, ttl_seconds: int) -> None:
        self.manager = manager
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()

    def _key(self, query: str) -> str:
        return f"kv_cache::{query}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._key(query)
        payload = self.manager.client.get(key)
        if payload is None:
            self.stats.record_miss()
            return None
        self.stats.record_hit()
        data = json.loads(payload)
        ttl = self.manager.client.ttl(key)
        if ttl is not None and ttl >= 0:
            data["ttl"] = ttl
        return data

    def set(self, query: str, answer: Dict[str, Any]) -> None:
        self.manager.client.set(self._key(query), json.dumps(answer), ex=self.ttl_seconds)

    def clear(self) -> None:
        keys = self.manager.client.keys(self._key("*"))
        if keys:
            self.manager.client.delete(*keys)


@dataclass
class SemanticCacheConfig:
    index_name: str
    distance_threshold: float
    vector_size: int


class SemanticCache:
    def __init__(self, manager: RedisManager, config: SemanticCacheConfig) -> None:
        self.manager = manager
        self.config = config
        self.stats = CacheStats()
        self.index = self._ensure_index()

    def _ensure_index(self) -> SearchIndex:
        schema = {
            "index": {
                "name": self.config.index_name,
                "prefix": "semantic_cache",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "query", "type": "text"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": self.config.vector_size,
                        "distance_metric": "COSINE",
                        "algorithm": "FLAT",
                    },
                },
                {"name": "answer", "type": "text"},
                {"name": "metadata", "type": "text"},
            ],
        }
        index = SearchIndex.from_dict(schema, redis_client=self.manager.client)
        if not index.exists():
            LOGGER.info("Creating Redis semantic cache index %s", self.config.index_name)
            index.create(overwrite=True, drop=True)
        return index

    def get(self, query_vector: list[float], top_k: int = 1) -> Optional[Dict[str, Any]]:
        vector_query = VectorQuery(
            vector=query_vector,
            vector_field_name="vector",
            return_fields=["answer", "metadata", "query"],
            num_results=top_k,
        )
        results = self.index.query(vector_query)
        if not results:
            self.stats.record_miss()
            return None
        best = results[0]
        distance = best.get("vector_distance", 1.0)
        if distance > self.config.distance_threshold:
            self.stats.record_miss()
            return None

        self.stats.record_hit()
        metadata = json.loads(best.get("metadata", "{}"))
        return {
            "answer": best.get("answer"),
            "metadata": metadata,
            "distance": distance,
        }

    def set(self, query: str, query_vector: list[float], answer: str, metadata: Dict[str, Any]) -> None:
        import uuid

        doc_id = str(uuid.uuid4())
        record = {
            "id": doc_id,
            "query": query,
            "vector": query_vector,
            "answer": answer,
            "metadata": json.dumps(metadata),
        }
        self.index.load([record], id_field="id")

    def clear(self) -> None:
        if self.index.exists():
            self.index.delete(drop=True)

