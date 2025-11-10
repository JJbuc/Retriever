import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataPaths:
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")
    reports_dir: Path = Path("reports")

    def ensure(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class GeminiSettings:
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
    generator_model: str = os.getenv("GEMINI_GENERATOR_MODEL", "models/gemini-2.5-flash")
    judge_model: str = os.getenv("GEMINI_JUDGE_MODEL", "models/gemini-2.5-flash")


@dataclass
class EmbeddingSettings:
    provider: str = os.getenv("EMBED_PROVIDER", "gemini").lower()
    model_name: str = os.getenv(
        "EMBED_MODEL_NAME",
        os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
    )


@dataclass
class RedisSettings:
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ttl_seconds: int = int(os.getenv("REDIS_TTL_SECONDS", "86400"))
    semantic_index_name: str = os.getenv("REDIS_SEMANTIC_INDEX", "semantic_cache_idx")
    semantic_distance_threshold: float = float(os.getenv("REDIS_SEMANTIC_THRESHOLD", "0.15"))


@dataclass
class Neo4jSettings:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class BenchmarkSettings:
    num_query_repeats: int = int(os.getenv("BENCHMARK_QUERY_REPEATS", "20"))
    cache_repeat_fraction: float = float(os.getenv("BENCHMARK_CACHE_REPEAT_FRACTION", "0.5"))
    warmup_runs: int = int(os.getenv("BENCHMARK_WARMUP_RUNS", "1"))


@dataclass
class AppConfig:
    data_paths: DataPaths = field(default_factory=DataPaths)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    neo4j: Neo4jSettings = field(default_factory=Neo4jSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def ensure_dirs(self) -> None:
        self.data_paths.ensure()


CONFIG = AppConfig()

