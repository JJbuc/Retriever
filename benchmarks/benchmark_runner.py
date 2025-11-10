from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from config import CONFIG, AppConfig
from retrievers.graph_rag_retriever import GraphRAGRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.kv_cache_retriever import KVCacheRetriever
from retrievers.semantic_cache_retriever import SemanticCacheRetriever
from retrievers.vector_db_retriever import VectorDBRetriever
from utils.data_loader import CorpusBuilder, build_sha_index
from utils.embedding_utils import EmbeddingClient, EmbeddingConfig
from utils.gemini_helpers import GeminiService
from utils.metrics import directory_size_mb
from utils.redis_utils import RedisManager
from utils.types import QueryMetrics, SetupMetrics

from .metrics_recorder import MetricsRecorder

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, CONFIG.log_level.upper(), logging.INFO))


@dataclass
class QueryTask:
    query_id: int
    text: str
    is_repeat: bool


class BenchmarkRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        self.corpus_builder = CorpusBuilder(self.config.data_paths.data_dir)
        embedding_provider = self.config.embedding.provider
        embedding_api_key = self.config.gemini.api_key if embedding_provider == "gemini" else None
        self.embedding_client = EmbeddingClient(
            api_key=embedding_api_key,
            config=EmbeddingConfig(
                provider=self.config.embedding.provider,
                model_name=self.config.embedding.model_name,
            ),
        )
        self.gemini_service = GeminiService(
            api_key=self.config.gemini.api_key,
            generator_model=self.config.gemini.generator_model,
            judge_model=self.config.gemini.judge_model,
        )
        self.redis_manager = RedisManager(self.config.redis.url)

        storage_root = self.config.data_paths.storage_dir

        self.retrievers = [
            VectorDBRetriever(storage_root, self.embedding_client, self.gemini_service),
            HybridRetriever(storage_root, self.embedding_client, self.gemini_service),
            KVCacheRetriever(
                storage_root,
                self.embedding_client,
                self.gemini_service,
                self.redis_manager,
                ttl_seconds=self.config.redis.ttl_seconds,
            ),
            SemanticCacheRetriever(
                storage_root,
                self.embedding_client,
                self.gemini_service,
                self.redis_manager,
                index_name=self.config.redis.semantic_index_name,
                distance_threshold=self.config.redis.semantic_distance_threshold,
            ),
            GraphRAGRetriever(
                storage_root,
                self.gemini_service,
                neo4j_uri=self.config.neo4j.uri,
                neo4j_user=self.config.neo4j.user,
                neo4j_password=self.config.neo4j.password,
                neo4j_database=self.config.neo4j.database,
            ),
        ]

        self.metrics = MetricsRecorder(self.config.data_paths.reports_dir)
        self._baseline_hashes: Dict[str, str] = {}

    def load_questions(self, questions_path: Path) -> List[str]:
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        questions = [line.strip() for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not questions:
            raise ValueError("Questions file is empty.")
        return questions

    def build_query_plan(self, questions: Sequence[str]) -> List[QueryTask]:
        total_queries = self.config.benchmark.num_query_repeats
        repeat_fraction = self.config.benchmark.cache_repeat_fraction
        plan: List[QueryTask] = []

        # Ensure at least one pass over each question
        unique_sequence = list(questions)
        for idx, question in enumerate(unique_sequence):
            if len(plan) >= total_queries:
                break
            plan.append(QueryTask(query_id=len(plan), text=question, is_repeat=False))

        remaining = total_queries - len(plan)
        if remaining > 0:
            repeats_needed = int(total_queries * repeat_fraction)
            repeats_needed = max(repeats_needed, len(questions))
            for i in range(repeats_needed):
                if len(plan) >= total_queries:
                    break
                plan.append(
                    QueryTask(
                        query_id=len(plan),
                        text=questions[i % len(questions)],
                        is_repeat=True,
                    )
                )

        # If still short, cycle remaining questions without repeat flag
        while len(plan) < total_queries:
            question = questions[len(plan) % len(questions)]
            plan.append(QueryTask(query_id=len(plan), text=question, is_repeat=False))

        return plan

    def run_setup(self) -> List[SetupMetrics]:
        LOGGER.info("Building corpus artifacts...")
        artifacts = self.corpus_builder.build()
        data_paths = self.corpus_builder.discover_documents()
        self._baseline_hashes = build_sha_index(data_paths)

        setup_results: List[SetupMetrics] = []
        for retriever in self.retrievers:
            LOGGER.info("Building retriever: %s", retriever.name)
            metrics = retriever.build(artifacts)
            storage_paths = retriever.storage_paths()
            metrics.storage_mb = directory_size_mb(storage_paths)
            setup_results.append(metrics)
            self.metrics.add_setup(metrics)
        return setup_results

    def run_queries(self, questions: Sequence[str]) -> List[QueryMetrics]:
        plan = self.build_query_plan(questions)
        results: List[QueryMetrics] = []
        for retriever in self.retrievers:
            LOGGER.info("Running queries for %s", retriever.name)
            for task in plan:
                try:
                    _, metrics = retriever.retrieve(
                        task.text,
                        task.query_id,
                        is_repeat=task.is_repeat,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("Query failed for %s: %s", retriever.name, exc)
                    metrics = QueryMetrics(
                        retriever_name=retriever.name,
                        query_id=task.query_id,
                        query_text=task.text,
                        retrieval_latency_ms=0.0,
                        end_to_end_latency_ms=0.0,
                        context_relevance=0.0,
                        answer_relevance=0.0,
                        tokens_consumed=0,
                        error=str(exc),
                        extra={"is_repeat": task.is_repeat},
                    )
                self.metrics.add_query(metrics)
                results.append(metrics)
        return results

    def run_update_benchmark(self) -> None:
        LOGGER.info("Measuring update performance after data change...")
        current_hashes = build_sha_index(self.corpus_builder.discover_documents())
        changed_files = [name for name, digest in current_hashes.items() if self._baseline_hashes.get(name) != digest]
        timings: Dict[str, float] = {}
        if not changed_files:
            LOGGER.info("No data changes detected; skipping update benchmark.")
            return
        LOGGER.info("Detected changed files: %s", ", ".join(changed_files))
        updated_artifacts = self.corpus_builder.build()
        for retriever in self.retrievers:
            start = time.perf_counter()
            retriever.build(updated_artifacts)
            elapsed = time.perf_counter() - start
            timings[retriever.name] = elapsed
            self.metrics.add_update(retriever.name, {"total_seconds": elapsed}, retriever.storage_paths())
        self._baseline_hashes = current_hashes

    def execute(self, questions_path: Path, run_queries: bool = True, run_updates: bool = False) -> None:
        questions: List[str] = []
        if run_queries:
            questions = self.load_questions(questions_path)
        self.run_setup()
        if run_queries:
            self.run_queries(questions)
        if run_updates:
            self.run_update_benchmark()
        self.metrics.write_csv()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark retriever strategies.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("questions.txt"),
        help="Path to questions file.",
    )
    parser.add_argument("--skip-queries", action="store_true", help="Skip query benchmarking.")
    parser.add_argument("--run-updates", action="store_true", help="Run update benchmark after setup.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    runner = BenchmarkRunner(CONFIG)
    runner.execute(
        questions_path=args.questions,
        run_queries=not args.skip_queries,
        run_updates=args.run_updates,
    )


if __name__ == "__main__":
    main()

