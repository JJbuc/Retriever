from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd

REPORTS_DIR = Path("reports")
SETUP_CSV = REPORTS_DIR / "setup_metrics.csv"
QUERY_CSV = REPORTS_DIR / "query_metrics.csv"
ANALYSIS_MD = REPORTS_DIR / "analysis_summary.md"


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def summarize_setup(setup_df: pd.DataFrame) -> str:
    rows = []
    for _, row in setup_df.iterrows():
        rows.append(
            f"- **{row['retriever_name']}**: build {row['total_build_seconds']:.2f}s "
            f"(chunk {row.get('breakdown_chunking_seconds', 0):.2f}s, "
            f"embed {row.get('breakdown_embedding_seconds', 0):.2f}s, "
            f"index {row.get('breakdown_indexing_seconds', 0):.2f}s, "
            f"graph {row.get('breakdown_graph_seconds', 0):.2f}s); "
            f"peak memory {row['memory_peak_mb']:.1f} MB; storage {row['storage_mb']:.2f} MB"
        )
    return "\n".join(rows)


def summarize_queries(query_df: pd.DataFrame) -> str:
    summary_parts = []
    grouped = query_df.groupby("retriever_name")
    for retriever, frame in grouped:
        mean_retrieval = frame["retrieval_latency_ms"].mean()
        mean_end_to_end = frame["end_to_end_latency_ms"].mean()
        context_score = frame["context_relevance"].mean()
        answer_score = frame["answer_relevance"].mean()
        tokens = frame["tokens_consumed"].mean()
        cache_hit_rate = 0.0
        if "extra_cache_hit" in frame.columns:
            cache_hits = frame[frame["extra_cache_hit"] == True]  # noqa: E712
            if not cache_hits.empty:
                cache_hit_rate = len(cache_hits) / len(frame)
        summary_parts.append(
            f"- **{retriever}**: retrieval {mean_retrieval:.1f} ms, end-to-end {mean_end_to_end:.1f} ms, "
            f"context score {context_score:.2f}, answer score {answer_score:.2f}, "
            f"tokens avg {tokens:.1f}, cache hit rate {cache_hit_rate:.2%}"
        )
    return "\n".join(summary_parts)


def summarize_cache_effectiveness(query_df: pd.DataFrame) -> str:
    parts = []
    for retriever in ["kv_cache", "semantic_cache"]:
        if retriever not in query_df["retriever_name"].unique():
            continue
        frame = query_df[query_df["retriever_name"] == retriever]
        if "extra_cache_hit" not in frame.columns:
            continue
        hit_frame = frame[frame["extra_cache_hit"] == True]  # noqa: E712
        miss_frame = frame[frame["extra_cache_hit"] == False]  # noqa: E712
        hit_latency = hit_frame["end_to_end_latency_ms"].mean() if not hit_frame.empty else float("nan")
        miss_latency = miss_frame["end_to_end_latency_ms"].mean() if not miss_frame.empty else float("nan")
        parts.append(
            f"- **{retriever}**: hits {len(hit_frame)}, misses {len(miss_frame)}, "
            f"hit latency {hit_latency:.1f} ms vs miss latency {miss_latency:.1f} ms"
        )
    if not parts:
        return "- Cache effectiveness metrics unavailable (no cache-specific data)."
    return "\n".join(parts)


def summarize_recommendations(setup_df: pd.DataFrame, query_df: pd.DataFrame) -> str:
    fastest = (
        query_df.groupby("retriever_name")["end_to_end_latency_ms"].mean().sort_values().head(1).index.tolist()
    )
    highest_answer = (
        query_df.groupby("retriever_name")["answer_relevance"].mean().sort_values(ascending=False).head(1).index.tolist()
    )
    minimal_setup = setup_df.sort_values("total_build_seconds").head(1)["retriever_name"].tolist()
    recommendations = [
        f"- **Low latency**: {', '.join(fastest)}",
        f"- **Best answer quality**: {', '.join(highest_answer)}",
        f"- **Fastest setup**: {', '.join(minimal_setup)}",
        "- **Hybrid**: choose `hybrid` when balancing lexical + semantic signals matters.",
        "- **Graph RAG**: reserve for multi-hop and relationship-heavy queries where structured traversals help.",
    ]
    return "\n".join(recommendations)


def main() -> None:
    setup_df = load_frame(SETUP_CSV)
    query_df = load_frame(QUERY_CSV)

    markdown = dedent(
        f"""
        # Retrieval Benchmark Analysis

        ## Setup Costs
        {summarize_setup(setup_df)}

        ## Query Performance
        {summarize_queries(query_df)}

        ## Cache Effectiveness
        {summarize_cache_effectiveness(query_df)}

        ## Recommendations
        {summarize_recommendations(setup_df, query_df)}
        """
    ).strip()

    ANALYSIS_MD.write_text(markdown, encoding="utf-8")
    print(f"Wrote analysis summary to {ANALYSIS_MD}")


if __name__ == "__main__":
    main()

