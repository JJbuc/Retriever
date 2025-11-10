# Retriever Benchmark Suite

This project implements and benchmarks five retrieval strategies to inform production trade-offs between setup complexity and query performance. The stack is pure Python with Google Gemini for embeddings/generation/judging, Redis for caching, and optional Neo4j for the knowledge-graph retriever.

## Project Structure

- `retrievers/` – Individual retriever implementations
  - `vector_db_retriever.py`
  - `hybrid_retriever.py`
  - `kv_cache_retriever.py`
  - `semantic_cache_retriever.py`
  - `graph_rag_retriever.py`
- `benchmarks/` – Benchmark orchestration and analysis utilities
- `utils/` – Shared helpers (data loading, embeddings, Gemini, Redis, metrics, etc.)
- `data/` – PDF corpus to ingest (already populated)
- `questions.txt` – List of benchmark queries (one per line)
- `reports/` – Generated CSVs + analysis summaries

## Quickstart

1. **Create/activate the virtual environment**
   ```powershell
   cd C:\Users\Jay Jani\Documents\Python_Code\Retriever
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Set environment variables** (see below)
4. **Run the benchmark**
   ```powershell
   python run_benchmark.py --questions questions.txt
   ```
   Outputs:
   - `reports/setup_metrics.csv`
   - `reports/query_metrics.csv`
   - `reports/update_metrics.csv` (when `--run-updates` is used)

5. **Generate the analysis summary**
   ```powershell
   python -m benchmarks.analyze_results
   ```
   Produces `reports/analysis_summary.md`.

## Environment Variables

Create a `.env` file or export the following variables (defaults shown):

```
GROQ_API_KEY=your-key
GROQ_EMBED_MODEL=llama-embed-english-v3
GROQ_GENERATOR_MODEL=llama3-70b-8192
GROQ_JUDGE_MODEL=llama3-70b-8192

REDIS_URL=redis://localhost:6379/0
REDIS_TTL_SECONDS=86400
REDIS_SEMANTIC_INDEX=semantic_cache_idx
REDIS_SEMANTIC_THRESHOLD=0.15

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

## Update Benchmark

After modifying documents in `data/`, rerun:

```powershell
python run_benchmark.py --run-updates
```

`update_metrics.csv` captures rebuild timings and storage deltas per retriever.

## Notes

- Gemini is used for embeddings, final answer generation, and judge scores.
- `kv_cache` and `semantic_cache` retrievers depend on Redis Stack (vector commands).
- `graph_rag` builds an in-memory NetworkX graph and can push edges to Neo4j if configured.
- The analysis script aggregates CSV metrics into a one-page summary covering setup cost, query performance, cache effectiveness, and recommendations.

