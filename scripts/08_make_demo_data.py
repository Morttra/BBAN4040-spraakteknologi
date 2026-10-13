"""Copy a small subset of questions, chunks and a prebuilt BM25 index into demo/data/ for the reproduction demo.

The demo must run in about ten minutes on an ordinary laptop, so it gets about 20 questions and only
the chunks it needs: those retrieved for the demo questions plus a random sample of others, so that
the search still has something to search through.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/bm25.jsonl,
        the `demo` settings
Writes: demo/data/questions.parquet, demo/data/chunks.parquet, demo/data/bm25/ (committed to Git, so
        keep it small)
Uses:   norrag.retrieval
Run:    python scripts/08_make_demo_data.py      (seconds)
"""
