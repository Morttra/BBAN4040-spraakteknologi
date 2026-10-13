"""Score every question under each model x size x condition (no retrieval or RAG) and log the per-question results under results/raw/.

The experiment: Gemma without context, Borealis without context, and Gemma with chunks from each
retriever, all at 4B and 12B.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/*.jsonl,
        the `qa` settings
Writes: results/raw/qa/<model>__<condition>.jsonl, one line per question (qid, prompt, option
        log-probs, predicted, correct), plus a .meta.json with the model id, revision and settings
Uses:   norrag.models, norrag.pipeline (which uses prompts and scoring), norrag.records
Run:    python scripts/05_run_qa.py --model gemma-4b --condition rag_bm25 --limit 200

The longest step: one full condition can take hours, and 12B is roughly three times slower than 4B.
Use --limit for quick tests and run full conditions overnight; the script skips questions already done.
"""
