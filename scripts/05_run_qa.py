"""Score every question under each model x condition (no retrieval or RAG) and log the per-question results under results/raw/.

The experiment, with the 12B models: Borealis without context against Gemma with chunks from the
retriever chosen in the retrieval evaluation (the main comparison, fine-tuning vs RAG), and Gemma
without context as a reference.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/*.jsonl,
        the `qa` settings
Writes: results/raw/qa/<model>__<condition>.jsonl, one line per question (qid, prompt, option
        log-probs, predicted, correct), plus a .meta.json with the model id, revision and settings
Uses:   norrag.models, norrag.pipeline (which uses prompts and scoring), norrag.records
Run:    python scripts/05_run_qa.py --model gemma-12b --condition rag_bm25 --limit 200

The longest step: one full condition can take hours with the 12B models.
Use --limit for quick tests and run full conditions overnight; the script skips questions already done.
"""
