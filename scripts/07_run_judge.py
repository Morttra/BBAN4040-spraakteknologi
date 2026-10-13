"""Run the LLM judge on the retrieved chunks and log every judgement (prompt, model, raw answer) under results/raw/.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/*.jsonl,
        annotations/key.csv, the `judge` settings and OPENAI_API_KEY from .env
Writes: results/raw/judge/<judge model>.jsonl, one line per (qid, chunk_id): prompt, raw reply, label
Uses:   norrag.judge, norrag.records
Run:    python scripts/07_run_judge.py --sample        first only the chunks you labelled by hand
        python scripts/07_run_judge.py --limit 1000    then more, once agreement looks good

This costs money: check the cost of a small run before starting a large one.
"""
