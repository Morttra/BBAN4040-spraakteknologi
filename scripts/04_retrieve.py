"""Retrieve the top-20 chunks for every question with each retriever and save one run file per retriever under results/raw/.

Reads:  data/processed/questions.parquet, data/indexes/<retriever>/, the `retrieval` settings
Writes: results/raw/runs/<retriever>.jsonl, one line per question, e.g.
            {"qid": "...", "retriever": "bm25", "query": "...", "chunk_ids": [20 ids, best first], "scores": [...]}
        and results/raw/runs/<retriever>.meta.json with the settings used
Uses:   norrag.retrieval (query.build_query, base.get_retriever), norrag.records
Run:    python scripts/04_retrieve.py --retriever bm25      (minutes)

The run files feed 05 (chunks for the prompts), 06 (annotation), 07 (judge) and 08 (demo).
"""
