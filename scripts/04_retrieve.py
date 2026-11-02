"""Retrieve the top-20 chunks for every question with each retriever and save one run file per retriever under results/raw/.

Reads:  data/processed/questions.parquet, data/indexes/<retriever>/, the `retrieval` settings
Writes: results/raw/runs/<retriever>.jsonl, one line per question, e.g.
            {"qid": "...", "retriever": "bm25", "query": "...", "chunk_ids": [20 ids, best first], "scores": [...]}
        and results/raw/runs/<retriever>.meta.json with the settings used
Uses:   norrag.retrieval (query.build_query, base.get_retriever, base.index_dir), norrag.records
Run:    python scripts/04_retrieve.py                        both retrievers in the config
        python scripts/04_retrieve.py --retriever bm25       only one
        python scripts/04_retrieve.py --limit 50             a test on the first 50 questions
                                                             (saved as <retriever>_limit50.jsonl)

The run files feed 05 (chunks for the prompts), 06 (annotation), 07 (judge) and 08 (demo).
How good the retrieved chunks are is measured later, with norrag.metrics, from these run files and the labels.
"""

import argparse

import pandas as pd

from norrag.config import get_path, load_config
from norrag.records import write_jsonl, write_meta
from norrag.retrieval.base import get_retriever, index_dir
from norrag.retrieval.query import build_query


def main():
    parser = argparse.ArgumentParser(description="Retrieve the top chunks for every question.")
    parser.add_argument("--retriever", choices=["bm25", "dense"], help="run only this one (default: all in the config)")
    parser.add_argument("--limit", type=int, help="use only the first N questions, for testing")
    args = parser.parse_args()

    config = load_config()
    settings = config["retrieval"]
    if settings["query_mode"] is None:
        raise SystemExit("Choose retrieval.query_mode in configs/experiment.yaml first (full, last_sentence or with_options).")

    questions = pd.read_parquet(get_path(config, "data_processed") / "questions.parquet")
    if args.limit:
        questions = questions.head(args.limit)
    queries = [build_query(q.question, q.options, settings["query_mode"]) for q in questions.itertuples()]

    names = [args.retriever] if args.retriever else settings["retrievers"]
    for name in names:
        retriever = get_retriever(name, config)
        retriever.load(index_dir(config, name))
        results = retriever.search(queries, k=settings["top_n"])

        rows = [
            {"qid": qid, "retriever": name, "query": query, "chunk_ids": chunk_ids, "scores": scores}
            for qid, query, (chunk_ids, scores) in zip(questions["qid"], queries, results)
        ]
        file_name = f"{name}_limit{args.limit}.jsonl" if args.limit else f"{name}.jsonl"
        path = get_path(config, "results_raw") / "runs" / file_name
        write_jsonl(path, rows)
        write_meta(path, config, retriever=name, query_mode=settings["query_mode"], n_questions=len(rows))
        print(f"Saved {len(rows):,} {name} results to {path}")


if __name__ == "__main__":
    main()
