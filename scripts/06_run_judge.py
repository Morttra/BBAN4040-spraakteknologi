"""Run the LLM judge on the retrieved chunks and log every judgement (model, raw reply, label) under results/raw/.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/*.jsonl,
        annotations/key.csv (with --sample), the `judge` settings and the API key from .env
Writes: results/raw/judge/<judge model>.jsonl, one line per (qid, chunk_id): raw reply, label and token
        counts, plus a .meta.json with the prompt template and the reply schema. The file is committed
        (see .gitignore).
Uses:   norrag.judge, norrag.annotation (the same pool of chunks as the sheets), norrag.records
Run:    python scripts/06_run_judge.py --sample            the 469 chunks you labelled by hand
        python scripts/06_run_judge.py --sample --limit 5  a test on 5 of them
        python scripts/06_run_judge.py                     every pooled chunk for every question,
                                                           one call at a time (about 13 hours)

This costs money: check the cost of a small run before starting a large one. Pairs already judged are
skipped, so a stopped run continues where it left off, and the --sample judgements are reused later.
"""

import argparse

import pandas as pd

from norrag.annotation import pool_chunks
from norrag.config import get_path, load_config
from norrag.judge import JUDGE_PROMPT, JUDGE_SCHEMA, judge_many, make_client
from norrag.records import read_jsonl_df, write_meta


def main():
    parser = argparse.ArgumentParser(description="Let the LLM judge label retrieved chunks as relevant or not.")
    parser.add_argument("--sample", action="store_true", help="only the chunks you labelled by hand (annotations/key.csv)")
    parser.add_argument("--limit", type=int, help="only the first N pairs, for testing")
    args = parser.parse_args()

    config = load_config()
    settings = config["judge"]
    questions = pd.read_parquet(get_path(config, "data_processed") / "questions.parquet").set_index("qid", drop=False)
    if args.sample:
        pairs = pd.read_csv(get_path(config, "annotations") / "key.csv")[["qid", "chunk_id"]]
    else:  # every chunk in some retriever's top k for every question: the same pool as the annotation sheets
        runs_dir = get_path(config, "results_raw") / "runs"
        runs = pd.concat([read_jsonl_df(runs_dir / f"{name}.jsonl") for name in config["retrieval"]["retrievers"]])
        pairs = pool_chunks(runs, questions.qid, config["annotation"]["pool_k"])[["qid", "chunk_id"]]
    if args.limit:
        pairs = pairs.head(args.limit)

    chunk_texts = pd.read_parquet(get_path(config, "data_processed") / "chunks.parquet", columns=["chunk_id", "text"],
                                  filters=[("chunk_id", "in", list(pairs.chunk_id.unique()))]).set_index("chunk_id")["text"]
    path = get_path(config, "results_raw") / "judge" / f"{settings['model']}.jsonl"
    write_meta(path, config, provider=settings["provider"], model=settings["model"], effort=settings["effort"],
               judge_prompt=JUDGE_PROMPT, judge_schema=JUDGE_SCHEMA if settings["provider"] == "anthropic" else None)
    client = make_client(settings["provider"])
    judge_many(client, settings, pairs, questions, chunk_texts, path)


if __name__ == "__main__":
    main()
