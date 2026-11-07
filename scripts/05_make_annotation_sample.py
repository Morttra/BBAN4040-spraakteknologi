"""Draw the question sample (at most one per quiz), pool and shuffle the top-k chunks from all retrievers, and write blind annotation sheets to annotations/.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/*.jsonl,
        the `annotation` settings
Writes: annotations/sheet_<name>.xlsx, one per annotator to fill in, and annotations/key.csv (which
        question, chunk and retrievers each row belongs to; don't look at it while labelling),
        plus annotations/key.meta.json with the settings used
Uses:   norrag.annotation, norrag.records
Run:    python scripts/05_make_annotation_sample.py      (seconds)

Run it once, then label. It refuses to overwrite sheets that are already there.
"""

import pandas as pd

from norrag.annotation import pool_chunks, sample_questions, write_sheets
from norrag.config import get_path, load_config
from norrag.records import read_jsonl_df, write_meta


def main():
    config = load_config()
    settings = config["annotation"]

    questions = pd.read_parquet(get_path(config, "data_processed") / "questions.parquet")
    sample = sample_questions(questions, settings["n_questions"], settings["seed"])

    runs_dir = get_path(config, "results_raw") / "runs"
    runs = pd.concat([read_jsonl_df(runs_dir / f"{name}.jsonl") for name in config["retrieval"]["retrievers"]])
    pool = pool_chunks(runs, sample.qid, settings["pool_k"])

    # Read only the texts of the pooled chunks, not all 2.5 million
    chunk_texts = pd.read_parquet(get_path(config, "data_processed") / "chunks.parquet", columns=["chunk_id", "text"],
                                  filters=[("chunk_id", "in", list(pool.chunk_id))]).set_index("chunk_id")["text"]

    out_dir = get_path(config, "annotations")
    try:
        write_sheets(pool, questions, chunk_texts, out_dir, settings["annotators"], settings["seed"])
    except FileExistsError as error:
        raise SystemExit(error)  # the message only, without the long Python traceback
    write_meta(out_dir / "key.csv", config, n_questions=len(sample), n_rows=len(pool))
    sheets = ", ".join(f"sheet_{name}.xlsx" for name in settings["annotators"])
    print(f"Wrote {len(pool)} rows for {len(sample)} questions to {out_dir}: {sheets} and key.csv")


if __name__ == "__main__":
    main()
