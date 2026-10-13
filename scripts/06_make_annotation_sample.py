"""Draw the stratified question sample, pool and shuffle the top-k chunks from all retrievers, and write blind annotation sheets to annotations/.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/*.jsonl,
        the `annotation` settings
Writes: annotations/sheet_<name>.csv, one per annotator to fill in, and annotations/key.csv (which
        question, chunk and retrievers each row belongs to; don't look at it while labelling)
Uses:   norrag.annotation
Run:    python scripts/06_make_annotation_sample.py      (seconds)

Run it once, then label. Running it again with another seed gives a different sample.
"""
