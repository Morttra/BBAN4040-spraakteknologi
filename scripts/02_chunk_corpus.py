"""Split the cleaned Wikipedia corpus into chunks with stable chunk_ids and save them under data/processed/.

Reads:  data/processed/articles.parquet and the `chunking` settings in configs/experiment.yaml
Writes: data/processed/chunks.parquet (chunk_id, article_id, language, title, position, text, n_words)
        and data/processed/chunks.meta.json (the settings used, git commit, time)
Uses:   norrag.chunking, norrag.records
Run:    python scripts/02_chunk_corpus.py      (a couple of minutes)

If you change the chunk settings, rerun 02, 03 and 04: every later step depends on the chunks.
"""

import pandas as pd

from norrag.chunking import chunk_corpus
from norrag.config import get_path, load_config
from norrag.records import write_meta


def main():
    config = load_config()
    settings = config["chunking"]
    processed = get_path(config, "data_processed")

    articles = pd.read_parquet(processed / "articles.parquet", columns=["article_id", "language", "title", "text"])
    chunks = chunk_corpus(articles, settings["max_words"], settings["overlap_words"], settings["add_title"])

    chunks.to_parquet(processed / "chunks.parquet", index=False)
    write_meta(processed / "chunks.parquet", config, n_articles=len(articles), n_chunks=len(chunks))
    print(f"Saved {len(chunks):,} chunks from {len(articles):,} articles to data/processed/chunks.parquet")


if __name__ == "__main__":
    main()
