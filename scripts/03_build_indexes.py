"""Build and save the BM25 and dense indexes over the Wikipedia chunks.

Reads:  data/processed/chunks.parquet and the `retrieval` settings
Writes: data/indexes/bm25/ and data/indexes/dense/<model name>/, each with an index.meta.json
Uses:   norrag.retrieval (base.get_retriever and base.index_dir, and through them bm25, dense and text)
Run:    python scripts/03_build_indexes.py                       both retrievers in the config
        python scripts/03_build_indexes.py --retriever bm25      only one
        python scripts/03_build_indexes.py --retriever dense --limit 20000   a test on the first 20,000 chunks,
                                                                  saved to a separate "_sample20000" folder

BM25 takes minutes. The dense index embeds every chunk and takes hours; it saves its work block by
block, so if it is stopped, running the same command again continues where it left off.
"""

import argparse
import time

import pandas as pd

from norrag.config import get_path, load_config
from norrag.records import write_meta
from norrag.retrieval.base import get_retriever, index_dir


def main():
    parser = argparse.ArgumentParser(description="Build the retrieval indexes over the chunks.")
    parser.add_argument("--retriever", choices=["bm25", "dense"], help="build only this one (default: all in the config)")
    parser.add_argument("--limit", type=int, help="use only the first N chunks, for testing (saved to a separate folder)")
    args = parser.parse_args()

    config = load_config()
    chunks = pd.read_parquet(get_path(config, "data_processed") / "chunks.parquet", columns=["chunk_id", "text"])
    if args.limit:
        chunks = chunks.head(args.limit)

    names = [args.retriever] if args.retriever else config["retrieval"]["retrievers"]
    for name in names:
        directory = index_dir(config, name)
        if args.limit:  # keep test indexes apart from the real ones
            directory = directory.with_name(f"{directory.name}_sample{args.limit}")

        print(f"Building the {name} index over {len(chunks):,} chunks in {directory} ...")
        start = time.time()
        get_retriever(name, config).build(chunks, directory)
        minutes = (time.time() - start) / 60

        write_meta(directory / "index", config, retriever=name, n_chunks=len(chunks), minutes=round(minutes, 1))
        print(f"Done in {minutes:.1f} minutes.")


if __name__ == "__main__":
    main()
