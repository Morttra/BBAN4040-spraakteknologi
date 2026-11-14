"""Common interface shared by all retrievers: build an index, save and load it, and search for the top-k chunks.

Both retrievers (bm25.py and dense.py) are classes with the same three methods, so the scripts can use
either one without knowing which it is:
      build(chunks, directory)    compute the index from the chunks DataFrame and save it to directory
      load(directory)             read a saved index back into memory
      search(queries, k)          for each query: (chunk_ids, scores), best first
A class is a template for an object that keeps its index in memory between searches.

What goes here:
- get_retriever(name, config): a new, empty retriever of the right kind, "bm25" or "dense".
- index_dir(config, name) -> Path: where that retriever's index is saved, e.g. data/indexes/bm25/.

Used by: scripts/03_build_indexes.py, scripts/04_retrieve.py, demo/demo.ipynb.
"""

from norrag.config import get_path
from norrag.retrieval.bm25 import BM25Retriever
from norrag.retrieval.dense import DenseRetriever


def get_retriever(name, config):
    """Return a new retriever of the given kind ("bm25" or "dense"), set up from the config."""
    settings = config["retrieval"]
    if name == "bm25":
        return BM25Retriever(k1=settings["bm25_k1"], b=settings["bm25_b"])
    if name == "dense":
        return DenseRetriever(settings["dense_model"], batch_size=settings["dense_batch_size"])
    raise ValueError(f"Unknown retriever {name!r}; use 'bm25' or 'dense'.")


def index_dir(config, name):
    """Return the folder where a retriever's index is saved; dense indexes get a folder per model."""
    folder = get_path(config, "indexes") / name
    if name == "dense":
        folder = folder / config["retrieval"]["dense_model"].replace("/", "__")  # "NbAiLab/x" -> "NbAiLab__x"
    return folder
