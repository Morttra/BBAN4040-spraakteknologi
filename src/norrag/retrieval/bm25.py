"""BM25 retriever over stemmed Norwegian text.

BM25 is a refined TF-IDF used by most search engines: each extra occurrence of a word counts a bit
less than the one before, and long chunks are not favoured just for being long. Two settings, k1 and
b, control this; the usual defaults are k1 = 1.5 and b = 0.75.

What goes here:
- class BM25Retriever with build(chunks, directory), load(directory) and search(queries, k), the same
  three methods as the dense retriever (see base.py).

The index is built with the bm25s library, which stores it as a sparse matrix and can search millions of
chunks quickly. Every chunk and every query goes through text.analyze, so they are split and stemmed
the same way.
Used by: scripts 03 and 04 through base.get_retriever, and the demo (BM25 is quick to build on a
small subset).
"""

import json
from pathlib import Path

import bm25s
import numpy as np
from tqdm import tqdm

from norrag.retrieval.text import analyze


class BM25Retriever:
    """A BM25 index over the chunks, kept in memory between searches."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.model = None  # the bm25s index, created by build() or load()
        self.chunk_ids = None  # chunk_ids in index order: position i in the index is chunk_ids[i]

    def build(self, chunks, directory):
        """Index the chunks (a DataFrame with chunk_id and text) and save the index to directory."""
        # Turn every chunk into a list of word numbers; vocab maps each stemmed word to its number.
        # Numbers in a NumPy array take far less memory than millions of lists of words.
        vocab = {}
        docs = []
        for text in tqdm(chunks["text"], desc="Analysing chunks", unit=" chunks"):
            word_ids = [vocab.setdefault(word, len(vocab)) for word in analyze(text)]  # new words get the next number
            docs.append(np.array(word_ids, dtype=np.int32))

        self.model = bm25s.BM25(k1=self.k1, b=self.b)
        self.model.index((docs, vocab))
        self.chunk_ids = chunks["chunk_id"].tolist()

        directory = Path(directory)
        self.model.save(directory)
        with open(directory / "chunk_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f)

    def load(self, directory):
        """Load an index saved by build()."""
        directory = Path(directory)
        self.model = bm25s.BM25.load(directory)
        with open(directory / "chunk_ids.json", encoding="utf-8") as f:
            self.chunk_ids = json.load(f)

    def search(self, queries, k=20):
        """Return the top-k chunks for each query, as a list of (chunk_ids, scores) pairs, best first."""
        query_words = [analyze(query) for query in queries]
        positions, scores = self.model.retrieve(query_words, k=k)  # one row per query
        return [
            ([self.chunk_ids[i] for i in row], row_scores.tolist())
            for row, row_scores in zip(positions, scores)
        ]
