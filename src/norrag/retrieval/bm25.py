"""BM25 retriever over stemmed Norwegian text.

BM25 is a refined TF-IDF used by most search engines: each extra occurrence of a word counts a bit
less than the one before, and long chunks are not favoured just for being long. Two settings, k1 and
b, control this; the usual defaults are k1 = 1.5 and b = 0.75.

What goes here:
- class BM25Retriever(Retriever): build() runs text.analyze on every chunk and indexes the result
  with the bm25s library; search() returns the top-k chunk_ids and scores.
- save/load: bm25s saves and loads its index as a folder.

Used by: scripts 03 and 04 through base.get_retriever, and the demo (BM25 is quick to build on a
small subset).
"""
