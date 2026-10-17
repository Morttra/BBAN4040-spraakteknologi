"""Common interface shared by all retrievers: build an index, save and load it, and search for the top-k chunks.

What goes here:
- class Retriever with the methods every retriever has:
      build(chunks)               compute the index from the chunks DataFrame
      save(directory)             write the index to disk
      load(directory)             read it back
      search(queries, k)          for each query: (chunk_ids, scores), best first
  A class is a template for an object that keeps its index in memory between searches. bm25.py
  and dense.py each define a class that builds on Retriever and fills in these methods.
- get_retriever(name, config) -> Retriever: return the right one for "bm25" or "dense".

Used by: scripts/03_build_indexes.py, scripts/04_retrieve.py, scripts/08_make_demo_data.py, the demo,
tests/test_retrieval.py.
"""
