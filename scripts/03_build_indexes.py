"""Build and save the BM25 and dense indexes over the Wikipedia chunks.

Reads:  data/processed/chunks.parquet and the `retrieval` settings
Writes: data/indexes/bm25/, data/indexes/dense/<model name>/
Uses:   norrag.retrieval (base.get_retriever, and through it bm25, dense and text)
Run:    python scripts/03_build_indexes.py --retriever bm25      (one retriever at a time)

BM25 takes minutes. The dense index embeds every chunk and can take hours, so add a
--limit option and try it on a few thousand chunks first.
"""
