"""Split the cleaned Wikipedia corpus into chunks with stable chunk_ids and save them under data/processed/.

Reads:  data/processed/articles.parquet and the `chunking` settings in configs/experiment.yaml
Writes: data/processed/chunks.parquet (chunk_id, article_id, language, title, position, text)
Uses:   norrag.chunking
Run:    python scripts/02_chunk_corpus.py      (minutes)

If you change the chunk settings, rerun 02, 03 and 04: every later step depends on the chunks.
"""
