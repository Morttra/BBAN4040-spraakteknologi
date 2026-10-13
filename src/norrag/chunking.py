"""Split Wikipedia articles into passages with stable chunk_ids that are shared by all retrievers.

Why: whole articles are too long to put in a prompt, so each article is cut into chunks and the
retrievers search among the chunks. All three retrievers must use exactly the same chunks, so that a
relevance label on a chunk counts for every retriever that returns it.

What goes here:
- split_text(text, max_words, overlap_words) -> list of str: cut one text into pieces of about
  max_words words; neighbouring pieces share overlap_words words, so a fact on a boundary is not lost.
- make_chunk_id(article_id, position) -> str: e.g. "nb-1348455-0003". The same input always gives
  the same id; that is what "stable" means.
- chunk_corpus(articles, max_words, overlap_words, add_title) -> DataFrame: one row per chunk with
  chunk_id, article_id, language, title, position and text.

Used by: scripts/02_chunk_corpus.py, scripts/08_make_demo_data.py, tests/test_chunking.py.
Decisions (configs/experiment.yaml, `chunking`): chunk size, overlap, and whether each chunk starts
with the article title.
"""
