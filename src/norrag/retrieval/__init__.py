"""Retrievers that return the top-k chunks for a question: BM25 and dense embeddings.

Both follow the same pattern (see base.py), so the scripts can treat them alike:
build an index from the chunks once -> save it -> load it later -> search for each question.
This file can stay empty.
"""
