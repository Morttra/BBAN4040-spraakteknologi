"""Unit tests for chunking: chunk_ids are stable between runs and the chunks cover the article text.

A test is a small function whose name starts with test_ and that checks one thing with `assert`.
Run all tests with `pytest` from the repo folder. Ideas:
- chunking the same text twice gives the same chunk_ids,
- no chunk is longer than max_words,
- every word of the text ends up in at least one chunk,
- an empty or very short article gives zero or one chunk.
"""
