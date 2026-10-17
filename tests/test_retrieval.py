"""Unit tests for the retrievers on a tiny toy corpus.

Ideas: five made-up chunks such as "Oslo er hovedstaden i Norge."; a query with an obvious answer
must return that chunk first for BM25; search returns exactly k results; saving and loading
gives the same results; text.analyze gives "huset" and "husene" the same stem.
"""
