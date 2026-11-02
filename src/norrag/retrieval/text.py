"""Norwegian tokenization and Snowball stemming used by the BM25 retriever.

Sparse retrievers match words, so "huset" and "husene" should count as the same word "hus". Stemming
cuts words down to a shared stem.

What goes here:
- tokenize(text) -> list of str: lowercase and split into words, dropping punctuation. Numbers are kept:
  years matter in quiz questions.
- stem(tokens) -> list of str: Snowball's Norwegian stemmer (PyStemmer: Stemmer.Stemmer("norwegian")).
- analyze(text) -> list of str: tokenize and stem in one call; BM25 uses this for both the chunks and
  the queries, so they are treated the same way.

No stopword list: BM25 already gives very common words ("og", "i", "det") almost no weight.
Note: the Snowball stemmer is made for Bokmål, one reason the project uses Bokmål only.
Used by: retrieval/bm25.py.
"""

import re

import Stemmer

# A word is a run of letters and/or digits in any alphabet, so "kriváň" and "göteborg" stay whole.
# Hyphens and punctuation split words: "NRK-quizen" becomes "nrk" and "quizen", so it also matches "NRK".
WORD_PATTERN = re.compile(r"[^\W_]+")

# The stemmer is created once and reused; PyStemmer also remembers words it has already stemmed
STEMMER = Stemmer.Stemmer("norwegian")


def tokenize(text):
    """Lowercase the text and split it into words (letters and digits), dropping punctuation."""
    return WORD_PATTERN.findall(text.lower())


def stem(tokens):
    """Cut each word down to its Norwegian stem, e.g. "husene" -> "hus"."""
    return STEMMER.stemWords(tokens)


def analyze(text):
    """Turn a text into the list of stemmed words that BM25 indexes and searches with."""
    return stem(tokenize(text))
