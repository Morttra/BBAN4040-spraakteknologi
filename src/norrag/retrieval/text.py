"""Norwegian tokenization and Snowball stemming used by the sparse retrievers (TF-IDF and BM25).

Sparse retrievers match words, so "huset" and "husene" should count as the same word "hus". Stemming
cuts words down to a shared stem.

What goes here:
- tokenize(text) -> list of str: lowercase and split into words, dropping punctuation. Keep numbers:
  years matter in quiz questions.
- stem(tokens) -> list of str: Snowball's Norwegian stemmer (PyStemmer: Stemmer.Stemmer("norwegian")).
- STOPWORDS (optional): very common words such as "og", "i" and "det" that carry little meaning.
- analyze(text) -> list of str: tokenize, remove stopwords and stem in one call. TF-IDF and BM25 both
  use this, so they see the text the same way.

Note: the Snowball stemmer is made mainly for Bokmål and works less well on Nynorsk; worth a sentence
in the report.
Used by: retrieval/tfidf.py, retrieval/bm25.py.
"""
