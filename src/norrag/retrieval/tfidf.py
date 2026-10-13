"""TF-IDF retriever with cosine similarity over stemmed Norwegian text.

TF-IDF gives each word in a chunk a weight that is high when the word is frequent in that chunk but
rare in the corpus as a whole. A query is turned into the same kind of vector, and chunks are ranked
by cosine similarity (how much the two vectors point in the same direction).

What goes here:
- class TfidfRetriever(Retriever): build() fits scikit-learn's TfidfVectorizer with text.analyze as
  the analyzer; search() multiplies the query vectors with the chunk matrix and keeps the top k.
- save/load: the fitted vectorizer (joblib) and the chunk matrix (scipy.sparse.save_npz).

Used by: scripts 03 and 04, through base.get_retriever.
"""
