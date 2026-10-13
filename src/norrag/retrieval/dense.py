"""Dense retriever that embeds chunks and queries with a sentence-embedding model (sentence-BERT or multilingual-e5) and searches by cosine similarity.

A sentence-embedding model turns a text into a vector (e.g. 768 numbers) so that texts with similar
meaning get similar vectors, even when they use different words.

What goes here:
- class DenseRetriever(Retriever): build() embeds every chunk with sentence-transformers, in batches;
  search() embeds the queries and keeps the chunks with the highest cosine similarity.
  Embedding the full corpus is the slowest step in the project (likely hours), so test on a sample first.
- Query and passage prefixes: multilingual-e5 expects "query: " and "passage: " in front of the texts;
  check each model's card and keep the prefixes in one small dictionary.
- save/load: the vectors as a NumPy file (float16 halves the size) plus the chunk_ids in the same order.

Used by: scripts 03 and 04, through base.get_retriever.
Decision (`retrieval.dense_model`): which sentence-BERT model, e.g. NbAiLab/nb-sbert-base or
NbAiLab/nb-sbert-v2-base; multilingual-e5 and NbAiLab/borealis-embed-212m later if there is time.
"""
