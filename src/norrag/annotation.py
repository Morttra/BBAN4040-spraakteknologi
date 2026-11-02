"""Build pooled, blinded annotation sheets from the retrieval runs and read the human relevance labels back (evaluation layer 1).

How layer 1 works: draw a sample of questions, collect the top chunks from ALL retrievers ("pooling"),
remove duplicates, shuffle them and hide which retriever found each chunk ("blind"). Each of you then
labels every (question, chunk) pair as relevant (1) or not (0), following annotations/README.md.

What goes here:
- sample_questions(questions, n, seed) -> DataFrame: a random sample spread over the quiz topics.
- pool_chunks(runs, qids, k) -> DataFrame: every chunk that is in some retriever's top k.
- write_sheets(pool, questions, chunks, out_dir, annotators): one blind CSV per annotator to fill in
  (item_id, question, correct answer, chunk text, empty `relevant` column) and a separate key.csv
  (item_id -> qid, chunk_id and the retrievers that found it). The sheets open in Excel or Numbers.
- load_labels(sheet_paths, key_path) -> DataFrame: the labels joined back to qid and chunk_id.

Used by: scripts/06_make_annotation_sample.py, notebooks/00_main.ipynb.
"""
