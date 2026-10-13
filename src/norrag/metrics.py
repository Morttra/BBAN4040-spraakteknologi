"""Evaluation metrics: QA accuracy, retrieval hit@k and MRR@k, judge-human agreement (Cohen's kappa) and the retrieval x answer error breakdown (layer 3).

What goes here (each function takes pandas DataFrames and returns a number or a table):
- accuracy(qa) -> float, and accuracy_by(qa, ["model", "condition", "language"]) -> DataFrame.
- chance_accuracy(questions) -> float: the expected score from guessing. Not 25 %, because the number
  of options varies.
- hit_at_k(run, labels, k) -> float: share of questions with at least one relevant chunk in the top k.
- mrr_at_k(run, labels, k) -> float: the average of 1/rank of the first relevant chunk (0 if none).
- cohens_kappa(labels_a, labels_b) -> float: agreement beyond chance, between the two of you or
  between the judge and you (sklearn.metrics.cohen_kappa_score does the maths).
- error_breakdown(qa, hits) -> DataFrame: each question as retrieval hit/miss x answer right/wrong.

Used by: notebooks/00_main.ipynb, tests/test_metrics.py.
"""
