"""Evaluation metrics: QA accuracy, retrieval hit@k and MRR@k, confidence intervals and significance tests, judge-human agreement (Cohen's kappa) and the retrieval x answer error breakdown (layer 3).

What goes here:
- Retrieval, for one question (retrieved_ids best first, relevant_ids the chunks labelled relevant):
  hit_at_k and mrr_at_k. Averaging them over all questions gives the scores per retriever.
  retrieval_scores(runs, labels, k) applies both to every question and retriever in the run files.
  Recall@k is left out: we only know which chunks are relevant among those we labelled, not among all
  2.5 million.
- Format check: letter_mass(logprobs) is the share of the model's probability on the option letters
  for one question; the mean over all questions should be at least 90 %.
- QA: accuracy is the mean of the `correct` column in a QA file, so bootstrap_ci(results.correct)
  gives it together with its confidence interval; chance_accuracy(n_options) is the score expected
  from guessing, not 25 %, because the number of options varies.
- Uncertainty: bootstrap_ci(values) gives a 95 % confidence interval for a score computed from
  per-question values (accuracy, hit@k, MRR, or the difference between two systems);
  mcnemar_test(correct_a, correct_b) tells whether two systems answering the same questions really differ.
- Agreement: cohens_kappa(labels_a, labels_b) is agreement beyond chance, between the two of you or
  between the judge and you; judge_confusion(human, judge) puts the judge's labels against the human
  labels in a 2x2 table, showing whether the judge is too generous or too strict.
- Error breakdown: error_breakdown(qa, hits) counts each question as retrieval hit/miss x answer
  right/wrong, with the accuracy for hits and for misses.

Used by: notebooks/00_main.ipynb, notebooks/02_error-analysis.ipynb, demo/demo.ipynb.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def hit_at_k(retrieved_ids, relevant_ids, k):
    """1 if at least one relevant chunk is among the top k, else 0. The main retrieval metric."""
    return 1.0 if set(retrieved_ids[:k]) & set(relevant_ids) else 0.0
    

def mrr_at_k(retrieved_ids, relevant_ids, k):
    """1/rank of the first relevant chunk in the top k (1 if it is first, 0.5 if second, ...), or 0 if none."""
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def retrieval_scores(runs, labels, k):
    """hit@k and MRR@k for every question and retriever, with the chunk labels from us or from the judge.

    runs: one row per (qid, retriever) with chunk_ids best first, as in the run files.
    labels: one row per labelled (qid, chunk_id) with relevant = 1 or 0.
    Only the questions in labels are scored, so the 50 hand-labelled questions can be scored on their own.
    Returns one row per (qid, retriever) with the columns qid, retriever, hit and mrr.
    """
    runs = runs[runs.qid.isin(labels.qid)]
    # Every chunk in a top k must have a label, otherwise a relevant chunk could be counted as a miss
    labelled = set(zip(labels.qid, labels.chunk_id))
    missing = sum((row.qid, chunk_id) not in labelled for row in runs.itertuples() for chunk_id in row.chunk_ids[:k])
    if missing:
        raise ValueError(f"{missing} chunks in the top {k} have no label: is k larger than annotation.pool_k?")
    # The set of relevant chunk ids for each question; a question with none gets an empty set
    relevant = labels[labels.relevant == 1].groupby("qid")["chunk_id"].agg(set)
    rows = []
    for row in runs.itertuples():
        relevant_ids = relevant.get(row.qid, set())
        rows.append({"qid": row.qid, "retriever": row.retriever,
                     "hit": hit_at_k(row.chunk_ids, relevant_ids, k), "mrr": mrr_at_k(row.chunk_ids, relevant_ids, k)})
    return pd.DataFrame(rows)


def letter_mass(logprobs):
    """The share of the model's probability for its next token that lands on the option letters (0 to 1).

    logprobs: the log-probabilities of the option letters for one question, as saved in the QA files.
    Near 1 means the model wants to answer with a letter. Lower means it would rather have started with
    something else, such as "**" or "Svaret", and the comparison between the letters says less.
    """
    return float(np.exp(logprobs).sum())  # exp turns a log-probability back into a probability


def chance_accuracy(n_options):
    """The accuracy expected from guessing at random: the mean of 1 / (number of options) over the questions.

    n_options holds one count per question. Not simply 25 %: a question with 3 options is guessed right
    a third of the time, one with 2 options half the time.
    """
    return float(np.mean(1 / np.asarray(n_options)))


def bootstrap_ci(values, statistic=np.mean, n_resamples=1000, confidence=0.95, seed=42):
    """Confidence interval for a score by resampling the questions; returns (score, low, high).

    values has one entry per question, e.g. hit@5 as 0/1 or a model's answers as 1 (right) / 0 (wrong).
    For the difference between two systems on the same questions, pass the per-question differences.
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    # Draw n questions with replacement (some twice, some not at all) and score them, many times over
    scores = [statistic(values[rng.integers(0, n, size=n)]) for _ in range(n_resamples)]
    tail = (1 - confidence) / 2 * 100  # 2.5 for a 95 % interval
    low, high = np.percentile(scores, [tail, 100 - tail])
    return float(statistic(values)), float(low), float(high)


def mcnemar_test(correct_a, correct_b):
    """Exact McNemar test for two systems on the same questions; returns (only_a, only_b, p_value).

    Only the questions where exactly one system is right count: only_a (A right, B wrong) and only_b
    (A wrong, B right). If the systems were equally good, those would be about equally many; the
    p-value is the chance of a split at least this uneven if they were.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    if only_a + only_b == 0:  # the systems never disagree
        return only_a, only_b, 1.0
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue
    return only_a, only_b, float(p_value)


def cohens_kappa(labels_a, labels_b):
    """Agreement between two raters beyond what chance would give: 1 is perfect, 0 no better than chance.

    labels_a and labels_b hold one label per item, in the same order: the two annotators, or the judge
    and the final human labels. Plain agreement can look high just because most chunks are not
    relevant; kappa subtracts the agreement two raters would reach by guessing with the same shares of
    1s and 0s: kappa = (agreement - chance agreement) / (1 - chance agreement).
    """
    return float(cohen_kappa_score(np.asarray(labels_a, dtype=int), np.asarray(labels_b, dtype=int)))


def error_breakdown(qa, hits):
    """Each question as retrieval hit or miss x answer right or wrong: a 2x2 table of counts, with accuracy per row.

    qa: one QA file (qid, correct). hits: the retrieval_scores rows for one retriever (qid, hit).
    The rows split the questions by whether a relevant chunk was in that retriever's top k; the
    accuracy in each row shows how often the answer was right with and without one.
    """
    merged = qa[["qid", "correct"]].merge(hits[["qid", "hit"]], on="qid", validate="one_to_one")
    if len(merged) != len(qa):
        raise ValueError(f"{len(qa) - len(merged)} questions in qa have no hit value")
    retrieval = merged.hit.map({1.0: "relevant chunk in top k", 0.0: "no relevant chunk"})
    answer = merged.correct.map({True: "right", False: "wrong"})
    table = pd.crosstab(retrieval.rename("retrieval"), answer.rename("answer")).reindex(  # fixed order, 0 if empty
        index=["relevant chunk in top k", "no relevant chunk"], columns=["right", "wrong"], fill_value=0)
    table["accuracy"] = table["right"] / (table["right"] + table["wrong"])
    return table


def judge_confusion(human, judge):
    """The judge's labels against the human labels (1 = relevant, 0 = not) as a 2x2 table of counts.

    Rows are the human labels, columns the judge's. The diagonal is where they agree; "human: not
    relevant, judge: relevant" counts the times the judge is too generous, the opposite cell the times
    it is too strict.
    """
    names = ["not relevant", "relevant"]
    # labels=[0, 1] keeps both rows and columns even if one label never occurs
    table = confusion_matrix(np.asarray(human, dtype=int), np.asarray(judge, dtype=int), labels=[0, 1])
    return pd.DataFrame(table, index=[f"human: {name}" for name in names],
                        columns=[f"judge: {name}" for name in names])
