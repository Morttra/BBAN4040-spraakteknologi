"""Load NRK-Quiz-QA (Bokmål and Nynorsk) from Hugging Face into a pandas DataFrame with one row per question.

What goes here:
- load_nrk_quiz(languages=("nb", "nn")) -> DataFrame with the columns
  qid, language, quiz, question, options (list of answer texts), labels (list of "A", "B", ...),
  answer (the correct label) and answer_index (its position in the lists).
  On Hugging Face (ltg/nrk_quiz_qa) each language is its own config ("nb", "nn") with one "test" split.

Note: the number of options varies between questions (2-5), so never assume there are four.
Used by: scripts/01_prepare_data.py.
"""

import pandas as pd
from datasets import load_dataset


def load_nrk_quiz(languages=("nb", "nn"), dataset="ltg/nrk_quiz_qa", revision=None):
    """Load the quiz questions for the given languages into one DataFrame, one row per question.

    revision pins a dataset version (e.g. "v2.0"); None means whatever is newest on Hugging Face.
    """
    rows = []
    for language in languages:
        for item in load_dataset(dataset, language, split="test", revision=revision):
            labels = item["choices"]["label"]
            rows.append({
                "qid": item["id"],
                "language": language,
                "quiz": item["quiz"],
                "question": item["question"],
                "options": item["choices"]["text"],
                "labels": labels,
                "answer": item["answer"],
                "answer_index": labels.index(item["answer"]),  # position of the correct label, e.g. "B" -> 1
            })
    return pd.DataFrame(rows)
