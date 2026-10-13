"""Load NRK-Quiz-QA (Bokmål and Nynorsk) from Hugging Face into a pandas DataFrame with one row per question.

What goes here:
- load_nrk_quiz(languages=("nb", "nn")) -> DataFrame with the columns
  qid, language, quiz, question, options (list of answer texts), labels (list of "A", "B", ...),
  answer (the correct label) and answer_index (its position in the lists).
  On Hugging Face (ltg/nrk_quiz_qa) each language is its own config ("nb", "nn") with one "test" split.

Note: the number of options varies between questions (2-5), so never assume there are four.
Used by: scripts/01_prepare_data.py.
"""
