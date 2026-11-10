"""LLM judge that labels whether a retrieved chunk lets you determine the correct answer to a question (evaluation layer 2).

The judge sees the question, its options, the correct answer and one chunk, and answers only "Y" or
"N", the same pattern as course notebook 4-2. It uses the same definition of "relevant" as our own
labels (annotations/README.md): our labels from layer 1 show how often it agrees with us, and it can
label far more chunks than we can by hand.

What goes here:
- JUDGE_PROMPT: the instruction text, with the definition and borderline cases from annotations/README.md.
- build_judge_prompt(question_row, chunk_text) -> str: JUDGE_PROMPT filled in for one (question, chunk) pair,
  with the options and the answer written as on the annotation sheets.
- judge_chunk(client, provider, model, prompt) -> dict: one API call; returns the raw reply and the
  parsed label (1, 0, or None if the reply is not Y or N).
- judge_many(...): loop over (qid, chunk_id) pairs, skip pairs already in the results file and append
  each judgement as it arrives, so a crash loses nothing. Each line holds qid, chunk_id, model, raw
  reply and label; the prompt template is saved once in the .meta.json, which keeps the file small
  enough to commit.

Uses the key in .env for the provider in the config (config.get_secret). Every call costs a little money.
Used by: scripts/06_run_judge.py.
Settings (`judge`): provider (anthropic or openai) and model.
"""

from norrag.prompts import format_options

# The definition and borderline cases are the ones in annotations/README.md, so the judge and we label the same thing
JUDGE_PROMPT = """You decide whether an excerpt from Norwegian Wikipedia answers a quiz question.

The excerpt is relevant if it states the correct answer to this question, or lets you work it out with a simple step and no outside knowledge. Judge from the excerpt alone, not from what you know about the topic. The title on the first line of the excerpt is part of it.
- The answer appears, but in another context: not relevant.
- The answer needs a simple step, such as a birth year for a question about age: relevant.
- The excerpt only rules out some of the options: not relevant.
- The answer is written differently, such as an abbreviation or the full name: relevant.
- The answer is only clear together with the title: relevant.

Question: {question}
Options:
{options}
Correct answer: {answer}

Excerpt:
{chunk}

Is the excerpt relevant? Answer with one letter: Y or N."""


def build_judge_prompt(question_row, chunk_text):
    """JUDGE_PROMPT filled in for one (question, chunk) pair, with the options and the answer as on the sheets."""
    return JUDGE_PROMPT.format(
        question=question_row.question,
        options=format_options(question_row.labels, question_row.options),
        answer=f"{question_row.answer}: {question_row.options[question_row.answer_index]}",
        chunk=chunk_text,
    )
