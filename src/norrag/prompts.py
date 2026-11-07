"""Prompt templates for the multiple-choice questions, with and without retrieved Wikipedia context.

What goes here:
- TEMPLATES: named prompt templates for the question and its options. Keep old templates when you add
  new ones, so earlier results can still be explained from their saved template name. Every template
  ends with the start of the answer as its own paragraph ("\n\nSvar:"): with the chat template,
  scoring.py moves it to the start of the model's reply.
- INSTRUCTION: the instruction before every question, with or without chunks.
- CONTEXT_INTRO: the text around the retrieved chunks, put before the instruction in the RAG conditions.
- format_options(labels, options) -> str: the answer options as text, e.g. "A: 1905" on one line,
  "B: 1946" on the next, and so on.
- build_prompt(question_row, chunks=None, template="noreval_p3") -> str: the instruction and the
  filled template (question and options); for RAG, the chunk texts first.

The wording is identical for Gemma and Borealis, and only the context part differs between conditions.
Template "noreval_p3" is NorEval's prompt p3 for NRK-Quiz-QA, unchanged (lm-evaluation-harness,
lm_eval/tasks/noreval/nrk_quiz_qa/nob/utils.py); our instruction comes before it.
The wording was chosen with a format check on 50 questions: how much probability lands on the option
letters, never whether the answer is correct.
Used by: pipeline.py.
Settings (`qa`): prompt_template, and k, the number of chunks in the prompt.
"""

TEMPLATES = {
    "noreval_p3": "Spørsmål: {question}\n{options}\n\nSvar:",
}

# Before every question, with or without chunks. Without it, the models often answered in a full
# sentence or in bold instead of with a letter; without "bokstaven for", Borealis often wrote the
# option text instead of the letter (format check on 50 questions, 2026-11-07)
INSTRUCTION = (
    "Ett av alternativene er alltid riktig. "
    "Start svaret ditt med bokstaven for riktig alternativ før eventuell begrunnelse for valget.\n\n"
)

# RAG only: the chunks, then a sentence that runs straight into INSTRUCTION. When the chunks did not
# help, Gemma often refused ("Ingen av alternativene er nevnt i utdragene") instead of answering
CONTEXT_INTRO = (
    "Følgende utdrag fra Wikipedia kan være til hjelp, men er ikke alltid relevante.\n\n"
    "{context}\n\n"
    "Hvis utdragene ikke var relevante for spørsmålet, svar ut fra det du selv vet. "
)


def format_options(labels, options):
    """The answer options as text, one per line: "A: 1905", "B: 1946", and so on."""
    return "\n".join(f"{label}: {option}" for label, option in zip(labels, options))


def build_prompt(question_row, chunks=None, template="noreval_p3"):
    """The instruction and the filled template; for RAG, the chunk texts (best first) come before them."""
    prompt = INSTRUCTION + TEMPLATES[template].format(
        question=question_row.question,
        options=format_options(question_row.labels, question_row.options),
    )
    if chunks:  # RAG: number the chunks [1], [2], ... and put them before the instruction
        context = "\n\n".join(f"[{number}] {text}" for number, text in enumerate(chunks, start=1))
        prompt = CONTEXT_INTRO.format(context=context) + prompt
    return prompt
