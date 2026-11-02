"""Prompt templates for the multiple-choice questions, with and without retrieved Wikipedia context.

What goes here:
- TEMPLATES: a dictionary of named prompt templates, e.g. one without context and one that starts
  with the retrieved chunks. Keep old templates when you add new ones, so earlier results can still
  be explained from their saved template name.
- format_options(labels, options) -> str: the answer options as text, e.g. "A: 1905" on one line,
  "B: 1946" on the next, and so on.
- build_prompt(question_row, chunks=None, template="...") -> str: fill a template with the question,
  its options and, for RAG, the chunk texts.

The wording must be identical for Gemma and Borealis; only the context part differs between conditions.
NorEval's own NRK-Quiz-QA prompts (lm-evaluation-harness, lm_eval/tasks/noreval/nrk_quiz_qa/*/utils.py)
are a good starting point; credit them if you reuse them.
Used by: pipeline.py.
Decisions (`qa`): the template wording, how the chunks
are introduced, and k, the number of chunks in the prompt.
"""
