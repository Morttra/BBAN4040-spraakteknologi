"""Run one experimental condition (model x retrieval setting) over a set of questions: retrieve, build the prompt, score the options and log the result.

This is the heart of the project: scripts/07_run_qa.py calls it for the full runs and the demo calls
it on 100 questions, so both run exactly the same method.

What goes here:
- run_condition(questions, model, tokenizer, model_name, condition, config, run=None, chunk_texts=None)
  yields one record per question, as soon as it is scored, so a long run can save each one at once.
  For each question it
    1. takes the top-k chunk_ids from the run (RAG conditions only; `run` maps qid -> chunk_ids,
       `chunk_texts` maps chunk_id -> text),
    2. builds the prompt (prompts.py),
    3. scores every option letter (scoring.py),
    4. yields one record: qid, model, condition, chunk_ids, prompt, the option log-probs, the
       predicted letter, the correct letter and correct (True/False).
- Conditions: "no_rag" (Gemma or Borealis without context) and "rag_bm25", "rag_dense"
  (Gemma with the chunks from that retriever).

Used by: scripts/07_run_qa.py, demo/demo.ipynb.
Settings (`qa`): k, prompt_template and chat_template.
"""

from norrag.prompts import build_prompt
from norrag.scoring import option_logprobs, predict


def run_condition(questions, model, tokenizer, model_name, condition, config, run=None, chunk_texts=None):
    """Score every question under one condition, yielding one record per question as soon as it is done."""
    settings = config["qa"]
    for row in questions.itertuples():
        chunk_ids = [] if condition == "no_rag" else list(run[row.qid][:settings["k"]])
        chunks = [chunk_texts[chunk_id] for chunk_id in chunk_ids] or None  # None: no context in the prompt
        prompt = build_prompt(row, chunks=chunks, template=settings["prompt_template"])
        logprobs = option_logprobs(model, tokenizer, prompt, row.labels, chat_template=settings["chat_template"])
        predicted = predict(logprobs)
        yield {
            "qid": row.qid,
            "model": model_name,
            "condition": condition,
            "chunk_ids": chunk_ids,
            "prompt": prompt,
            "logprobs": logprobs,  # one per option, in the order A, B, C, ...
            "predicted": str(row.labels[predicted]),
            "answer": str(row.answer),
            "correct": bool(predicted == row.answer_index),  # bool(): json cannot save numpy's True/False
        }
