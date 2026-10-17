"""Run one experimental condition (model x retrieval setting) over a set of questions: retrieve, build the prompt, score the options and log the result.

This is the heart of the project: scripts/05_run_qa.py calls it for the full runs and the demo calls
it on 20 questions, so both run exactly the same method.

What goes here:
- run_condition(questions, model, tokenizer, condition, config, run=None, chunks=None) -> list of dicts.
  For each question it
    1. takes the top-k chunk_ids from the run file and looks up their texts (RAG conditions only),
    2. builds the prompt (prompts.py),
    3. scores every option (scoring.py),
    4. returns one record: qid, model, condition, prompt, option log-probs, predicted label and
       correct (True/False).
- Conditions: "no_rag" (Gemma or Borealis without context) and "rag_bm25", "rag_dense"
  (Gemma with the chunks from that retriever).

Used by: scripts/05_run_qa.py, demo/demo.ipynb.
"""
