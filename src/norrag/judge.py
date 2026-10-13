"""LLM judge that labels whether a retrieved chunk lets you determine the correct answer to a question (evaluation layer 2).

The judge sees the question, the correct answer and one chunk, and answers only "Y" or "N", the same
pattern as course notebook 4-2. It can label far more chunks than we can by hand; our own labels from
layer 1 show how often it agrees with us.

What goes here:
- JUDGE_PROMPT: the instruction text, using the same definition of "relevant" as annotations/README.md.
- judge_chunk(client, model, question, answer, chunk_text) -> dict: one API call with temperature 0;
  returns the raw reply and the parsed label (1, 0, or None if the reply is not Y or N).
- judge_many(...): loop over (qid, chunk_id) pairs, skip pairs already in the results file
  (records.done_ids) and append each judgement as it arrives, so a crash loses nothing.

Uses the OpenAI key in .env (config.get_secret). Every call costs a little money.
Used by: scripts/07_run_judge.py.
Decision (`judge`): which judge model; not a Gemma model, so it does not judge its own family.
"""
