"""LLM judge that labels whether a retrieved chunk lets you determine the correct answer to a question (evaluation layer 2).

The judge sees the question, its options, the correct answer and one chunk, and answers only "Y" or
"N", the same pattern as course notebook 4-2 (with Anthropic, the API enforces this: JUDGE_SCHEMA). It uses the same definition of "relevant" as our own
labels (annotations/README.md): our labels from layer 1 show how often it agrees with us, and it can
label far more chunks than we can by hand.

What goes here:
- JUDGE_PROMPT: the instruction text, with the definition and borderline cases from annotations/README.md.
- JUDGE_SCHEMA: the only replies the Anthropic API lets the judge give, {"relevant": "Y"} or {"relevant": "N"}.
- build_judge_prompt(question_row, chunk_text) -> str: JUDGE_PROMPT filled in for one (question, chunk) pair,
  with the options and the answer written as on the annotation sheets.
- make_client(provider) -> client: an Anthropic or OpenAI client, with the key from .env.
- anthropic_request(settings, prompt) -> dict: the request for one judgement (model, effort, schema, no
  thinking), shared by single calls and batches so both judge in exactly the same way.
- to_label(letter) -> 1, 0 or None; read_reply(message) -> dict: the raw reply, the label and the token
  counts from one Anthropic reply, also shared by single calls and batches.
- judge_chunk(client, settings, prompt) -> dict: one API call with the `judge` settings from the
  config (provider, model, effort); returns the raw reply, the parsed label (1, 0, or None if the
  reply is not Y or N) and the token counts (for the cost).
  No automatic fallback to another model: every label comes from the one judge model, and a refusal
  is logged as "[refused]" with no label.
- pairs_to_judge(pairs, out_path) -> list: the pairs not in out_path yet.
- judge_many(client, settings, pairs, questions, chunk_texts, out_path) -> int: judge every
  (qid, chunk_id) pair in `pairs` one call at a time, skip pairs already in out_path and append each
  judgement as it arrives, so a crash loses nothing. Each line holds qid, chunk_id, model, raw reply,
  label and token counts; the prompt template is saved once in the .meta.json, which keeps the file
  small enough to commit. Skipping also means a later, larger run reuses the judgements already made.
- judge_batch(client, settings, pairs, questions, chunk_texts, out_path) -> Counter: the same with the
  Message Batches API (Anthropic only, half price, results within 24 hours), built from
  submit_batch (send and save the batch id), wait_for_batch (check every minute) and collect_batch
  (append the results). The lines it writes are the same as judge_many's, plus the batch_id.

Uses the key in .env for the provider in the config (config.get_secret). Every call costs a little money.
Used by: scripts/06_run_judge.py.
Settings (`judge`): provider (anthropic or openai), model and effort.
"""

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic
import openai
from tqdm import tqdm

from norrag.config import get_secret
from norrag.prompts import format_options
from norrag.records import done_ids, write_jsonl

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

# The only replies the API allows (Anthropic "structured outputs"): {"relevant": "Y"} or {"relevant": "N"}.
# In the first run on the 469 hand-labelled pairs (2026-11-13), 20 replies began an explanation instead of
# the letter and were cut off; the schema fixes the form of the answer and leaves the question unchanged
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "string", "enum": ["Y", "N"]}},
    "required": ["relevant"],
    "additionalProperties": False,
}


def build_judge_prompt(question_row, chunk_text):
    """JUDGE_PROMPT filled in for one (question, chunk) pair, with the options and the answer as on the sheets."""
    return JUDGE_PROMPT.format(
        question=question_row.question,
        options=format_options(question_row.labels, question_row.options),
        answer=f"{question_row.answer}: {question_row.options[question_row.answer_index]}",
        chunk=chunk_text,
    )


def make_client(provider):
    """An API client for the judge's provider, using its key from .env."""
    # max_retries=5: a busy server (error 429 or 5xx) is retried with growing pauses before the script stops
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"), max_retries=5)
    if provider == "openai":
        return openai.OpenAI(api_key=get_secret("OPENAI_API_KEY"), max_retries=5)
    raise ValueError(f"Unknown judge provider {provider!r}: use anthropic or openai")


def anthropic_request(settings, prompt):
    """The settings for one judgement on the Anthropic API: the same for single calls and for batches."""
    return {
        "model": settings["model"],
        "max_tokens": 16,  # caps the cost if anything goes wrong; {"relevant":"Y"} is 11 tokens
        # "between_tools" turns thinking off on Claude Sonnet 5.5 (other Claude models: {"type": "disabled"})
        "thinking": {"type": "between_tools"},
        # "format" allows only the replies in JUDGE_SCHEMA
        "output_config": {"effort": settings["effort"], "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        "messages": [{"role": "user", "content": prompt}],
    }


def to_label(letter):
    """1 for "Y", 0 for "N" and None for anything else."""
    return {"Y": 1, "N": 0}.get(letter.strip().rstrip(".").upper())


def read_reply(message):
    """The raw reply, the label (1, 0 or None) and the token counts from one Anthropic reply."""
    reply = "".join(block.text for block in message.content if block.type == "text")
    if message.stop_reason == "refusal":  # the model declined to answer
        reply = "[refused]"
    # Only a finished reply is guaranteed to be valid JSON; a refusal or a cut-off reply gets no letter
    letter = json.loads(reply)["relevant"] if message.stop_reason == "end_turn" else ""
    return {"reply": reply, "label": to_label(letter),
            "input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens}


def judge_chunk(client, settings, prompt):
    """Ask the judge about one (question, chunk) pair; returns the raw reply, the label (1, 0 or None) and the token counts."""
    if settings["provider"] == "anthropic":
        return read_reply(client.messages.create(**anthropic_request(settings, prompt)))
    # OpenAI: still free text, the letter is the whole reply
    response = client.chat.completions.create(model=settings["model"], messages=[{"role": "user", "content": prompt}])
    reply = response.choices[0].message.content or ""
    return {"reply": reply, "label": to_label(reply),
            "input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}


def pairs_to_judge(pairs, out_path):
    """The (qid, chunk_id) pairs in `pairs` that are not in out_path yet, in the same order."""
    done = done_ids(out_path, key=["qid", "chunk_id"])  # pairs judged by an earlier run
    todo = [(row.qid, row.chunk_id) for row in pairs.itertuples() if (row.qid, row.chunk_id) not in done]
    print(f"{len(pairs) - len(todo)} pairs already judged, {len(todo)} to go -> {out_path}")
    return todo


def judge_many(client, settings, pairs, questions, chunk_texts, out_path):
    """Judge each (qid, chunk_id) pair not yet in out_path, saving every judgement at once; returns how many were judged."""
    todo = pairs_to_judge(pairs, out_path)
    for qid, chunk_id in tqdm(todo, desc=f"judge {settings['model']}", unit=" chunks"):
        result = judge_chunk(client, settings, build_judge_prompt(questions.loc[qid], chunk_texts[chunk_id]))
        write_jsonl(out_path, [{"qid": qid, "chunk_id": chunk_id, "model": settings["model"], **result}], append=True)
    return len(todo)


def submit_batch(client, settings, todo, questions, chunk_texts, state_path):
    """Send every pair in todo as one Message Batch; save the batch id and which pair each request is to state_path."""
    # A custom_id may only hold letters, digits, "-" and "_" (a qid holds a "."), so the pairs are numbered
    pair_ids = {f"pair-{n:05d}": [qid, chunk_id] for n, (qid, chunk_id) in enumerate(todo)}
    requests = [{"custom_id": custom_id,
                 "params": anthropic_request(settings, build_judge_prompt(questions.loc[qid], chunk_texts[chunk_id]))}
                for custom_id, (qid, chunk_id) in pair_ids.items()]
    batch = client.messages.batches.create(requests=requests)
    state = {"batch_id": batch.id, "submitted": datetime.now().isoformat(timespec="seconds"), "pairs": pair_ids}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    print(f"Sent batch {batch.id} with {len(requests)} pairs; its id is saved in {state_path}")
    return state


def wait_for_batch(client, batch_id, poll_seconds=60):
    """Check the batch every poll_seconds until it has ended (most end within an hour, all within 24 hours)."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        failed = counts.errored + counts.expired + counts.canceled
        print(f"{datetime.now():%H:%M} {batch.processing_status}: {counts.succeeded} done, "
              f"{counts.processing} waiting, {failed} failed", flush=True)
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_seconds)


def collect_batch(client, settings, state, out_path):
    """Append every succeeded judgement in the batch to out_path; returns how many results of each type there were."""
    done = done_ids(out_path, key=["qid", "chunk_id"])  # if a collection was stopped halfway, nothing is written twice
    counts = Counter()  # succeeded, errored, expired or canceled
    rows = []
    for item in client.messages.batches.results(state["batch_id"]):  # in any order, matched through custom_id
        result_type = item.result.type
        counts[result_type] += 1
        qid, chunk_id = state["pairs"][item.custom_id]
        if result_type == "succeeded" and (qid, chunk_id) not in done:
            rows.append({"qid": qid, "chunk_id": chunk_id, "model": settings["model"],
                         **read_reply(item.result.message), "batch_id": state["batch_id"]})
        elif result_type == "errored" and counts[result_type] <= 3:  # show the first few errors
            print(f"Error for {item.custom_id} ({qid}, {chunk_id}): {item.result.error}")
    write_jsonl(out_path, rows, append=True)
    return counts


def judge_batch(client, settings, pairs, questions, chunk_texts, out_path, poll_seconds=60):
    """Judge the pairs not yet in out_path with the Message Batches API (half price): send, wait, collect.

    The batch id is kept in a .batch.json next to out_path until the results are saved. If the script is
    stopped while it waits, running it again picks up the same batch instead of paying for a new one.
    Pairs that failed are not saved, so running it again afterwards sends them in a new batch.
    """
    state_path = Path(out_path).with_suffix(".batch.json")
    if state_path.exists():  # a batch was sent earlier and its results are not saved yet
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        print(f"Picking up batch {state['batch_id']} ({len(state['pairs'])} pairs) sent {state['submitted']}")
    else:
        todo = pairs_to_judge(pairs, out_path)
        if not todo:
            return Counter()
        state = submit_batch(client, settings, todo, questions, chunk_texts, state_path)
    wait_for_batch(client, state["batch_id"], poll_seconds)
    counts = collect_batch(client, settings, state, out_path)
    state_path.unlink()  # the results are saved; the batch is finished with
    print(f"Saved {counts['succeeded']} judgements to {out_path}; results: {dict(counts)}")
    if counts["succeeded"] < len(state["pairs"]):
        print("Some pairs failed: run the script again to send them in a new batch")
    return counts
