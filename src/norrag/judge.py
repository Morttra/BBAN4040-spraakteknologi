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
- judge_chunk(client, settings, prompt) -> dict: one API call with the `judge` settings from the
  config (provider, model, effort); returns the raw reply, the parsed label (1, 0, or None if the
  reply is not Y or N) and the token counts (for the cost).
  No automatic fallback to another model: every label comes from the one judge model, and a refusal
  is logged as "[refused]" with no label.
- judge_many(client, settings, pairs, questions, chunk_texts, out_path) -> int: judge every
  (qid, chunk_id) pair in `pairs` one call at a time, skip pairs already in out_path and append each
  judgement as it arrives, so a crash loses nothing. Each line holds qid, chunk_id, model, raw reply,
  label and token counts; the prompt template is saved once in the .meta.json, which keeps the file
  small enough to commit. Skipping also means a later, larger run reuses the judgements already made.

Uses the key in .env for the provider in the config (config.get_secret). Every call costs a little money.
Used by: scripts/06_run_judge.py.
Settings (`judge`): provider (anthropic or openai), model and effort.
"""

import json

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


def judge_chunk(client, settings, prompt):
    """Ask the judge about one (question, chunk) pair; returns the raw reply, the label (1, 0 or None) and the token counts."""
    messages = [{"role": "user", "content": prompt}]
    if settings["provider"] == "anthropic":
        # "between_tools" turns thinking off on Claude Sonnet 5.5 (other Claude models: {"type": "disabled"});
        # "format" allows only the replies in JUDGE_SCHEMA; max_tokens=16 caps the cost if anything goes wrong
        output_config = {"effort": settings["effort"], "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}}
        response = client.messages.create(model=settings["model"], max_tokens=16, thinking={"type": "between_tools"},
                                          output_config=output_config, messages=messages)
        reply = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "refusal":  # the model declined to answer
            reply = "[refused]"
        # Only a finished reply is guaranteed to be valid JSON; a refusal or a cut-off reply gets no letter
        letter = json.loads(reply)["relevant"] if response.stop_reason == "end_turn" else ""
        tokens = (response.usage.input_tokens, response.usage.output_tokens)
    else:  # OpenAI: still free text, the letter is the whole reply
        response = client.chat.completions.create(model=settings["model"], messages=messages)
        reply = letter = response.choices[0].message.content or ""
        tokens = (response.usage.prompt_tokens, response.usage.completion_tokens)
    label = {"Y": 1, "N": 0}.get(letter.strip().rstrip(".").upper())  # anything but Y or N gives None
    return {"reply": reply, "label": label, "input_tokens": tokens[0], "output_tokens": tokens[1]}


def judge_many(client, settings, pairs, questions, chunk_texts, out_path):
    """Judge each (qid, chunk_id) pair not yet in out_path, saving every judgement at once; returns how many were judged."""
    done = done_ids(out_path, key=["qid", "chunk_id"])  # pairs judged by an earlier run
    todo = [(row.qid, row.chunk_id) for row in pairs.itertuples() if (row.qid, row.chunk_id) not in done]
    print(f"{len(pairs) - len(todo)} pairs already judged, {len(todo)} to go -> {out_path}")
    for qid, chunk_id in tqdm(todo, desc=f"judge {settings['model']}", unit=" chunks"):
        result = judge_chunk(client, settings, build_judge_prompt(questions.loc[qid], chunk_texts[chunk_id]))
        write_jsonl(out_path, [{"qid": qid, "chunk_id": chunk_id, "model": settings["model"], **result}], append=True)
    return len(todo)
