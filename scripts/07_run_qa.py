"""Score every question under each model x condition (no retrieval or RAG) and log the per-question results under results/raw/.

The experiment, with the 12B models: Borealis without context against Gemma with chunks from the
retriever chosen in the retrieval evaluation (the main comparison, fine-tuning vs RAG), and Gemma
without context as a reference.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/*.jsonl,
        the `qa` settings
Writes: results/raw/qa/<model>__<condition>.jsonl, one line per question (qid, chunk_ids, prompt,
        option log-probs, predicted, answer, correct), plus a .meta.json with the model id, revision,
        prompt texts and settings
Uses:   norrag.models, norrag.pipeline (which uses prompts and scoring), norrag.records
Run:    python scripts/07_run_qa.py --model borealis-12b --condition no_rag
        python scripts/07_run_qa.py --model gemma-12b --condition rag_bm25 --limit 200
                                                    a test on the first 200 questions
                                                    (saved as gemma-12b__rag_bm25_limit200.jsonl)

The longest step: one full condition can take hours with the 12B models (no_rag about 45 minutes,
RAG about 6 hours on the M1 Max). Each question is saved as soon as it is scored, and a restarted run
skips the questions already done, so stopping it (Ctrl+C) loses nothing.
"""

import argparse

import pandas as pd
from tqdm import tqdm

from norrag.config import get_path, load_config
from norrag.models import load_model, model_revision
from norrag.pipeline import run_condition
from norrag.prompts import CONTEXT_INTRO, INSTRUCTION, TEMPLATES
from norrag.records import done_ids, read_jsonl, write_jsonl, write_meta


def main():
    config = load_config()
    settings = config["qa"]
    parser = argparse.ArgumentParser(description="Score every question under one model x condition.")
    parser.add_argument("--model", required=True, choices=list(settings["models"]), help="a model name under qa.models")
    parser.add_argument("--condition", required=True, choices=["no_rag", "rag_bm25", "rag_dense"])
    parser.add_argument("--limit", type=int, help="use only the first N questions, for testing")
    args = parser.parse_args()

    questions = pd.read_parquet(get_path(config, "data_processed") / "questions.parquet")
    if args.limit:
        questions = questions.head(args.limit)
    file_name = f"{args.model}__{args.condition}" + (f"_limit{args.limit}" if args.limit else "") + ".jsonl"
    path = get_path(config, "results_raw") / "qa" / file_name
    todo = questions[~questions.qid.isin(done_ids(path))]  # skip questions saved by an earlier, stopped run
    print(f"{len(questions) - len(todo)} questions already done, {len(todo)} to go -> {path}")
    if todo.empty:
        return

    run, chunk_texts = None, None
    if args.condition != "no_rag":  # rag_bm25 -> results/raw/runs/bm25.jsonl
        run_path = get_path(config, "results_raw") / "runs" / f"{args.condition.removeprefix('rag_')}.jsonl"
        run = {r["qid"]: r["chunk_ids"][:settings["k"]] for r in read_jsonl(run_path)}
        needed = [chunk_id for qid in todo.qid for chunk_id in run[qid]]
        chunk_texts = pd.read_parquet(get_path(config, "data_processed") / "chunks.parquet", columns=["chunk_id", "text"],
                                      filters=[("chunk_id", "in", needed)]).set_index("chunk_id")["text"]

    spec = settings["models"][args.model]
    model, tokenizer = load_model(spec["id"], spec["revision"])
    write_meta(path, config, model=args.model, model_id=spec["id"], revision=model_revision(spec["id"], spec["revision"]),
               condition=args.condition, template=TEMPLATES[settings["prompt_template"]], instruction=INSTRUCTION,
               context_intro=CONTEXT_INTRO, n_questions=len(questions))
    records = run_condition(todo, model, tokenizer, args.model, args.condition, config, run=run, chunk_texts=chunk_texts)
    for record in tqdm(records, total=len(todo), desc=f"{args.model} {args.condition}", unit=" questions"):
        write_jsonl(path, [record], append=True)  # saved at once: a stopped run loses nothing


if __name__ == "__main__":
    main()
