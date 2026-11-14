"""Copy a small subset of questions and chunks into demo/data/ for the reproduction demo.

The demo must run in about ten minutes on an ordinary laptop, so it gets about 20 questions and only
the chunks it needs: the top chunks the retriever found for each demo question on the full collection,
plus a random sample of others, so that the search still has something to search through. The demo
builds its own BM25 index from these chunks (about a second), so no index is stored. The report's
answers from the 12B models to the same questions are copied too, so the demo can show them next to
its own and runs without anything outside demo/data/.

Reads:  data/processed/questions.parquet, data/processed/chunks.parquet, results/raw/runs/bm25.jsonl,
        results/raw/qa/ (the three conditions in the report), the `demo` settings
Writes: demo/data/questions.parquet, demo/data/chunks.parquet, demo/data/results_12b.parquet and
        demo/data/questions.meta.json (committed to Git, so keep it small)
Uses:   norrag.records
Run:    python scripts/08_make_demo_data.py      (a few seconds)
"""

import pandas as pd

from norrag.config import get_path, load_config
from norrag.records import read_jsonl, read_jsonl_df, write_meta


def main():
    config = load_config()
    settings = config["demo"]
    chunks_path = get_path(config, "data_processed") / "chunks.parquet"
    out_dir = get_path(config, "demo_data")

    # Random questions, the same every time (fixed seed)
    questions = pd.read_parquet(get_path(config, "data_processed") / "questions.parquet")
    demo_questions = questions.sample(settings["n_questions"], random_state=config["seed"]).sort_values("qid")

    # Every chunk the retriever saved for these questions on the full collection (retrieval.top_n per question)
    run_path = get_path(config, "results_raw") / "runs" / f"{settings['retriever']}.jsonl"
    run = {row["qid"]: row["chunk_ids"] for row in read_jsonl(run_path)}
    retrieved = {chunk_id for qid in demo_questions.qid for chunk_id in run[qid]}

    # Plus a random sample of all the other chunks
    all_ids = pd.read_parquet(chunks_path, columns=["chunk_id"]).chunk_id
    others = all_ids[~all_ids.isin(retrieved)].sample(settings["n_other_chunks"], random_state=config["seed"])
    keep = sorted(retrieved | set(others))
    chunks = pd.read_parquet(chunks_path, columns=["chunk_id", "title", "text"], filters=[("chunk_id", "in", keep)])

    # The report's answers to the same questions: the three conditions compared in 00_main.ipynb
    qa_dir = get_path(config, "results_raw") / "qa"
    files = ["borealis-12b__no_rag", "gemma-12b__no_rag", f"gemma-12b__rag_{settings['retriever']}"]
    results_12b = pd.concat([read_jsonl_df(qa_dir / f"{name}.jsonl") for name in files])
    columns = ["qid", "model", "condition", "chunk_ids", "logprobs", "predicted", "answer", "correct"]
    results_12b = results_12b[results_12b.qid.isin(demo_questions.qid)][columns]

    demo_questions.to_parquet(out_dir / "questions.parquet", index=False)
    chunks.to_parquet(out_dir / "chunks.parquet", index=False)
    results_12b.to_parquet(out_dir / "results_12b.parquet", index=False)
    write_meta(out_dir / "questions.parquet", config, n_questions=len(demo_questions),
               n_retrieved_chunks=len(retrieved), n_other_chunks=len(others), results_12b=files)
    names = ["questions.parquet", "chunks.parquet", "results_12b.parquet"]
    size_mb = sum((out_dir / name).stat().st_size for name in names) / 1e6
    print(f"{len(demo_questions)} questions, {len(chunks)} chunks ({len(retrieved)} retrieved, {len(others)} others) "
          f"and {len(results_12b)} 12B answers -> {out_dir}, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
