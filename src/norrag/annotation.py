"""Build pooled, blinded annotation sheets from the retrieval runs and read the human relevance labels back (evaluation layer 1).

How layer 1 works: draw a sample of questions, collect the top chunks from ALL retrievers ("pooling"),
remove duplicates, shuffle them and hide which retriever found each chunk ("blind"). Each of you then
labels every (question, chunk) pair as relevant (1) or not (0), following annotations/README.md.

What goes here:
- sample_questions(questions, n, seed) -> DataFrame: a random sample with at most one question per quiz,
  so the n questions cover n different quiz themes.
- pool_chunks(runs, qids, k) -> DataFrame: every chunk that is in some retriever's top k, once per
  question, with the retrievers that found it (columns qid, chunk_id, retrievers). `runs` holds the
  lines of all run files together (columns qid, retriever, chunk_ids).
- write_sheets(pool, questions, chunk_texts, out_dir, annotators, seed): one blind Excel sheet per
  annotator to fill in (item_id, question, options, correct answer, chunk text, empty `relevant` and
  `comment` columns) and a separate key.csv (item_id -> qid, chunk_id and the retrievers that found it).
  Never overwrites: a sheet may already hold hours of labels.
- save_sheet(sheet, path): save a sheet as .xlsx with wrapped text and fixed column widths. Excel, not
  CSV, because Norwegian Excel expects semicolons in CSV files and can garble æøå.
- load_labels(sheet_paths, key_path) -> DataFrame: the filled-in sheets, one row per (annotator, item),
  joined back to qid, chunk_id and retrievers through the key. Stops with a message naming the rows
  if a label is missing or not 1 or 0, or if rows were deleted or added.
- final_labels(labels, resolved_path) -> DataFrame: one label per item (columns item_id, qid, chunk_id,
  retrievers, relevant): the shared label where the annotators agree, the decision in
  annotations/resolved.csv where they don't. Kappa uses `labels`; the retriever scores use these.

Used by: scripts/05_make_annotation_sample.py, notebooks/00_main.ipynb.
"""

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment

from norrag.prompts import format_options

# Column widths in the Excel sheets, in characters
COLUMN_WIDTHS = {"item_id": 8, "question": 45, "options": 35, "answer": 25, "chunk": 90, "relevant": 10, "comment": 30}


def sample_questions(questions, n, seed):
    """A random sample of n questions, at most one from each quiz."""
    shuffled = questions.sample(frac=1, random_state=seed)  # all questions in random order
    # Keep the first question from each quiz; quizzes with many questions are still more likely to be picked
    return shuffled.drop_duplicates("quiz").head(n)


def pool_chunks(runs, qids, k):
    """Every (question, chunk) pair in at least one retriever's top k, once, with the retrievers that found it."""
    runs = runs[runs.qid.isin(qids)]
    # One row per (retriever, question, chunk): cut each list to the top k, then give each chunk its own row
    top_k = runs.assign(chunk_id=runs.chunk_ids.str[:k]).explode("chunk_id")
    # A chunk found by both retrievers becomes one row, so it is labelled once
    return top_k.groupby(["qid", "chunk_id"], sort=False)["retriever"].agg(list).reset_index(name="retrievers")


def write_sheets(pool, questions, chunk_texts, out_dir, annotators, seed):
    """Write one blind Excel sheet per annotator, and the key that links each row to its question and chunk."""
    out_dir = Path(out_dir)
    sheet_paths = [out_dir / f"sheet_{name}.xlsx" for name in annotators]
    key_path = out_dir / "key.csv"
    existing = [str(path) for path in [*sheet_paths, key_path] if path.exists()]
    if existing:
        raise FileExistsError(f"Already there; move or delete them first if you really want new sheets: {existing}")

    # Shuffle, then put each question's chunks back together (a stable sort keeps the shuffled order
    # within a question): the order no longer shows which retriever found a chunk, or how high it ranked
    pool = pool.sample(frac=1, random_state=seed).sort_values("qid", kind="stable")
    rows = pool.assign(item_id=range(1, len(pool) + 1)).merge(questions, on="qid", how="left")
    sheet = pd.DataFrame({
        "item_id": rows.item_id,
        "question": rows.question,
        "options": [format_options(labels, options) for labels, options in zip(rows.labels, rows.options)],
        "answer": [f"{letter}: {options[i]}" for letter, options, i in zip(rows.answer, rows.options, rows.answer_index)],
        "chunk": rows.chunk_id.map(chunk_texts),
        "relevant": None,  # to be filled in: 1 or 0
        "comment": None,
    })
    for path in sheet_paths:  # the same sheet for each of you
        save_sheet(sheet, path)
    key = rows[["item_id", "qid", "chunk_id"]].assign(retrievers=rows.retrievers.str.join("+"))
    key.to_csv(key_path, index=False)


def save_sheet(sheet, path):
    """Save a sheet as .xlsx with wrapped text and fixed column widths, so each chunk reads as a paragraph."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="labels", index=False)
        worksheet = writer.sheets["labels"]
        worksheet.freeze_panes = "A2"  # the header row stays in view while you scroll
        for cells, width in zip(worksheet.columns, COLUMN_WIDTHS.values()):
            worksheet.column_dimensions[cells[0].column_letter].width = width
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")


def load_labels(sheet_paths, key_path):
    """Read the filled-in sheets back, one row per (annotator, item), joined to qid and chunk_id through the key."""
    key = pd.read_csv(key_path)
    labels = []
    for path in map(Path, sheet_paths):
        annotator = path.stem.removeprefix("sheet_")  # sheet_a.xlsx -> "a"
        sheet = pd.read_excel(path, usecols=["item_id", "relevant", "comment"])
        # Joining on item_id, not on row order, so sorting the sheet in Excel does no harm
        if sorted(sheet.item_id) != sorted(key.item_id):
            raise ValueError(f"Sheet {annotator}: rows are missing or added compared with key.csv")
        sheet["relevant"] = pd.to_numeric(sheet.relevant, errors="coerce")  # text such as "ja" becomes empty
        bad = sheet.item_id[~sheet.relevant.isin([0, 1])].tolist()
        if bad:
            raise ValueError(f"Sheet {annotator}: {len(bad)} rows without 1 or 0 in `relevant`, item_id {bad[:10]}")
        labels.append(sheet.assign(annotator=annotator, relevant=sheet.relevant.astype(int)))
    columns = ["annotator", "item_id", "qid", "chunk_id", "retrievers", "relevant", "comment"]
    return pd.concat(labels).merge(key, on="item_id")[columns]


def final_labels(labels, resolved_path):
    """One label per item: the shared label where the annotators agree, the decision in resolved.csv where they don't."""
    by_annotator = labels.pivot(index="item_id", columns="annotator", values="relevant")  # one column per annotator
    agreed = by_annotator.nunique(axis=1) == 1  # True where everyone gave the same label
    resolved = pd.read_csv(resolved_path).set_index("item_id")["relevant"]
    missing = sorted(set(by_annotator.index[~agreed]) - set(resolved.index))
    if missing:
        raise ValueError(f"No decision in {resolved_path} for the disagreements on item_id {missing}")
    final = by_annotator.iloc[:, 0].where(agreed, resolved).astype(int)  # where they disagree, take the decision
    items = labels.drop_duplicates("item_id").set_index("item_id")[["qid", "chunk_id", "retrievers"]]
    return items.assign(relevant=final).reset_index()
