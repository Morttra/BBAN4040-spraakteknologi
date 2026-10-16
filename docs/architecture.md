# How the project fits together

Step-by-step guide to how the code is organised, how the files connect and in which order to build them, written for first-time programmers.

Start with the flowchart in the [README](../README.md#how-the-pipeline-fits-together); this page explains it.

## The project in three stages

1. **Prepare** (scripts 01–03, run once): fetch the quiz questions and Wikipedia, cut Wikipedia into chunks and build one search index per retriever.
2. **Experiment** (04–05): find the top chunks for every question, then let each model score the answer options with and without those chunks.
3. **Evaluate** (06–07 and the notebooks): check how good the retrieved chunks are, using your own labels and the LLM judge, then turn all results into tables and figures.

Every step reads files that an earlier step wrote and writes new files for the next one. That is why each step can be run, and rerun, on its own.

## Four kinds of files

| Where | What it is | How you use it |
|---|---|---|
| `src/norrag/` | The toolbox: functions that each do one job, such as cutting a text into chunks | Imported, e.g. `from norrag.chunking import split_text` |
| `scripts/` | The pipeline steps: each one reads files, uses the toolbox and writes new files | Run in the terminal, e.g. `python scripts/02_chunk_corpus.py` |
| `notebooks/`, `demo/` | Looking at data and results, making tables and figures | Open in VS Code and run cell by cell |
| `configs/experiment.yaml` | Every setting and choice, in one place | Change a value, then rerun the steps that depend on it |

Why not do everything in notebooks? Some steps run for hours, and a script can be restarted where it stopped. Keeping the real work in `src/norrag/` also means the scripts, the demo and the tests all use exactly the same code.

## Each step: what it uses, reads and writes

Every file starts with a few lines saying what goes in it, so open the file to read the details. This table shows how the files connect. `config.py` is used by all of them to load the settings and find the folders.

| Step | Script | Uses from `src/norrag/` | Reads | Writes |
|---|---|---|---|---|
| 1 | `01_prepare_data.py` | `data/nrk_quiz.py`, `data/wikipedia.py` | Hugging Face (quiz and Wikipedia) | `data/processed/questions.parquet`, `articles.parquet` |
| 2 | `02_chunk_corpus.py` | `chunking.py` | `articles.parquet` | `data/processed/chunks.parquet` |
| 3 | `03_build_indexes.py` | `retrieval/` (`base`, `text`, `tfidf`, `bm25`, `dense`) | `chunks.parquet` | `data/indexes/<retriever>/` |
| 4 | `04_retrieve.py` | `retrieval/query.py`, `retrieval/base.py`, `records.py` | questions, indexes | `results/raw/runs/<retriever>.jsonl` |
| 5 | `05_run_qa.py` | `models.py`, `pipeline.py` (which uses `prompts.py` and `scoring.py`), `records.py` | questions, chunks, run files | `results/raw/qa/<model>__<condition>.jsonl` |
| 6 | `06_make_annotation_sample.py` | `annotation.py` | questions, chunks, run files | `annotations/sheet_<name>.csv`, `annotations/key.csv` |
| – | You label the sheets by hand | | your sheet | the same sheet, `relevant` column filled in |
| 7 | `07_run_judge.py` | `judge.py`, `records.py` | questions, chunks, run files, `key.csv` | `results/raw/judge/<judge model>.jsonl` |
| 8 | `08_make_demo_data.py` | `retrieval/` | questions, chunks, BM25 run file | `demo/data/` |
| – | `notebooks/00_main.ipynb` | `metrics.py`, `annotation.py`, `records.py` | everything in `results/raw/` and `annotations/` | `results/tables/`, `results/figures/` |

The other two notebooks only look at data: `01_data-exploration.ipynb` reads `data/processed/`, and `02_error-analysis.ipynb` reads the results and chunks to show examples of mistakes.

## Where your choices go

The choices you have not made yet are marked `DECISION` in `configs/experiment.yaml`. When you change one, rerun the steps that depend on it:

| Setting | Rerun |
|---|---|
| `chunking` (size, overlap, title) | 02, 03, 04, 05 |
| `retrieval.query_mode` | 04, 05 |
| `retrieval.dense_model` | 03 and 04 for the dense retriever, then 05 |
| `qa` (k, scoring, prompt template, chat template) | 05 |
| `annotation` | 06, then label again |
| `judge` | 07 |

Write each choice and the reason for it in `docs/decisions.md`. You need the reasons for M2 and for the method section.

## Recommended order of work

Build a thin version of the whole chain first, on a few hundred questions and the simplest retriever (BM25), before making any single part good. Problems in how the steps fit together show up early, and you have real numbers for M2 on 27 October.

The work splits into two tracks that you can do at the same time, one each, and then join.

**Step 0: together, first session**
1. Both of you: clone the repo, set up the environment (see the [Git cheatsheet](git-cheatsheet.md)) and check that `pip install -e ".[dev]"` works.
2. Write `config.py` and `records.py`. They are small, and everything else needs them.

**Track A: data and retrieval**
1. `data/nrk_quiz.py` and `data/wikipedia.py`, then `01_prepare_data.py`. Look at the result in `01_data-exploration.ipynb`.
2. `chunking.py` and `tests/test_chunking.py`, then `02_chunk_corpus.py`.
3. `retrieval/text.py`, `retrieval/base.py`, `retrieval/bm25.py`, `retrieval/query.py` and `tests/test_retrieval.py`, then `03` and `04` for BM25 only.

**Track B: models and scoring**
1. `models.py`: load `google/gemma-3-270m-it` in a notebook and check that it runs. It is small and quick to download.
2. `prompts.py` and `scoring.py`: score 20 questions in the notebook. The 270M model should land near chance and the 4B model clearly above it; if not, something is wrong.
3. `pipeline.py` with `no_rag` only, then `05_run_qa.py --limit 200` for Gemma 4B and Borealis 4B.

**Join: the first RAG run (before M2 on 27 October)**
1. `05_run_qa.py --condition rag_bm25 --limit 200`.
2. `metrics.accuracy` and a first table in `00_main.ipynb`: Gemma, Borealis and Gemma + BM25 on the same 200 questions. That is a solid M2 status.

**After M2**
1. `retrieval/tfidf.py` and `retrieval/dense.py` (sentence-BERT), then `03` and `04` for all three retrievers.
2. Write the annotation guideline in `annotations/README.md`, then `annotation.py` and `06`, and label your sheets.
3. `judge.py` and `07` on the hand-labelled sample, check the agreement, then run it on more chunks.
4. Full `05` runs overnight: 4B first, then 12B.

**Final weeks (deadline 18 November)**
1. Finish `metrics.py` and `tests/test_metrics.py`, so that `00_main.ipynb` produces every table and figure.
2. `02_error-analysis.ipynb` for the Discussion section.
3. `08_make_demo_data.py` and `demo/demo.ipynb`; time the demo on a laptop.
4. Fill in "Running the pipeline" and "Reproduction demo" in the README, and pin the package versions.

## How to try things yourself

- **Read the file headers.** `split_text(text, max_words, overlap_words) -> list of str` means a function called `split_text` that takes three inputs and gives back a list of strings. The names are suggestions: change them if you like, but update the header to match.
- **Try a function in a notebook before using it in a script.** With these lines in the first cell, the notebook picks up changes you save in `src/` without a restart:

  ```python
  %load_ext autoreload
  %autoreload 2
  from norrag.chunking import split_text
  split_text("Oslo er hovedstaden i Norge. Byen ligger innerst i Oslofjorden.", max_words=5, overlap_words=1)
  ```

- **Develop on small data.** Give every slow script a `--limit` option, so that `--limit 50` runs in seconds while you work on it.
- **Run the tests** with `pytest` from the repo folder after changing something in `src/`.
- **Seeing the flowchart in VS Code:** GitHub draws it automatically. VS Code's Markdown preview needs the extension "Markdown Preview Mermaid Support" first.
