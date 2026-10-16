# Data

Where the project's data comes from and how it is laid out; everything else in this folder is git-ignored and recreated with `scripts/01_prepare_data.py` and `scripts/02_chunk_corpus.py`.

## Sources

| Data | Source | Licence |
|---|---|---|
| Quiz questions, version v2.0: Bokmål (3,516) and Nynorsk (1,274) | NRK-Quiz-QA, [ltg/nrk_quiz_qa](https://huggingface.co/datasets/ltg/nrk_quiz_qa) on Hugging Face | MIT |
| Norwegian Wikipedia, Bokmål (617,937 articles) and Nynorsk (167,653), dump of 1 November 2023 | Wikimedia, via [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) on Hugging Face (configs `20231101.no` and `20231101.nn`) | CC BY-SA 3.0 and GFDL |

Both datasets are downloaded through Hugging Face, which caches them in `~/.cache/huggingface`; the versions used are pinned in `configs/experiment.yaml`.

## Layout after running the scripts

```
data/
├── processed/
│   ├── questions.parquet   one row per quiz question (01)
│   ├── articles.parquet    one row per cleaned Wikipedia article (01)
│   └── chunks.parquet      one row per chunk, with its chunk_id (02)
└── indexes/          one folder per retriever: tfidf/, bm25/, dense/<model>/ (03)
```
