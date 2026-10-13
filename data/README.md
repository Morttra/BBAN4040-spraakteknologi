# Data

Where the project's data comes from and how it is laid out; everything else in this folder is git-ignored and recreated with `scripts/01_prepare_data.py` and `scripts/02_chunk_corpus.py`.

## Sources

| Data | Source | Licence |
|---|---|---|
| Quiz questions, Bokmål (3,600) and Nynorsk (1,330) | NRK-Quiz-QA, [ltg/nrk_quiz_qa](https://huggingface.co/datasets/ltg/nrk_quiz_qa) on Hugging Face | MIT |
| Norwegian Wikipedia, Bokmål (492,864 articles) and Nynorsk (139,927), dump from about 20 March 2019 | Språkbanken, National Library of Norway: [Tekster fra norsk Wikipedia](https://data.norge.no/nb/datasets/23ae9838-2ad9-3d6f-8e61-4dfe0bb09376/tekster-fra-norsk-wikipedia) ([download](https://www.nb.no/sbfil/tekst/2019_wikipedia.tar.gz), [documentation](https://www.nb.no/sbfil/dok/2019_wikipedia.pdf)) | CC BY-SA 3.0 |

## Layout after running the scripts

```
data/
├── raw/              the downloaded Wikipedia archive and its unpacked JSON files (01)
├── processed/
│   ├── questions.parquet   one row per quiz question (01)
│   ├── articles.parquet    one row per cleaned Wikipedia article (01)
│   └── chunks.parquet      one row per chunk, with its chunk_id (02)
└── indexes/          one folder per retriever: tfidf/, bm25/, dense/<model>/ (03)
```
