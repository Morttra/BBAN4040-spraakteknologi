# Data

Where the project's data comes from and how it is laid out; everything else in this folder is git-ignored and recreated with `scripts/01_prepare_data.py` and `scripts/02_chunk_corpus.py`.

## Sources

| Data | Source | Licence |
|---|---|---|
| Quiz questions, version v2.0: the 3,516 Bokmål questions (the 1,274 Nynorsk questions are not used) | NRK-Quiz-QA, [ltg/nrk_quiz_qa](https://huggingface.co/datasets/ltg/nrk_quiz_qa) on Hugging Face, pinned to revision `v2.0` | MIT |
| Norwegian Wikipedia, Bokmål (`nowiki`), articles only, as of 20 March 2025 | [Wikimedia Enterprise HTML dumps](https://dumps.wikimedia.org/other/enterprise_html/), run `20250320`: `nowiki-NS0-20250320-ENTERPRISE-HTML.json.tar.gz` (6.2 GB) | CC BY-SA 3.0 (stated in each record) |

The exact URL and the quiz version are in `configs/experiment.yaml`. The project uses Bokmål only (`data.languages`); the code also handles Nynorsk.

## Why this Wikipedia dump

| Dump considered | Problem |
|---|---|
| Språkbanken, March 2019 | About 350 of the Bokmål quiz questions (10 %) mention 2019 or later |
| Hugging Face `wikimedia/wikipedia`, November 2023 | Built from the raw wiki markup with templates removed, so numbers written with templates are missing (e.g. "Galdhøpiggen har en høyde på meter over havet"); visible gaps in 1.8 % of Bokmål articles |
| **Wikimedia Enterprise HTML, March 2025 (used)** | None of the above: it holds the rendered pages, so template numbers are included, and it covers the quiz period. It is 16 months newer than the quiz's last year, so a few "per 2019"-style facts may have been updated since. |

## How a page becomes text

Each dump record is one page as JSON. The text is taken from its rendered HTML (`article_body.html`) by `html_to_text` and `clean_text` in `src/norrag/data/wikipedia.py`:

| Part of the page | Rule | Reason |
|---|---|---|
| Paragraphs, headings (`h2`–`h5`), list items, definition lists | Kept, one per line | The article's text; line breaks keep paragraphs apart for chunking |
| Fact box (`table.infobox`) | Kept as `label: value` lines, e.g. `Folketal: 10 535 (2023)` | Short, structured facts that quizzes often ask about |
| All other tables | Removed | Turned into one line they become number soup that confuses retrieval |
| Footnote markers (`[1]`), footnote lists, navigation boxes, maintenance notices, images and captions, style and script blocks | Removed | Not article text |
| Pointer notes above the first paragraph ("For kommunen, sjå Førde kommune") | Removed | Navigation, not content |
| Headings with no text under them (e.g. a section that only held a table) | Removed | Say nothing on their own |
| Everything from the first trailing heading: Se også, Referanser, Litteratur, Kilder, Eksterne lenker, Fotnoter, Noter (Nynorsk: Sjå òg, Kjelder, Referansar, Bakgrunnsstoff, Fotnotar, Notar, Merknader, Merknadar) | Removed | Sources and links, not facts |
| Articles with no text left | Removed | |
| Pages whose record has no HTML, only raw wiki markup (11 of 659,026) | Skipped and counted; the count is printed by script 01 | A separate cleaning path for a handful of pages is not worth it |
| A page that appears twice | The most recently edited copy is kept | `article_id` must be unique |

Short and long articles are all kept.

## Columns in `articles.parquet`

| Column | Example | Meaning |
|---|---|---|
| `article_id` | `nb-99985` | Language + Wikipedia's page id (page ids are only unique within one language) |
| `language` | `nb` | `nb` (Bokmål) |
| `title` | `Fylkesvei 405` | The article's title |
| `url` | `https://no.wikipedia.org/wiki/Fylkesvei_405` | The original page |
| `text` | | The cleaned text, one block per line |
| `n_words` | `73` | Number of words in `text` |
| `wikidata` | `Q6362968` | Wikidata id: the same subject has the same id in every language edition (missing for some pages) |
| `modified` | `2024-09-13` | Date of the article's last edit before the dump |

## Layout after running the scripts

```
data/
├── raw/              the Wikipedia dump file, 6.2 GB (01)
├── processed/
│   ├── questions.parquet   one row per quiz question (01)
│   ├── articles.parquet    one row per cleaned Wikipedia article (01)
│   └── chunks.parquet      one row per chunk, with its chunk_id (02)
└── indexes/          one folder per retriever: bm25/, dense/<model>/ (03)
```
