# BBAN4040-spraakteknologi
Course project in the course BBAN4040 Språkteknologi at NTNU Handelshøyskolen.

We compare fine-tuning and retrieval as two ways of making a language model better at questions about Norwegian history and culture (the Bokmål part of NRK-Quiz-QA): Borealis 12B (Gemma 3 fine-tuned on Norwegian instructions) without retrieval against Gemma 3 12B with retrieval over Bokmål Wikipedia (RAG). Gemma 3 12B without retrieval is included as a reference.

## Project structure

```
├── README.md                    This file: overview, setup and how to run everything
├── pyproject.toml               Package metadata and dependencies
├── .env.example                 Template for API keys (copy to .env, never commit .env)
├── configs/
│   └── experiment.yaml          Central experiment settings read by scripts and notebooks
├── src/norrag/                  Reusable code, imported by scripts and notebooks
│   ├── config.py                Settings, paths and secrets
│   ├── records.py               JSONL logging of runs with model, prompt and settings
│   ├── data/                    Loaders for NRK-Quiz-QA and Norwegian Wikipedia
│   ├── chunking.py              Wikipedia articles → chunks with stable chunk_ids
│   ├── retrieval/               BM25 and dense retrievers, query building
│   ├── prompts.py               Prompt templates with and without context
│   ├── models.py                Loading Gemma 3 and Borealis locally
│   ├── scoring.py               Log-likelihood multiple-choice scoring
│   ├── pipeline.py              Runs one condition end to end
│   ├── annotation.py            Evaluation layer 1: blind human annotation sheets
│   ├── judge.py                 Evaluation layer 2: LLM judge
│   └── metrics.py               Accuracy, hit@k, MRR@k, confidence intervals, McNemar, kappa, error breakdown
├── scripts/                     Long-running pipeline steps, run in order 01 → 08
├── notebooks/
│   ├── 00_main.ipynb            MAIN NOTEBOOK: all results in the report
│   ├── 01_data-exploration.ipynb
│   └── 02_error-analysis.ipynb
├── demo/                        Reproduction demo and its small committed data subset
├── data/                        Downloaded and processed data (git-ignored)
├── annotations/                 Human relevance labels and guidelines (committed)
├── results/                     raw/ logs (git-ignored), tables/ and figures/ (committed)
├── tests/                       Unit tests (pytest)
└── docs/                        Architecture guide, decision log, AI-use log and Git cheatsheet
```

## How the pipeline fits together

Blue boxes are scripts you run, grey shapes are files they write, green boxes are notebooks and the hexagon is manual work. Each script reads what the earlier ones wrote, so they run in order and can be rerun one at a time. [docs/architecture.md](docs/architecture.md) explains every step, what each file does and the order to build them in.

```mermaid
flowchart TD
    NRK[("NRK-Quiz-QA<br/>Hugging Face")]
    WIKI[("Bokmål Wikipedia, March 2025<br/>Wikimedia HTML dump")]

    NRK --> S01
    WIKI --> S01
    S01["01_prepare_data.py"] --> Q[/"questions.parquet"/]
    S01 --> A[/"articles.parquet"/]
    A --> S02["02_chunk_corpus.py"] --> C[/"chunks.parquet"/]
    C --> S03["03_build_indexes.py"] --> I[/"indexes: BM25, sentence-BERT"/]
    Q --> S04["04_retrieve.py"]
    I --> S04
    S04 --> R[/"run files: top-20 chunks per question"/]

    R --> S05["05_make_annotation_sample.py"] --> SH[/"blind annotation sheets"/]
    SH --> H{{"we label the chunks"}} --> L[/"human labels"/]
    R --> S06["06_run_judge.py"] --> J[/"LLM-judge labels"/]
    L --> CH{{"we choose the retriever"}}
    J --> CH

    Q --> S07["07_run_qa.py<br/>Borealis and Gemma 12B without chunks,<br/>Gemma with the chosen retriever"]
    C --> S07
    R --> S07
    CH --> S07
    S07 --> QA[/"QA results"/]

    QA --> M["00_main.ipynb"]
    R --> M
    L --> M
    J --> M
    M --> OUT[/"tables and figures for the report"/]

    R --> S08["08_make_demo_data.py"] --> D[/"demo/data"/] --> DEMO["demo.ipynb"]

    classDef script fill:#dbeafe,stroke:#2563eb,color:#0f172a
    classDef file fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef notebook fill:#dcfce7,stroke:#16a34a,color:#052e16
    classDef manual fill:#fef3c7,stroke:#d97706,color:#451a03
    class S01,S02,S03,S04,S05,S06,S07,S08 script
    class Q,A,C,I,R,QA,SH,L,J,OUT,D file
    class M,DEMO notebook
    class H,CH manual
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in the keys
```

## Running the pipeline

TODO

## Reproduction demo

TODO

## Data and licences

- NRK-Quiz-QA v2.0 ([ltg/nrk_quiz_qa](https://huggingface.co/datasets/ltg/nrk_quiz_qa)), Bokmål questions only, MIT licence.
- Norwegian Wikipedia (Bokmål) as of 20 March 2025, from the [Wikimedia Enterprise HTML dumps](https://dumps.wikimedia.org/other/enterprise_html/), CC BY-SA 3.0. See [data/README.md](data/README.md) for why this dump and how the pages are turned into text.
