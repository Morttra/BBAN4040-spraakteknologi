# Demo data

Small committed subset used by the reproduction demo (`demo/demo.ipynb`), so it runs without the full data or any of the scripts. Created by `scripts/08_make_demo_data.py`.

- `questions.parquet`: 100 random NRK-Quiz-QA questions (seed 42), in the same format as `data/processed/questions.parquet`
- `chunks.parquet`: the top 20 BM25 chunks for each of these questions on the full Wikipedia collection, plus 5,000 random other chunks; the demo builds its own BM25 index from them
- `results_12b.parquet`: the report's answers from the 12B models to the same questions (Borealis and Gemma without retrieval, Gemma with BM25)
- `questions.meta.json`: when and from which commit the files were made, with the settings

Licences: Wikipedia text is CC BY-SA 3.0, NRK-Quiz-QA is MIT-licensed.
