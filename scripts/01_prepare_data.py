"""Download NRK-Quiz-QA and Norwegian Wikipedia (nb + nn) and save cleaned copies under data/processed/.

Steps:
1. Load the NRK-Quiz-QA questions (norrag.data.nrk_quiz) and save them.
2. Load the Bokmål and Nynorsk Wikipedia (norrag.data.wikipedia), clean the text and save them as one table.

Reads:  Hugging Face: ltg/nrk_quiz_qa (v2.0) and wikimedia/wikipedia (20231101.no, 20231101.nn)
Writes: data/processed/questions.parquet, data/processed/articles.parquet
Run:    python scripts/01_prepare_data.py      (a few minutes; the first run downloads about 730 MB)
"""

import pandas as pd

from norrag.config import get_path, load_config
from norrag.data.nrk_quiz import load_nrk_quiz
from norrag.data.wikipedia import load_wikipedia


def main():
    config = load_config()
    data = config["data"]
    processed = get_path(config, "data_processed")

    # 1. Quiz questions
    questions = load_nrk_quiz(data["languages"], data["nrk_quiz"], data["nrk_quiz_revision"])
    questions.to_parquet(processed / "questions.parquet", index=False)
    print(f"Saved {len(questions):,} questions to data/processed/questions.parquet")

    # 2. Clean each language's Wikipedia and save them together as one table
    tables = []
    for language in data["languages"]:
        print(f"Cleaning Wikipedia ({language}) ...")
        tables.append(load_wikipedia(data["wikipedia"], data["wikipedia_configs"][language],
                                     language, data["wikipedia_revision"]))
    articles = pd.concat(tables, ignore_index=True)
    articles.to_parquet(processed / "articles.parquet", index=False)
    print(f"Saved {len(articles):,} articles to data/processed/articles.parquet")


if __name__ == "__main__":
    main()
