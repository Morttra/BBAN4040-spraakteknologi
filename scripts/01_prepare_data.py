"""Download NRK-Quiz-QA and Norwegian Wikipedia (nb + nn) and save cleaned copies under data/processed/.

Steps:
1. Load the NRK-Quiz-QA questions (norrag.data.nrk_quiz) and save them.
2. Download the Wikipedia dumps into data/raw/ (7.4 GB, done once), turn every page's HTML into clean
   text (norrag.data.wikipedia) and save Bokmål and Nynorsk together as one table.

Reads:  Hugging Face (ltg/nrk_quiz_qa, v2.0) and the Wikimedia Enterprise HTML dumps of 20 March 2025
Writes: data/processed/questions.parquet, data/processed/articles.parquet
Run:    python scripts/01_prepare_data.py      (download time, plus roughly 80 minutes of conversion)
"""

import pandas as pd

from norrag.config import get_path, load_config
from norrag.data.nrk_quiz import load_nrk_quiz
from norrag.data.wikipedia import download_file, load_wikipedia


def main():
    config = load_config()
    data = config["data"]
    processed = get_path(config, "data_processed")

    # 1. Quiz questions
    questions = load_nrk_quiz(data["languages"], data["nrk_quiz"], data["nrk_quiz_revision"])
    questions.to_parquet(processed / "questions.parquet", index=False)
    print(f"Saved {len(questions):,} questions to data/processed/questions.parquet")

    # 2. Wikipedia: download each language's dump (skipped if already there), convert and clean it
    tables = []
    for language in data["languages"]:
        path = download_file(data["wikipedia_urls"][language], get_path(config, "data_raw"))
        tables.append(load_wikipedia(path, language))
    articles = pd.concat(tables, ignore_index=True)
    articles.to_parquet(processed / "articles.parquet", index=False)
    print(f"Saved {len(articles):,} articles to data/processed/articles.parquet")


if __name__ == "__main__":
    main()
