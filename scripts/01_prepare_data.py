"""Download NRK-Quiz-QA and Norwegian Wikipedia (nb + nn) and save cleaned copies under data/processed/.

Steps:
1. Load the NRK-Quiz-QA questions (norrag.data.nrk_quiz) and save them.
2. Download and unpack the 2019 Wikipedia archive into data/raw/ (norrag.data.wikipedia; about 500 MB,
   done once).
3. Load the Bokmål and Nynorsk files, clean the text and save them as one table.

Reads:  Hugging Face (ltg/nrk_quiz_qa) and https://www.nb.no/sbfil/tekst/2019_wikipedia.tar.gz
Writes: data/processed/questions.parquet, data/processed/articles.parquet
Run:    python scripts/01_prepare_data.py      (minutes, mostly the download)
"""

import pandas as pd

from norrag.config import get_path, load_config
from norrag.data.nrk_quiz import load_nrk_quiz
from norrag.data.wikipedia import download_wikipedia, load_wikipedia


def main():
    config = load_config()
    data = config["data"]
    processed = get_path(config, "data_processed")

    # 1. Quiz questions
    questions = load_nrk_quiz(data["languages"], data["nrk_quiz"])
    questions.to_parquet(processed / "questions.parquet", index=False)
    print(f"Saved {len(questions):,} questions to data/processed/questions.parquet")

    # 2. Wikipedia archive (skipped if already downloaded and unpacked)
    raw = download_wikipedia(data["wikipedia_url"], get_path(config, "data_raw"))

    # 3. Clean each language and save them together as one table
    tables = []
    for language in data["languages"]:
        print(f"Cleaning Wikipedia ({language}) ...")
        tables.append(load_wikipedia(raw / data["wikipedia_files"][language], language))
    articles = pd.concat(tables, ignore_index=True)
    articles.to_parquet(processed / "articles.parquet", index=False)
    print(f"Saved {len(articles):,} articles to data/processed/articles.parquet")


if __name__ == "__main__":
    main()
