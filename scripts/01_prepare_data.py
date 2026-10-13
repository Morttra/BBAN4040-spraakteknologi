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
