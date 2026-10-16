"""Download Norwegian Wikipedia (Bokmål and Nynorsk) and clean the articles into a corpus ready for chunking.

Source: Wikimedia's dump of 1 November 2023, as published on Hugging Face: dataset "wikimedia/wikipedia",
configs "20231101.no" (Bokmål, 617,937 articles) and "20231101.nn" (Nynorsk, 167,653 articles).
Licence: CC BY-SA 3.0 and GFDL, per the dataset card. Each row has id, url, title and text (plain text).
The first download (about 730 MB) is cached by Hugging Face in ~/.cache/huggingface.

What goes here:
- clean_text(text) -> str: cut the trailing sections such as "Referanser", "Eksterne lenker" or "Se også"
  (Nynorsk: "Kjelder", "Sjå òg") and footnotes, which only add noise to retrieval.
- load_wikipedia(dataset, config_name, language, revision) -> DataFrame: one row per cleaned article
  with article_id, language, title, url, text and n_words.

Note: Wikipedia's page id is only unique within one language, so article_id combines language and
page id, e.g. "nb-1348455".
Used by: scripts/01_prepare_data.py.
Decisions: which sections to cut (END_HEADINGS); short and long articles are kept.
"""

import re

import pandas as pd
from datasets import load_dataset

# Headings of the trailing sections (see also, sources, footnotes, links), Bokmål and Nynorsk
END_HEADINGS = ["Se også", "Referanser", "Litteratur", "Kilder", "Eksterne lenker", "Fotnoter", "Noter",
                "Sjå òg", "Kjelder", "Referansar", "Bakgrunnsstoff", "Fotnotar", "Notar", "Merknader", "Merknadar"]

# A trailing-section heading standing alone on its own line
END_PATTERN = re.compile(r"^(?:" + "|".join(map(re.escape, END_HEADINGS)) + r")[ \t]*$", re.M)


def clean_text(text):
    """Cut everything from the first trailing section (sources, footnotes, links, see also)."""
    match = END_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


def load_wikipedia(dataset, config_name, language, revision=None):
    """Load one language from Hugging Face into a DataFrame of cleaned articles, one row per article."""
    articles = load_dataset(dataset, config_name, split="train", revision=revision).to_pandas()

    articles["text"] = articles["text"].map(clean_text)  # run clean_text on every article
    articles = articles[articles["text"] != ""]  # drop articles with nothing left after cleaning

    articles["article_id"] = language + "-" + articles["id"]
    articles["language"] = language
    articles["n_words"] = articles["text"].str.split().str.len()
    articles = articles.drop_duplicates("article_id")  # keep article_id unique, in case a page appears twice
    return articles[["article_id", "language", "title", "url", "text", "n_words"]].reset_index(drop=True)
