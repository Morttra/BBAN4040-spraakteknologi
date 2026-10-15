"""Download Norwegian Wikipedia (Bokmål and Nynorsk) and clean the articles into a corpus ready for chunking.

Source: Språkbanken's "Tekster fra norsk Wikipedia", a dump from about 20 March 2019 (CC BY-SA 3.0),
https://www.nb.no/sbfil/tekst/2019_wikipedia.tar.gz (about 500 MB). The archive holds nob.wikipedia.json
(Bokmål, 1.3 GB), nno.wikipedia.json (Nynorsk, 300 MB) and sme.wikipedia.json (Northern Sámi, not used).
Each file is one big JSON list of articles with pageid, title, text (already plain text),
contentcategories, hiddencategories, wikidata, revid and bytelength.

What goes here:
- download_wikipedia(url, dest_dir) -> Path: download and unpack the archive (skip if already done).
- load_wikipedia(path, language) -> DataFrame: one row per article with article_id, language, title,
  text, categories and wikidata. Loading the 1.3 GB file at once needs several GB of memory.
- clean_text(text) -> str: tidy one article, e.g. cut trailing sections such as "Eksterne lenker",
  "Referanser" or "Se også" (Nynorsk: "Kjelder", "Sjå òg") that only add noise to retrieval.

Note: pageid is only unique within one language's Wikipedia, so article_id combines language and
pageid, e.g. "nb-1348455".
Used by: scripts/01_prepare_data.py.
Decisions: which sections to cut, and whether to drop very short articles.
"""

import json
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

# Headings of the table of contents that Wikipedia puts near the top of longer articles
TOC_HEADINGS = ["Innhold", "Innhald", "Innhaldsliste"]
# Headings of the trailing sections (see also, sources, links), Bokmål and Nynorsk
END_HEADINGS = ["Se også", "Referanser", "Litteratur", "Kilder", "Eksterne lenker",
                "Sjå òg", "Kjelder", "Referansar", "Bakgrunnsstoff"]

# The table of contents: its heading line, then blank lines and numbered lines such as "4.1 Ernæring"
TOC_PATTERN = re.compile(r"^(?:" + "|".join(TOC_HEADINGS) + r")[ \t]*\n(?:[ \t]*\n|\d+(?:\.\d+)*[ \t][^\n]*\n)+", re.M)
# A trailing-section heading standing alone on its own line
END_PATTERN = re.compile(r"^(?:" + "|".join(map(re.escape, END_HEADINGS)) + r")[ \t]*$", re.M)


def download_wikipedia(url, dest_dir):
    """Download the Wikipedia archive into dest_dir and unpack it; both steps are skipped if already done."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / url.split("/")[-1]  # the file name at the end of the URL: 2019_wikipedia.tar.gz

    if not archive.exists():
        print(f"Downloading {url} (about 500 MB) ...")
        partial = archive.with_suffix(".part")  # download under a temporary name ...
        urllib.request.urlretrieve(url, partial)
        partial.rename(archive)  # ... and rename only when complete, so a broken download is never mistaken for a finished one

    if not any(dest_dir.rglob("*.wikipedia.json")):
        print(f"Unpacking {archive.name} ...")
        with tarfile.open(archive) as tar:
            tar.extractall(dest_dir, filter="data")  # filter="data" blocks files that try to write outside dest_dir

    return dest_dir


def clean_text(text):
    """Remove the table of contents and cut everything from the first trailing section (sources, links, see also)."""
    text = TOC_PATTERN.sub("", text, count=1)
    match = END_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


def load_wikipedia(path, language):
    """Read one language's JSON file into a DataFrame of cleaned articles, one row per article."""
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)  # the whole file is one JSON list; needs a few GB of memory for Bokmål

    rows = []
    for article in articles:
        text = clean_text(article["text"])
        if not text:  # nothing left after cleaning
            continue
        rows.append({
            "article_id": f"{language}-{article['pageid']}",  # pageid is only unique within one language
            "language": language,
            "title": article["title"],
            "text": text,
            "n_words": len(text.split()),
            "categories": article["contentcategories"],
            "wikidata": article["wikidata"] or None,  # about 850 articles have false instead of an id
        })
    return pd.DataFrame(rows)
