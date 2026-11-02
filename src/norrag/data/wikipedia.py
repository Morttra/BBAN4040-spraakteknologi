"""Download Norwegian Wikipedia (Bokmål) and clean the articles into a corpus ready for chunking.

Source: Wikimedia Enterprise HTML dumps of 20 March 2025 (https://dumps.wikimedia.org/other/enterprise_html/),
nowiki (Bokmål, 6.2 GB), articles only (namespace 0). Licence: CC BY-SA 3.0 (as stated in each record).
The code also handles other language editions such as nnwiki (Nynorsk), which the project does not use.
Each file is a .tar.gz holding several .ndjson files with one JSON record per page; the text is the
rendered page as HTML (article_body.html), so numbers written with Wikipedia templates are included.
The full conversion rules are documented in data/README.md.

What goes here:
- download_file(url, dest_dir) -> Path: download a file once (skipped if it is already there).
- iter_pages(path): read a dump file page by page, without unpacking it to disk.
- html_to_text(html) -> str: the page's headings, paragraphs, list items and fact boxes as plain text.
- clean_text(text) -> str: cut the trailing sections such as "Referanser", "Eksterne lenker" or "Se også"
  (Nynorsk: "Kjelder", "Sjå òg") and footnotes, which only add noise to retrieval.
- load_wikipedia(path, language) -> DataFrame: one row per cleaned article with article_id, language,
  title, url, text, n_words, wikidata and modified.

Note: Wikipedia's page id is only unique within one language, so article_id combines language and
page id, e.g. "nb-1348455".
Used by: scripts/01_prepare_data.py.
Decisions: what to drop (DROP_SELECTORS, other tables), which sections to cut (END_HEADINGS); short and
long articles are kept.
"""

import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Wikimedia refuses downloads from Python's default name ("Python-urllib/3.x") with 403 Forbidden and
# asks scripts to say who they are, so every download names the project
USER_AGENT = "norrag/0.1 (https://github.com/Morttra/BBAN4040-spraakteknologi; NTNU BBAN4040 student project)"

# Parts of a page that are removed before reading the text (CSS selectors: tag names, or .class names)
DROP_SELECTORS = [
    "style", "script", "link", "meta",  # page machinery, not text
    "sup.reference", "sup.mw-ref",  # footnote markers such as [1]
    ".reflist", "ol.references", ".mw-references-wrap",  # the footnote lists themselves
    ".navbox", ".metadata", ".ambox", ".noprint",  # navigation boxes and maintenance notices
    "figure", ".thumb", ".gallery",  # images and their captions
]
# Blocks of text that become lines: headings, paragraphs, list items and definition lists
HEADING_TAGS = ["h2", "h3", "h4", "h5"]
BLOCK_TAGS = HEADING_TAGS + ["p", "li", "dt", "dd"]

# Headings of the trailing sections (see also, sources, footnotes, links), Bokmål and Nynorsk
END_HEADINGS = ["Se også", "Referanser", "Litteratur", "Kilder", "Eksterne lenker", "Fotnoter", "Noter",
                "Sjå òg", "Kjelder", "Referansar", "Bakgrunnsstoff", "Fotnotar", "Notar", "Merknader", "Merknadar"]

# A trailing-section heading standing alone on its own line
END_PATTERN = re.compile(r"^(?:" + "|".join(map(re.escape, END_HEADINGS)) + r")[ \t]*$", re.M)


def _clean_spaces(text):
    """Join the words of a piece of text with single spaces and remove spaces before punctuation."""
    text = " ".join(text.split())
    return re.sub(r"\s+([,.;:!?)])", r"\1", text)


def _infobox_lines(table):
    """Turn a fact box (infobox) into "label: value" lines, one per row that has both."""
    lines = []
    for row in table.find_all("tr"):
        label, value = row.find("th"), row.find("td")
        if label and value:
            lines.append(f"{_clean_spaces(label.get_text(' '))}: {_clean_spaces(value.get_text(' '))}")
    return lines


def html_to_text(html, keep_infobox=True):
    """Turn one rendered Wikipedia page (HTML) into plain text with one heading, paragraph or list item per line.

    Fact boxes (infoboxes) become "label: value" lines when keep_infobox is True; all other tables are skipped.
    Pointer notes at the top ("For kommunen, sjå Førde kommune") and headings without any text under them
    are left out.
    """
    soup = BeautifulSoup(html, "lxml")
    for element in soup.select(", ".join(DROP_SELECTORS)):
        element.decompose()  # remove it from the page

    # Pointer notes: definition lists (<dl>) above the first paragraph that is not inside a table
    for element in soup.body.find_all(["p", "dl"]):
        if element.find_parent("table"):
            continue
        if element.name == "p":
            break  # reached the first real paragraph: stop looking
        element.decompose()

    # Collect the blocks in page order as (text, heading level); level 0 means "not a heading"
    blocks = []
    for element in soup.body.find_all(["table"] + BLOCK_TAGS):
        if element.name == "table":
            if keep_infobox and "infobox" in element.get("class", []) and not element.find_parent("table"):
                blocks.extend((line, 0) for line in _infobox_lines(element))
            continue
        if element.find_parent("table"):  # text inside tables is handled above, or skipped
            continue
        if element.name == "li":  # a list item's own text, without any list nested inside it
            text = "".join(s for s in element.find_all(string=True) if s.find_parent("li") is element)
        else:
            text = element.get_text()
        text = _clean_spaces(text)
        if text:
            level = int(element.name[1]) if element.name in HEADING_TAGS else 0  # "h3" -> 3
            blocks.append((text, level))

    # Lone headings: followed by a heading of the same or a higher level (or nothing), so their section has
    # no text. Trailing-section headings are kept, because clean_text needs them to know where to cut.
    lines = []
    for i, (text, level) in enumerate(blocks):
        next_level = blocks[i + 1][1] if i + 1 < len(blocks) else 1  # end of page counts as a new section
        is_lone = level > 0 and 0 < next_level <= level
        if is_lone and text not in END_HEADINGS:
            continue
        lines.append(text)
    return "\n".join(lines)


def clean_text(text):
    """Cut everything from the first trailing section (sources, footnotes, links, see also)."""
    match = END_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


def download_file(url, dest_dir):
    """Download url into dest_dir and return the file's path; skipped if the file is already there."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / url.split("/")[-1]  # the file name at the end of the URL
    if not path.exists():
        print(f"Downloading {url} ...")
        partial = path.with_name(path.name + ".part")  # download under a temporary name ...
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response, open(partial, "wb") as f:
            shutil.copyfileobj(response, f)  # copy the download to the file piece by piece
        partial.rename(path)  # ... and rename only when complete, so a broken download is never mistaken for a finished one
    return path


def iter_pages(path):
    """Yield the pages of a dump file one at a time, as dictionaries, reading straight from the .tar.gz."""
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:  # each member is one .ndjson file inside the archive
            if not member.name.endswith(".ndjson"):
                continue
            with tar.extractfile(member) as f:
                for line in f:
                    yield json.loads(line.decode("utf-8"))  # the archive gives bytes; decode them as UTF-8 text


def load_wikipedia(path, language):
    """Read one language's dump file into a DataFrame of cleaned articles, one row per article."""
    rows = []
    no_html = 0
    for page in tqdm(iter_pages(path), desc=f"Wikipedia ({language})", unit=" pages"):
        html = page.get("article_body", {}).get("html")
        if not html:  # a handful of pages only have raw wiki markup; they are skipped and counted
            no_html += 1
            continue
        text = clean_text(html_to_text(html))
        if not text:  # nothing left after cleaning
            continue
        rows.append({
            "article_id": f"{language}-{page['identifier']}",
            "language": language,
            "title": page["name"],
            "url": page["url"],
            "text": text,
            "n_words": len(text.split()),
            "wikidata": page.get("main_entity", {}).get("identifier"),  # e.g. "Q26793"; missing for a few pages
            "modified": page.get("date_modified", "")[:10],  # date of the last edit before the dump, "2024-04-28"
        })
    print(f"Wikipedia ({language}): skipped {no_html} pages without HTML")
    articles = pd.DataFrame(rows)
    # If a page appears more than once, keep the most recently edited copy so article_id stays unique
    articles = articles.sort_values("modified").drop_duplicates("article_id", keep="last")
    return articles.sort_index().reset_index(drop=True)
