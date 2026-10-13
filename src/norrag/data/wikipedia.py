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
