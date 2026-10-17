"""Split Wikipedia articles into passages with stable chunk_ids that are shared by all retrievers.

Why: whole articles are too long to put in a prompt, so each article is cut into chunks and the
retrievers search among the chunks. All three retrievers must use exactly the same chunks, so that a
relevance label on a chunk counts for every retriever that returns it.

What goes here:
- split_text(text, max_words, overlap_words) -> list of str: cut one text into pieces of about
  max_words words; neighbouring pieces share overlap_words words, so a fact on a boundary is not lost.
- make_chunk_id(article_id, position) -> str: e.g. "nb-1348455-0003". The same input always gives
  the same id; that is what "stable" means.
- chunk_corpus(articles, max_words, overlap_words, add_title) -> DataFrame: one row per chunk with
  chunk_id, article_id, language, title, position and text.

Used by: scripts/02_chunk_corpus.py, scripts/08_make_demo_data.py, tests/test_chunking.py.
Decisions (configs/experiment.yaml, `chunking`): chunk size, overlap, and whether each chunk starts
with the article title.
"""

import pandas as pd
from tqdm import tqdm


def split_text(text, max_words=100, overlap_words=20):
    """Cut a text into chunks of at most max_words words, keeping whole paragraphs together where possible.

    Each chunk after the first starts with the last overlap_words words of the chunk before it, so a fact
    on a boundary is not lost. Only paragraphs longer than max_words - overlap_words are cut in the middle.
    """
    # 1. Cut the text into pieces: one per paragraph (line), and long paragraphs into several pieces
    piece_size = max_words - overlap_words  # leaves room for the overlap at the start of the next chunk
    pieces = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        for start in range(0, len(words), piece_size):
            pieces.append(words[start:start + piece_size])

    # 2. Pack the pieces into chunks, starting a new chunk when the next piece would not fit
    chunks = []
    current = []  # the pieces in the chunk being built
    new_words = 0  # words in `current` that are not overlap from the previous chunk
    for piece in pieces:
        if new_words > 0 and sum(len(p) for p in current) + len(piece) > max_words:
            chunks.append(current)
            all_words = [word for p in current for word in p]
            tail = all_words[-overlap_words:] if overlap_words > 0 else []
            current = [tail] if tail else []
            new_words = 0
        current.append(piece)
        new_words += len(piece)
    if new_words > 0:  # the last chunk, unless it would only repeat the overlap
        chunks.append(current)

    # 3. Turn each chunk back into text: words joined by spaces, pieces (paragraphs) by line breaks
    return ["\n".join(" ".join(p) for p in chunk) for chunk in chunks]


def make_chunk_id(article_id, position):
    """Return the id of chunk number `position` (counting from 0) in an article, e.g. "nb-1348455-0003"."""
    return f"{article_id}-{position:04d}"  # :04d writes the number with four digits: 3 -> "0003"


def chunk_corpus(articles, max_words=100, overlap_words=20, add_title=True):
    """Chunk every article in the articles table; returns a DataFrame with one row per chunk.

    With add_title, each chunk's text starts with the article title on its own line. The title is not
    counted in max_words; n_words counts the article words in the chunk only.
    """
    rows = []
    for article in tqdm(articles.itertuples(index=False), total=len(articles), desc="Chunking", unit=" articles"):
        chunks = split_text(article.text, max_words, overlap_words)
        for position, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": make_chunk_id(article.article_id, position),
                "article_id": article.article_id,
                "language": article.language,
                "title": article.title,
                "position": position,
                "text": f"{article.title}\n{chunk}" if add_title else chunk,
                "n_words": len(chunk.split()),
            })
    return pd.DataFrame(rows)
