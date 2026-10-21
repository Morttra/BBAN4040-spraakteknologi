"""Build and save the BM25 and dense indexes over the Wikipedia chunks.

Reads:  data/processed/chunks.parquet and the `retrieval` settings
Writes: data/indexes/bm25/, data/indexes/dense/<model name>/
Uses:   norrag.retrieval (base.get_retriever, and through it bm25, dense and text)
Run:    python scripts/03_build_indexes.py --retriever bm25      (one retriever at a time)

BM25 takes minutes. The dense index embeds every chunk and can take hours, so add a
--limit option and try it on a few thousand chunks first.
"""
#0 Laster inn biblioteker
import pandas as pd
import re
import unicodedata
import html

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

#1 Leser inn Parquet fil
from norrag.config import get_path, load_config

config = load_config()
processed = get_path(config, "data_processed")

chunks = pd.read_parquet(processed / "chunks.parquet")
print(chunks.head())

#2 Vasking
#vasking_bm25
def preprocess_bm25(text: str) -> list[str]:
    if text is None:
        return []
    
    #gjør alt til lowercase
    text = str(text).lower()
    #beholder norske symboler, tall og bindestreker
    tokens = re.findall(r"[a-zæøå0-9]+(?:-[a-zæøå0-9]+)*", text) 
    
    return tokens

#vasking for BERT
def preprocess_bert(text) -> str:
    if text is None:
        return ""
    
    text = str(text)
    # Konverter HTML-entities, f.eks. &amp; -> &
    text = html.unescape(text)
    # Normaliser Unicode
    text = unicodedata.normalize("NFC", text)
    # Fjern HTML-tagger
    text = re.sub(r"<[^>]+>", " ", text)
    # Fjern kontrolltegn
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Normaliser mellomrom
    text = re.sub(r"\s+", " ", text).strip()

    return text

# 4. Lag begge representasjonene
chunks["chunk_id"] = chunks["chunk_id"].astype(str) if "chunk_id" in chunks.columns else chunks.index.astype(str)


chunks["text_clean"] = chunks["text"].apply(preprocess_bert)
chunks["bm25_tokens"] = chunks["text"].apply(preprocess_bm25)

#5 Fjern tomme chunks etter rensing
chunks = chunks[
    (chunks["text_clean"].str.len() > 0)
    & (chunks["bm25_tokens"].apply(len) > 0)
].reset_index(drop=True)

chunk_ids = chunks["chunk_id"].tolist()
chunk_id_to_index = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}


#BM25-indeks
# Konverter kolonnen til liste av tokenlister
tokenized_text_bm25 = chunks["bm25_tokens"].tolist()
bm25 = BM25Okapi(tokenized_text_bm25)

#BERT (nb-sbert)
model_bert = SentenceTransformer("NbAiLab/nb-sbert-base")
tokenized_text_bert = chunks["text_clean"].tolist()

embeddings = model_bert.encode(
    tokenized_text_bert,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True  # praktisk for kosinus-similarity
)
