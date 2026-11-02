"""Dense retriever that embeds chunks and queries with a sentence-embedding model (NB's sentence-BERT) and searches by cosine similarity.

A sentence-embedding model turns a text into a vector (e.g. 768 numbers) so that texts with similar
meaning get similar vectors, even when they use different words.

What goes here:
- class DenseRetriever with build(chunks, directory), load(directory) and search(queries, k), the same
  three methods as the BM25 retriever (see base.py).

Embedding the full corpus is the slowest step in the project (hours), so build() works in blocks and
saves each block as soon as it is done: a stopped run continues where it left off. Vectors are
normalised to length 1, so the dot product of two vectors is their cosine similarity.
Used by: scripts 03 and 04, through base.get_retriever.
Decision (`retrieval.dense_model`): which model; NbAiLab/nb-sbert-v2-base for now (it reads up to 512
tokens; nb-sbert-base reads only 75, i.e. about the first 50 words of a chunk).
"""

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DenseRetriever:
    """Chunk embeddings from a sentence-embedding model, kept in memory between searches."""

    def __init__(self, model_name, batch_size=16, block_size=50_000):
        self.model_name = model_name
        self.batch_size = batch_size  # texts embedded at once on the GPU
        self.block_size = block_size  # chunks per saved block file
        self.model = None  # loaded on first use, since it takes a while
        self.embeddings = None  # one row per chunk, created by load()
        self.chunk_ids = None

    def _get_model(self):
        """Load the embedding model the first time it is needed (the GPU is used if there is one)."""
        if self.model is None:
            if torch.backends.mps.is_available() or torch.cuda.is_available():
                # Half precision on a GPU: about 30 % faster, and the vectors are practically the same
                # (cosine similarity at least 0.9996 to full precision in a test on 500 chunks)
                self.model = SentenceTransformer(self.model_name, model_kwargs={"torch_dtype": torch.float16})
            else:
                self.model = SentenceTransformer(self.model_name)
        return self.model

    def _embed(self, texts, show_progress=False):
        """Embed a list of texts into normalised vectors (one row per text)."""
        return self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # length 1, so dot product = cosine similarity
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def build(self, chunks, directory):
        """Embed the chunks (a DataFrame with chunk_id and text) block by block, saving each block to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        chunk_ids = chunks["chunk_id"].tolist()

        # Blocks from an earlier run can only be reused if they were made from exactly the same chunks
        ids_path = directory / "chunk_ids.json"
        if ids_path.exists():
            with open(ids_path, encoding="utf-8") as f:
                if json.load(f) != chunk_ids:
                    raise ValueError(f"{directory} holds embeddings of other chunks; delete the folder and rebuild.")
        else:
            with open(ids_path, "w", encoding="utf-8") as f:
                json.dump(chunk_ids, f)

        texts = chunks["text"].tolist()
        starts = range(0, len(texts), self.block_size)
        for number, start in enumerate(tqdm(starts, desc="Embedding blocks", unit=" blocks")):
            block_path = directory / f"embeddings_{number:04d}.npy"
            if block_path.exists():  # done in an earlier run
                continue
            vectors = self._embed(texts[start:start + self.block_size], show_progress=True)
            np.save(block_path, vectors.astype(np.float16))  # float16 halves the size on disk

    def load(self, directory):
        """Load the embeddings saved by build() into memory."""
        directory = Path(directory)
        with open(directory / "chunk_ids.json", encoding="utf-8") as f:
            self.chunk_ids = json.load(f)
        blocks = [np.load(path) for path in sorted(directory.glob("embeddings_*.npy"))]
        self.embeddings = np.concatenate(blocks).astype(np.float32)  # float32 for fast multiplication
        if len(self.embeddings) != len(self.chunk_ids):
            raise ValueError(f"{directory}: embeddings for {len(self.embeddings):,} of {len(self.chunk_ids):,} chunks; "
                             "the build has not finished.")

    def search(self, queries, k=20, query_batch=128):
        """Return the top-k chunks for each query, as a list of (chunk_ids, scores) pairs, best first."""
        results = []
        for start in range(0, len(queries), query_batch):
            query_vectors = self._embed(queries[start:start + query_batch])
            similarities = self.embeddings @ query_vectors.T  # cosine similarity: one column per query
            for column in similarities.T:
                top = np.argpartition(column, -k)[-k:]  # the k best, in no particular order (fast)
                top = top[np.argsort(column[top])[::-1]]  # sort those k, best first
                results.append(([self.chunk_ids[i] for i in top], column[top].tolist()))
        return results
