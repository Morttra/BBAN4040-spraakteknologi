"""Retrieve the top-20 chunks for every question with each retriever and save one run file per retriever under results/raw/.

Reads:  data/processed/questions.parquet, data/indexes/<retriever>/, the `retrieval` settings
Writes: results/raw/runs/<retriever>.jsonl, one line per question, e.g.
            {"qid": "...", "retriever": "bm25", "query": "...", "chunk_ids": [20 ids, best first], "scores": [...]}
        and results/raw/runs/<retriever>.meta.json with the settings used
Uses:   norrag.retrieval (query.build_query, base.get_retriever), norrag.records
Run:    python scripts/04_retrieve.py --retriever bm25      (minutes)

The run files feed 05 (chunks for the prompts), 06 (annotation), 07 (judge) and 08 (demo).
"""
#query conversation
#evaluering og får topp k retrival for å sammenligne scores
import numpy as np

def bm25_search(query: str, k: int = 10):
    tokens = preprocess_bm25(query)
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]

    return [
        {
            "chunk_id": chunk_ids[i],
            "score": float(scores[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(top_idx)
    ]


def bert_search(query: str, k: int = 10):
    query_embedding = model_bert.encode(
        [preprocess_bert(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    similarities = embeddings @ query_embedding  # kosinus for normaliserte vektorer
    top_idx = np.argsort(similarities)[::-1][:k]

    return [
        {
            "chunk_id": chunk_ids[i],
            "score": float(similarities[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(top_idx)
    ]

#evaluering av begge
def get_topk_ids(results, k):
    return [r["chunk_id"] for r in results[:k]]


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def evaluate_retriever(search_function, eval_queries, relevant_docs, k=10):
    recalls, mrrs = [], []

    for query_id, query in eval_queries.items():
        results = search_function(query, k=k)
        retrieved_ids = get_topk_ids(results, k)
        relevant_ids = relevant_docs[query_id]

        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_ids, k))

    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"MRR@{k}": float(np.mean(mrrs)),
    }

#eksempel på queries og relevante dokumenter for evaluering fra ai
eval_queries = {
    "q1": "Hvordan påvirker renten obligasjonspriser?",
}

relevant_docs = {
    "q1": {"chunk_123", "chunk_456"},
}

bm25_eval = evaluate_retriever(bm25_search, eval_queries, relevant_docs, k=10)
bert_eval = evaluate_retriever(bert_search, eval_queries, relevant_docs, k=10)

print(bm25_eval)
print(bert_eval)    