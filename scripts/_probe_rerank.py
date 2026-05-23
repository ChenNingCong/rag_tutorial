"""Prototype the bi-encoder vs cross-encoder retrieval experiment on simple-wiki.

Goal: pick N passages, treat the `simplified` of each as a query whose
gold-relevant passage is the corresponding `text`. Show that the cross-encoder
reranker improves recall@1 / MRR over the bi-encoder.
"""
import warnings; warnings.filterwarnings("ignore")
import time

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder


def main():
    N = 3000                     # corpus size
    N_QUERIES = 100              # number of test queries (paraphrase pairs)
    TOP_K_BI = 30                # bi-encoder candidates passed to reranker
    KS = [1, 3, 5, 10]           # ks to evaluate recall at

    ds = load_dataset("sentence-transformers/simple-wiki", split="train", streaming=True)
    pairs = []
    for i, x in enumerate(ds):
        if i >= N:
            break
        pairs.append((x["text"], x["simplified"]))
    print(f"loaded {len(pairs)} (text, simplified) pairs from simple-wiki")

    # Corpus = the `text` field of every row.
    # Queries: take the simplified version and TRUNCATE to ~5 words to simulate a real
    # short search query (full paraphrases share too many words and saturate retrieval).
    import re
    STOP = {"the", "a", "an", "he", "she", "it", "they", "this", "that",
            "and", "or", "but", "of", "to", "in", "on", "for", "with",
            "is", "was", "are", "were", "be", "been"}
    def short_query(s, n=5):
        # drop punctuation, drop leading stop-words, take the next n tokens
        words = [w for w in re.findall(r"[A-Za-z0-9]+", s)]
        while words and words[0].lower() in STOP:
            words.pop(0)
        return " ".join(words[:n])

    corpus = [t for t, _ in pairs]
    queries = [short_query(s, n=5) for _, s in pairs[:N_QUERIES]]
    gold_idx = list(range(N_QUERIES))            # query i's gold passage is corpus[i]
    print("sample queries:", queries[:3])

    bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("encoding corpus...")
    D = bi.encode(corpus, normalize_embeddings=True, convert_to_numpy=True,
                  batch_size=64, show_progress_bar=False)
    print("encoding queries...")
    Q = bi.encode(queries, normalize_embeddings=True, convert_to_numpy=True,
                  batch_size=64, show_progress_bar=False)
    scores = Q @ D.T                              # (N_QUERIES, N)
    bi_rankings = np.argsort(-scores, axis=1)     # each row = doc indices sorted by score

    # Rerank top-K_BI with the cross-encoder.
    print(f"reranking top-{TOP_K_BI} per query with cross-encoder...")
    rerank_rankings = []
    t0 = time.perf_counter()
    for qi, q in enumerate(queries):
        candidates = bi_rankings[qi, :TOP_K_BI]
        pairs_to_score = [[q, corpus[di]] for di in candidates]
        ce_scores = cross.predict(pairs_to_score, show_progress_bar=False)
        order = candidates[np.argsort(-ce_scores)]
        rerank_rankings.append(order)
    rerank_rankings = np.array(rerank_rankings)
    rerank_time = time.perf_counter() - t0

    def recall_at_k(rankings, gold, k):
        gold = np.array(gold)
        rank_pos = (rankings[:, :k] == gold[:, None]).any(axis=1)
        return rank_pos.mean()

    def mrr(rankings, gold):
        gold = np.array(gold)
        rr = []
        for r, g in zip(rankings, gold):
            hit = np.where(r == g)[0]
            rr.append(1.0 / (hit[0] + 1) if hit.size else 0.0)
        return float(np.mean(rr))

    print()
    print(f"{'metric':>10} | {'bi-encoder':>12} | {'+ reranker':>12}")
    print("-" * 45)
    for k in KS:
        print(f"  R@{k:<3d}     | {recall_at_k(bi_rankings, gold_idx, k):12.4f} | "
              f"{recall_at_k(rerank_rankings, gold_idx, k):12.4f}")
    print(f"  MRR        | {mrr(bi_rankings, gold_idx):12.4f} | "
          f"{mrr(rerank_rankings, gold_idx):12.4f}")
    print(f"\nrerank wall time: {rerank_time:.1f}s for {N_QUERIES} queries")

    # Example where reranker fixes a bi-encoder mistake.
    for qi in range(N_QUERIES):
        bi_pos = int(np.where(bi_rankings[qi] == gold_idx[qi])[0][0])
        rr_pos = int(np.where(rerank_rankings[qi] == gold_idx[qi])[0][0])
        if bi_pos >= 3 and rr_pos == 0:
            print(f"\n=== query {qi} fixed by reranker ===")
            print(f"  query (simplified)         : {queries[qi][:160]}")
            print(f"  gold passage (text[{qi}])   : {corpus[qi][:160]}")
            print(f"  bi rank of gold            : {bi_pos + 1}")
            print(f"  bi top-1 (wrong)           : {corpus[bi_rankings[qi, 0]][:160]}")
            print(f"  cross rank of gold         : {rr_pos + 1}")
            break


if __name__ == "__main__":
    main()
