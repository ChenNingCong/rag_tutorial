"""Generate tutorial/05_reranking.ipynb.

Section 5: 2-stage retrieval. Empirically show that a small cross-encoder
reranker improves recall and MRR over a bi-encoder baseline on a real corpus
(simple-wiki). No LLM in this notebook -- the point is retrieval quality.
"""
from pathlib import Path

from _nb_utils import build_notebook, code, md

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorial" / "05_reranking.ipynb"


cells = [
    md(r"""
    # 5. Reranking — the 2-stage retrieval pipeline

    From notebook 4 we know two things:

    - **Bi-encoders** are cheap. You index the corpus once, then every query
      is one forward pass + a matrix multiply.
    - **Cross-encoders** are accurate but slow. They re-encode every `(q, d)`
      pair from scratch.

    The dominant pattern in modern retrieval gets the best of both:

    ```
          query
            │
            ▼
        ┌────────────────────────────┐
        │  STAGE 1 — bi-encoder      │   billions of docs, top-k cheap
        │  cosine over the entire    │
        │  corpus → top 50–200       │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  STAGE 2 — cross-encoder   │   50–200 docs, top-10 expensive
        │  re-score each (q, d) pair │
        │  → top 5–10                │
        └────────────────────────────┘
                     │
                     ▼
              best candidates
    ```

    This notebook is short and empirical: we'll measure the recall and MRR
    of stage 1 alone vs. stage 1 + stage 2 on a real corpus. No LLM is
    involved — we're isolating *retrieval quality*.

    References:

    - sentence-transformers docs on [retrieve & re-rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html).
    - The MS-MARCO MiniLM cross-encoders: <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2>.
    """),

    md(r"""
    ## 5.1 The setup — turning simple-wiki into a retrieval benchmark

    The HuggingFace dataset
    [`sentence-transformers/simple-wiki`](https://huggingface.co/datasets/sentence-transformers/simple-wiki)
    ships **(text, simplified)** sentence pairs from English Wikipedia. We
    use it as a tiny BEIR-style benchmark:

    - **Corpus**: the `text` field of the first 3 000 rows (3 000 documents).
    - **Queries**: the `simplified` field of the first 100 rows, with leading
      stop-words stripped and *truncated to 5 tokens* — so the query carries
      only partial information, the way a real search query does. A full
      paraphrase is too easy and saturates the bi-encoder at 100 % R@5.
    - **Gold**: query `i`'s relevant document is corpus row `i`.

    With those choices, the retrieval task is meaningfully hard but small
    enough to run on CPU in a minute or two.
    """),

    code(r"""
    import warnings; warnings.filterwarnings("ignore")
    import re
    import numpy as np
    from datasets import load_dataset

    N_CORPUS  = 3000
    N_QUERIES = 100

    ds = load_dataset("sentence-transformers/simple-wiki", split="train", streaming=True)
    rows = []
    for i, x in enumerate(ds):
        if i >= N_CORPUS:
            break
        rows.append((x["text"], x["simplified"]))
    print(f"loaded {len(rows)} text/simplified pairs")

    corpus = [t for t, _ in rows]                         # documents

    STOP = {"the","a","an","he","she","it","they","this","that",
            "and","or","but","of","to","in","on","for","with",
            "is","was","are","were","be","been"}
    def short_query(s, n=5):
        words = [w for w in re.findall(r"[A-Za-z0-9]+", s)]
        while words and words[0].lower() in STOP:
            words.pop(0)
        return " ".join(words[:n])

    queries  = [short_query(rows[i][1], n=5) for i in range(N_QUERIES)]
    gold_idx = list(range(N_QUERIES))                     # query i's gold doc is corpus[i]

    print("\nexample queries (truncated to 5 content words):")
    for i in range(5):
        print(f"  q{i:>2}: {queries[i]!r}")
        print(f"        gold: {corpus[i][:120]}...")
    """),

    md(r"""
    ## 5.2 Stage 1 — bi-encoder baseline

    Same model as everywhere else in this tutorial: `all-MiniLM-L6-v2`.
    Encode the corpus once, then for every query do `D @ q` and sort.
    """),

    code(r"""
    from sentence_transformers import SentenceTransformer

    bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("encoding corpus...")
    D = bi.encode(corpus, normalize_embeddings=True, convert_to_numpy=True,
                  batch_size=64, show_progress_bar=False)
    print("encoding queries...")
    Q = bi.encode(queries, normalize_embeddings=True, convert_to_numpy=True,
                  batch_size=64, show_progress_bar=False)
    scores = Q @ D.T                                      # (N_QUERIES, N_CORPUS)

    bi_rankings = np.argsort(-scores, axis=1)             # docs sorted by similarity, per query
    print("bi-encoder rankings shape:", bi_rankings.shape)
    """),

    md(r"""
    ## 5.3 Stage 2 — cross-encoder reranker

    Now layer the cross-encoder on top. For each query we take the
    **top 30** bi-encoder candidates and re-score each `(query, doc)` pair
    with the cross-encoder. Then we sort by the new scores.

    Crucial bookkeeping: the reranker can only re-order the candidates it
    was given. **It cannot add a document that the bi-encoder missed.**
    That means the reranker can improve recall *only* for k ≤ TOP_K_BI.
    """),

    code(r"""
    from sentence_transformers import CrossEncoder

    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    TOP_K_BI = 30   # how many bi-encoder candidates we pass to the reranker

    print(f"reranking top-{TOP_K_BI} per query with the cross-encoder...")
    rerank_rankings = []
    for qi, q in enumerate(queries):
        candidates = bi_rankings[qi, :TOP_K_BI]                    # doc ids from stage 1
        pairs = [[q, corpus[di]] for di in candidates]
        ce_scores = cross.predict(pairs, show_progress_bar=False)  # one logit per pair
        order = candidates[np.argsort(-ce_scores)]                 # reorder by reranker
        rerank_rankings.append(order)
    rerank_rankings = np.array(rerank_rankings)
    print("rerank_rankings shape:", rerank_rankings.shape)
    """),

    md(r"""
    ## 5.4 Measure recall and MRR

    Two retrieval metrics suffice for this comparison; notebook 6 covers
    a fuller battery (nDCG, top-k, RAGAS-style LLM judging).

    - **Recall@k** — what fraction of queries have *any* relevant document
      in the top k? Since each query has exactly one gold doc here, this is
      just "fraction of queries where the gold is in top-k".
    - **MRR** (Mean Reciprocal Rank) — average of $1 / r$, where $r$ is the
      rank of the gold document. A perfect ranker has MRR = 1; pushing the
      gold from rank 3 to rank 1 lifts the reciprocal rank from 0.33 to
      1.0 for that query.
    """),

    code(r"""
    def recall_at_k(rankings, gold, k):
        gold = np.array(gold)
        return float((rankings[:, :k] == gold[:, None]).any(axis=1).mean())

    def mrr(rankings, gold):
        gold = np.array(gold)
        rrs = []
        for r, g in zip(rankings, gold):
            hit = np.where(r == g)[0]
            rrs.append(1.0 / (hit[0] + 1) if hit.size else 0.0)
        return float(np.mean(rrs))

    print(f"{'metric':>10s} | {'bi-encoder':>12s} | {'+ reranker':>12s} | {'delta':>8s}")
    print("-" * 55)
    for k in [1, 3, 5, 10]:
        a = recall_at_k(bi_rankings, gold_idx, k)
        b = recall_at_k(rerank_rankings, gold_idx, k)
        print(f"  R@{k:<3d}   | {a:12.4f} | {b:12.4f} | {b-a:+8.4f}")
    a = mrr(bi_rankings, gold_idx); b = mrr(rerank_rankings, gold_idx)
    print(f"  MRR     | {a:12.4f} | {b:12.4f} | {b-a:+8.4f}")
    """),

    md(r"""
    Three things to notice in this table:

    1. **R@1 improves substantially.** The bi-encoder gets a lot of right
      docs into the top 10 — but the *top-1* slot is contested by close
      distractors, and that's exactly where a cross-encoder's joint
      attention over the (query, doc) pair pays off.
    2. **MRR follows R@1.** MRR is dominated by whether the gold doc is at
      position 1 vs. 2 vs. 3, so the same effect as R@1 shows up here too.
    3. **R@k saturates** at the bi-encoder's R@30 — the reranker can
      reorder the top 30 but it cannot add new docs. If you want to lift
      that ceiling you have to make stage 1 better (bigger model, hybrid
      BM25 + embeddings, etc.).
    """),

    md(r"""
    ## 5.5 A qualitative example

    Numbers are convincing but a worked case is sticky. Find the first query
    where the reranker moved the gold doc from a low rank to the top.
    """),

    code(r"""
    for qi in range(N_QUERIES):
        bi_pos = int(np.where(bi_rankings[qi] == gold_idx[qi])[0][0])
        rr_pos = int(np.where(rerank_rankings[qi] == gold_idx[qi])[0][0])
        if bi_pos >= 3 and rr_pos == 0:
            print(f"query q{qi}: {queries[qi]!r}\n")
            print(f"  gold passage corpus[{qi}]:")
            print(f"     {corpus[qi][:200]}...\n")
            print(f"  bi-encoder put the gold at rank {bi_pos+1}.  its top-1 was:")
            print(f"     {corpus[bi_rankings[qi, 0]][:200]}\n")
            print(f"  reranker moved the gold to rank {rr_pos+1}. ✓")
            break
    """),

    md(r"""
    The bi-encoder's top-1 in that example is *topically related* to the
    query (a sentence with overlapping vocabulary), but it is not the
    document that actually contains the answer to the query phrase. The
    cross-encoder, by jointly attending to query tokens and doc tokens,
    can tell apart "this doc talks about a similar topic" from "this doc
    *answers the query*".

    ## 5.6 When is it worth it?

    A 2-stage pipeline isn't free — adding the reranker roughly multiplies
    per-query latency. A rough rule of thumb:

    | situation                                          | what to do                              |
    |----------------------------------------------------|-----------------------------------------|
    | < ~1 000 docs, exact match domain                  | BM25 alone is often enough              |
    | clean corpus, short queries, < 100k docs           | bi-encoder alone                        |
    | RAG with an LLM downstream, latency budget ~1 s    | **bi-encoder + cross-encoder rerank**  |
    | very large corpus, top-quality search              | hybrid (BM25 + bi-encoder) → rerank     |
    | very low latency, top-quality search               | ColBERT-style late-interaction (out of scope here) |

    The default for production RAG is the third row: bi-encoder gets ~100
    candidates from the index, then a cheap cross-encoder picks the best 5
    to put in the LLM prompt. That's the pipeline notebook 7 will assume.
    """),
]


if __name__ == "__main__":
    path = build_notebook(cells, OUT)
    print(f"wrote {path}")
