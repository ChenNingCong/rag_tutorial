"""Generate tutorial/06_evaluation.ipynb.

Section 6: How to evaluate a RAG system.

- Retrieval metrics: top-k accuracy, recall@k, MRR, nDCG.
- Generation metrics: why string overlap (BLEU/ROUGE) fails; LLM-as-judge.
- The four canonical 'RAGAS' metrics implemented from scratch with Gemini.
"""
from pathlib import Path

from _nb_utils import build_notebook, code, md

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorial" / "06_evaluation.ipynb"


cells = [
    md(r"""
    # 6. Evaluating a RAG system

    There are two completely different "is RAG good?" questions hiding inside
    every RAG evaluation:

    1. **Retrieval evaluation** — does the *retriever* surface the right
      documents for each query? This is a ranking problem with a clear gold
      answer (which doc(s) are relevant).
    2. **Generation evaluation** — given the right (or wrong) context, does
      the *LLM* produce a faithful, relevant answer? This is an open-ended
      text-quality problem with no single gold output.

    Different problems → different metrics. Mixing them up — measuring
    answer quality and concluding "our embeddings are bad" — is one of the
    most common diagnostic mistakes.

    What we'll do:

    - **§6.1–6.3**: the four standard retrieval metrics — top-k accuracy,
      recall@k, MRR, nDCG — implemented from scratch on a worked example.
    - **§6.4**: why exact-string metrics (BLEU, ROUGE, exact match) silently
      mis-rank good generations.
    - **§6.5**: "LLM-as-judge" — and the four [RAGAS](https://docs.ragas.io/)
      metrics (Faithfulness, Answer Relevancy, Context Precision, Context
      Recall) coded ourselves with Gemini, so nothing is a black box.

    References:

    - Hugging Face cookbook: ["RAG evaluation"](https://huggingface.co/learn/cookbook/rag_evaluation).
    - The [RAGAS paper](https://arxiv.org/abs/2309.15217) and library docs.
    """),

    md(r"""
    ## 6.1 Retrieval metrics, on the toy example we already have

    We'll re-use the simple-wiki retrieval setup from notebook 5 (3 000-doc
    corpus, 30 short-keyword queries) so the numbers are comparable across
    notebooks. We *won't* re-run the cross-encoder here — the focus is on
    the metrics themselves, not on which retriever wins.
    """),

    code(r"""
    import warnings; warnings.filterwarnings("ignore")
    import re
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    N_CORPUS, N_QUERIES = 3000, 30
    STOP = {"the","a","an","he","she","it","they","this","that",
            "and","or","but","of","to","in","on","for","with",
            "is","was","are","were","be","been"}

    def short_query(s, n=5):
        words = [w for w in re.findall(r"[A-Za-z0-9]+", s)]
        while words and words[0].lower() in STOP:
            words.pop(0)
        return " ".join(words[:n])

    ds = load_dataset("sentence-transformers/simple-wiki", split="train", streaming=True)
    rows = [(x["text"], x["simplified"]) for i, x in enumerate(ds) if i < N_CORPUS]
    corpus    = [t for t, _ in rows]
    queries   = [short_query(rows[i][1]) for i in range(N_QUERIES)]
    gold_idx  = list(range(N_QUERIES))

    bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    D = bi.encode(corpus,  normalize_embeddings=True, convert_to_numpy=True,
                  batch_size=64, show_progress_bar=False)
    Q = bi.encode(queries, normalize_embeddings=True, convert_to_numpy=True,
                  batch_size=64, show_progress_bar=False)
    rankings = np.argsort(-(Q @ D.T), axis=1)
    print("bi-encoder rankings shape:", rankings.shape)
    """),

    md(r"""
    ### Top-k accuracy / Recall@k / Hit@k

    For a single-answer setup like ours (each query has exactly one
    relevant document), all three are the same number:

    $$
    \mathrm{Recall@k} \;=\;
    \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}[\text{gold of } q \text{ is in top-}k]
    $$

    Read it as "fraction of queries where the answer was in the top k".
    If there are *multiple* relevant documents per query (the BEIR /
    MS-MARCO setup), Recall@k generalises to "fraction of *relevant docs*
    in top-k, averaged over queries":

    $$
    \mathrm{Recall@k}\;=\;\frac{1}{|Q|}\sum_q \frac{|R_q \cap \text{top-}k|}{|R_q|}
    $$

    while top-k *accuracy* / Hit@k stays at "did *any* relevant doc make it".

    The single-relevant case is more common in RAG sanity checks; the
    multi-relevant case shows up in BEIR-style benchmarks. We'll implement
    the multi-relevant form because it covers both.
    """),

    code(r"""
    def recall_at_k(rankings, gold_lists, k):
        '''rankings: (Q, N), gold_lists: list of length Q, each a set of relevant doc ids'''
        rec = []
        for r, gold in zip(rankings, gold_lists):
            if not gold:                # skip queries with no relevant docs
                continue
            hits = sum(1 for d in r[:k] if d in gold)
            rec.append(hits / len(gold))
        return float(np.mean(rec))

    # Each query has exactly one gold doc in our setup.
    gold_lists = [{g} for g in gold_idx]

    for k in [1, 3, 5, 10]:
        print(f"  Recall@{k:<3d} = {recall_at_k(rankings, gold_lists, k):.3f}")
    """),

    md(r"""
    ### MRR (Mean Reciprocal Rank)

    Recall@k tells us *whether* the relevant doc is in the top k but says
    nothing about *where* in the top k. MRR fills that gap:

    $$
    \mathrm{MRR} \;=\;
    \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank of first relevant doc}}
    $$

    A relevant doc at rank 1 contributes 1.0; at rank 2 contributes 0.5; at
    rank 5 contributes 0.2. MRR is the standard "is the right answer near
    the top?" metric for single-answer retrieval.
    """),

    code(r"""
    def mrr(rankings, gold_lists):
        rrs = []
        for r, gold in zip(rankings, gold_lists):
            for pos, d in enumerate(r, start=1):
                if d in gold:
                    rrs.append(1.0 / pos)
                    break
            else:
                rrs.append(0.0)
        return float(np.mean(rrs))

    print(f"  MRR = {mrr(rankings, gold_lists):.3f}")
    """),

    md(r"""
    ### nDCG (normalized Discounted Cumulative Gain)

    MRR cares only about the *first* relevant doc. nDCG cares about all
    of them, and supports **graded** relevance (not just relevant/not).
    The definitions:

    - **Gain** of a doc at rank $i$: $g_i = 2^{\mathrm{rel}_i} - 1$.
      For binary relevance, that's just `1 if relevant else 0`.
    - **DCG@k**: $\sum_{i=1}^{k} g_i / \log_2(i+1)$ — accumulate gain but
      discount it the further down the ranking it appears.
    - **iDCG@k**: DCG@k of the *ideal* ranking (sort gold docs to the top).
    - **nDCG@k** = DCG@k / iDCG@k ∈ [0, 1].

    The point of the $\log_2$ discount: moving a relevant doc from rank 1 to
    rank 2 hurts a lot; moving it from rank 8 to rank 9 hurts almost nothing.
    """),

    code(r"""
    def dcg_at_k(rankings, rel, k):
        '''rel: dict {doc_id -> graded relevance int} for one query.'''
        gain = 0.0
        for i, d in enumerate(rankings[:k], start=1):
            r = rel.get(int(d), 0)
            if r > 0:
                gain += (2 ** r - 1) / np.log2(i + 1)
        return gain

    def ndcg_at_k(rankings, rel_lists, k):
        ndcgs = []
        for r, rel in zip(rankings, rel_lists):
            if not rel:
                continue
            dcg = dcg_at_k(r, rel, k)
            # ideal: rank docs by their relevance descending
            ideal_order = sorted(rel.keys(), key=lambda d: -rel[d])
            idcg = dcg_at_k(np.array(ideal_order), rel, k)
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(ndcgs))

    # binary relevance: 1 for the gold doc, 0 otherwise
    rel_lists = [{g: 1} for g in gold_idx]
    for k in [1, 3, 5, 10]:
        print(f"  nDCG@{k:<3d} = {ndcg_at_k(rankings, rel_lists, k):.3f}")
    """),

    md(r"""
    For *binary* single-answer retrieval, nDCG and MRR carry essentially the
    same information — they both discount by rank position. nDCG starts to
    differ when:

    - There are multiple relevant docs per query, or
    - Relevance is **graded** (rel = 3 for a perfect doc, 1 for "tangentially
      relevant", 0 for unrelated — common in TREC / MS-MARCO).

    Let's show the *graded* case on a small synthetic example so the formula
    isn't an abstraction:
    """),

    code(r"""
    # Imagine one query whose relevance judgements are:
    #   doc 0 -> 3  (perfect)
    #   doc 1 -> 0  (irrelevant)
    #   doc 2 -> 2  (partially relevant)
    #   doc 3 -> 1  (barely relevant)
    rel = {0: 3, 1: 0, 2: 2, 3: 1}

    cases = {
        "A: ideal     (0,2,3,1)":   np.array([0, 2, 3, 1]),
        "B: swap 1<->2 (2,0,3,1)":   np.array([2, 0, 3, 1]),
        "C: random   (1,3,0,2)":   np.array([1, 3, 0, 2]),
        "D: worst    (1,3,2,0)":   np.array([1, 3, 2, 0]),
    }

    for name, r in cases.items():
        print(f"  {name}  nDCG@4 = {dcg_at_k(r, rel, 4)/dcg_at_k(np.array([0,2,3,1]), rel, 4):.3f}")
    """),

    md(r"""
    nDCG@4 = 1.0 only for the ideal ordering. Putting a partially-relevant
    doc above a perfect one (case B) drops nDCG a little; putting the
    barely-relevant doc on top (case C) drops it more; putting the
    irrelevant doc *and* the barely-relevant doc above everything else
    (case D) drops it the most. That's nDCG doing its job.
    """),

    md(r"""
    ## 6.2 Why string-overlap metrics aren't enough for generation

    Retrieval metrics need gold *doc ids*. Generation metrics need to compare
    *free text*. The classical answer — BLEU, ROUGE, exact match — was
    designed for translation/summarization and breaks for RAG:

    - **The same fact can be phrased in many equally-correct ways.**
      `"Hillery served as President from 1976 to 1990"` and `"He was Ireland's
      sixth President for fourteen years starting in 1976"` are both correct;
      ROUGE-L between them is poor.
    - **Wrong but on-topic answers can score high.** An answer that confidently
      states the wrong date will share many tokens with the gold and rate
      well under ROUGE.

    A small demo makes the failure mode concrete:
    """),

    code(r"""
    from difflib import SequenceMatcher

    def rouge_l_ratio(a, b):
        '''A toy ROUGE-L approximation: longest matching subsequence ratio.'''
        return SequenceMatcher(None, a.split(), b.split()).ratio()

    gold = "Patrick Hillery served as President of Ireland from 1976 until 1990."

    candidates = [
        ("correct, different wording",
         "Ireland's sixth president was Patrick Hillery, in office from 1976 to 1990."),
        ("wrong fact, same words",
         "Patrick Hillery served as President of Ireland from 1986 until 1990."),
        ("a refusal",
         "I don't know based on the provided documents."),
    ]

    print(f"{'rouge-L':>8s}  {'verdict':25s}  candidate")
    for label, cand in candidates:
        print(f"  {rouge_l_ratio(gold, cand):.3f}    {label:25s}  {cand}")
    """),

    md(r"""
    Look at the wrong-fact answer: it shares almost every word with the gold
    (only "1976" → "1986"), so ROUGE rates it as the **best** of the three —
    despite being objectively false. The correct-but-paraphrased answer
    scores lower. Surface-overlap metrics measure how similar two strings
    *look*, not whether they say the same *thing*. That's why RAG eval moved
    to LLM-as-judge.

    ## 6.3 LLM-as-judge

    The idea is unreasonably simple: give a strong LLM the question, the
    candidate answer, and (sometimes) the gold answer or the retrieved
    context — then ask it for a verdict. The LLM does the semantic
    comparison for you. The result is far better correlated with human
    judgement than BLEU/ROUGE, at the cost of:

    - **Compute** — every eval is now an API call.
    - **Bias** — judges have known systematic biases (preference for
      longer answers, for their own outputs, etc.). Use a *different*
      model as judge than as generator when possible, and average over
      many examples.
    - **Reproducibility** — set `temperature=0`, log the exact prompt.

    We'll use Gemini as our judge throughout.
    """),

    code(r"""
    import os, re
    from pathlib import Path

    KEY_FILE = Path("..") / "key.env"
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = KEY_FILE.read_text(encoding="utf-8").strip()

    from google import genai
    from google.genai import types

    client = genai.Client()
    # NOTE: We do NOT retry on 429 / 503. If the call fails, the traceback shows
    # up immediately so you know to wait ~30s and re-run the cell, rather than
    # blocking on a silent sleep.
    JUDGE_MODEL = "gemini-3-flash-preview"

    def judge(prompt: str) -> str:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return (resp.text or "").strip()

    print("judge client ready, model:", JUDGE_MODEL)
    """),

    md(r"""
    ## 6.4 The four RAGAS metrics, from scratch

    The [RAGAS framework](https://docs.ragas.io/) crystallised four
    LLM-as-judge metrics that have become a de-facto standard for RAG
    evaluation. We code each one directly so nothing is hidden inside a
    library.

    | metric                  | needs               | "did the model …?"                              |
    |-------------------------|---------------------|-------------------------------------------------|
    | **Faithfulness**        | answer, context     | …only say things supported by the context?      |
    | **Answer relevancy**    | question, answer    | …actually address the user's question?          |
    | **Context precision**   | question, context   | …retrieve mostly *relevant* chunks (not noise)? |
    | **Context recall**      | question, context, gold | …retrieve *enough* to answer the question?  |

    Faithfulness + answer relevancy measure the generator. Context precision
    + context recall measure the retriever. Together they answer "which box
    is broken?" — which is exactly what notebook 8 needs.

    A toy RAG example to score:
    """),

    code(r"""
    # One handcrafted RAG instance covering all four cases.
    EXAMPLE = {
        "question": "Who was the sixth President of Ireland and when did he serve?",
        "context": [
            # relevant
            "Patrick John 'Paddy' Hillery (1923-2008) was an Irish Fianna Fáil "
            "politician and the sixth President of Ireland from 1976 until 1990.",
            # tangentially related
            "Hillery served as Social Affairs Commissioner before becoming President.",
            # irrelevant noise
            "The orchestra was formed in 1932 by Sir Thomas Beecham.",
        ],
        # what the RAG system actually produced (faithful answer)
        "answer": (
            "The sixth President of Ireland was Patrick Hillery, who served from 1976 "
            "until 1990."
        ),
        # gold reference (only used for context-recall)
        "gold": "Patrick Hillery was the sixth President of Ireland from 1976 to 1990.",
    }
    print(EXAMPLE["question"])
    """),

    md(r"""
    ### Faithfulness

    *Does every factual statement in the answer come from the context?*

    Implementation sketch (matches RAGAS at a high level):

    1. Ask the judge to split the answer into atomic factual claims.
    2. For each claim, ask the judge whether the context *entails* it.
    3. Faithfulness = (# claims supported) / (# claims).

    A faithfulness of 1.0 means everything in the answer is grounded in the
    retrieved chunks. <1.0 means the model is making things up.
    """),

    code(r"""
    import json

    def extract_json(text):
        '''Gemini sometimes wraps JSON in ```json ... ``` fences. Strip and parse.'''
        m = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else None

    # IMPORTANT design choice: each metric below makes EXACTLY ONE judge call
    # that returns a JSON list of (claim, supported) verdicts. That keeps the
    # whole evaluation under the free-tier rate limit and runs in seconds.

    def faithfulness(answer: str, context: list[str]) -> dict:
        ctx = "\n".join(f"- {c}" for c in context)
        raw = judge(
            "Split the ANSWER into atomic factual claims. For EACH claim, decide "
            "whether the CONTEXT directly supports it.\n"
            'Reply with ONLY a JSON array like '
            '[{"claim": "...", "supported": true}, ...]. No other text.\n\n'
            f"CONTEXT:\n{ctx}\n\nANSWER: {answer}"
        )
        items = extract_json(raw) or [{"claim": answer, "supported": False}]
        supported = sum(1 for x in items if x.get("supported"))
        return {"score": supported / max(len(items), 1), "items": items}

    res = faithfulness(EXAMPLE["answer"], EXAMPLE["context"])
    print(f"faithfulness = {res['score']:.2f}")
    for it in res["items"]:
        mark = "yes" if it.get("supported") else "no"
        print(f"  [{mark:3s}] {it.get('claim')}")
    """),

    md(r"""
    ### Answer relevancy

    *Does the answer address the question that was asked?*

    Classic RAGAS trick: generate **N candidate questions** *from the
    answer* and measure how similar each is to the original question.
    Intuition: if the answer is on-topic, you can recover something close
    to the original question from it; if the answer drifted, you cannot.

    We score similarity with our embedding model — and reuse the cosine
    we've used throughout the tutorial.
    """),

    code(r"""
    def answer_relevancy(question: str, answer: str, n: int = 3) -> dict:
        gen = judge(
            f"Given this ANSWER, write {n} different questions for which it would be "
            "the correct answer. Reply as a JSON array of strings, nothing else.\n\n"
            f"ANSWER: {answer}"
        )
        gen_qs = extract_json(gen) or []
        if not gen_qs:
            return {"score": 0.0, "generated": []}

        # cosine between each generated question and the original (uses the embedding
        # model from earlier in the notebook -- no API call).
        q_emb = bi.encode([question] + gen_qs, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
        sims  = q_emb[1:] @ q_emb[0]
        return {"score": float(sims.mean()), "generated": list(zip(gen_qs, sims.tolist()))}

    res = answer_relevancy(EXAMPLE["question"], EXAMPLE["answer"])
    print(f"answer relevancy = {res['score']:.3f}\n  generated questions:")
    for q, s in res["generated"]:
        print(f"    cos={s:+.3f}  {q!r}")
    """),

    md(r"""
    ### Context precision

    *Of the chunks we retrieved, how many are actually relevant to the
    question?* Penalises retrieving noise (the orchestra sentence in our
    example).
    """),

    code(r"""
    def context_precision(question: str, context: list[str]) -> dict:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(context))
        raw = judge(
            "For EACH numbered CHUNK below, decide whether it is directly useful "
            "for answering the QUESTION.\n"
            'Reply with ONLY a JSON array like '
            '[{"index": 0, "useful": true}, ...]. No other text.\n\n'
            f"QUESTION: {question}\n\nCHUNKS:\n{numbered}"
        )
        items = extract_json(raw) or []
        useful = sum(1 for x in items if x.get("useful"))
        return {
            "score": useful / max(len(items) or len(context), 1),
            "items": items,
        }

    res = context_precision(EXAMPLE["question"], EXAMPLE["context"])
    print(f"context precision = {res['score']:.2f}")
    for it in res["items"]:
        mark = "yes" if it.get("useful") else "no"
        idx = it.get("index", -1)
        chunk = EXAMPLE["context"][idx] if 0 <= idx < len(EXAMPLE["context"]) else "(?)"
        print(f"  [{mark:3s}] {chunk[:90]}{'...' if len(chunk)>90 else ''}")
    """),

    md(r"""
    ### Context recall

    *Did we retrieve **enough** context to answer the question fully?* Uses
    the gold answer to know what should be there. RAGAS splits the gold
    answer into claims and checks each one against the retrieved context.
    """),

    code(r"""
    def context_recall(gold_answer: str, context: list[str]) -> dict:
        ctx = "\n".join(f"- {c}" for c in context)
        raw = judge(
            "Split the GOLD answer into atomic factual claims. For EACH claim, decide "
            "whether it can be derived from the CONTEXT.\n"
            'Reply with ONLY a JSON array like '
            '[{"claim": "...", "derivable": true}, ...]. No other text.\n\n'
            f"CONTEXT:\n{ctx}\n\nGOLD: {gold_answer}"
        )
        items = extract_json(raw) or [{"claim": gold_answer, "derivable": False}]
        supported = sum(1 for x in items if x.get("derivable"))
        return {"score": supported / max(len(items), 1), "items": items}

    res = context_recall(EXAMPLE["gold"], EXAMPLE["context"])
    print(f"context recall = {res['score']:.2f}")
    for it in res["items"]:
        mark = "yes" if it.get("derivable") else "no"
        print(f"  [{mark:3s}] {it.get('claim')}")
    """),

    md(r"""
    ## 6.5 Putting them together: a triage table

    The point of having *four* metrics rather than one is **diagnostic
    decomposition** — different scores tell you to look in different
    boxes of the pipeline. The interpretation table below is what people
    actually use in practice:

    | faithfulness | ans-relevancy | ctx-precision | ctx-recall | likely cause                                |
    |--------------|---------------|---------------|------------|---------------------------------------------|
    |  high        |    high       |    high       |   high     | the system is working                        |
    |  low         |    high       |    high       |   high     | LLM hallucinates **despite** good context — try a stronger generator, or tighten the prompt |
    |  high        |    low        |    *           |   *        | LLM is being faithful but not *answering* the question — usually a prompt-engineering issue |
    |  *           |    *          |    low        |   high     | retriever grabs too much noise — add a **reranker** (notebook 5) or filter on metadata |
    |  *           |    *          |    high       |   low      | retriever misses the needed info — improve **chunking** (nb 2.4), or hybrid BM25 + bi-encoder |
    |  *           |    *          |    low        |   low      | retrieval is genuinely broken — start from the embedding model and chunking |

    The four metric calls above were each shown individually so the JSON
    judge prompts and verdicts are inspectable. In production you would
    of course wrap them into a single `ragas_like()` function that
    returns all four scores. We intentionally do NOT rerun all four here
    — the free-tier per-minute quota is 5 requests, and we've already
    spent 4 calls above. Re-running them in one cell will trip the
    rate limit and surface a 429.
    """),

    md(r"""
    A final caveat about LLM-as-judge: *the judge is also an LLM, and it
    can be wrong.* In production, calibrate the judge against a small
    human-rated set before you trust its scores, and always run it on
    several seeds / phrasings of the prompt. The numbers above are *signal*,
    not ground truth.

    Next: **notebook 7** uses these metrics as the trigger for *agentic*
    RAG variants — corrective RAG (retry the retriever if context recall is
    low), multi-round RAG, etc.
    """),
]


if __name__ == "__main__":
    path = build_notebook(cells, OUT)
    print(f"wrote {path}")
