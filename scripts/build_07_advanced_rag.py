"""Generate tutorial/07_advanced_rag.ipynb.

Section 7: Advanced RAG patterns -- corrective RAG, multi-round RAG, agentic RAG.

API-budget note: each pattern below is illustrated with exactly ONE end-to-end
example and the smallest number of Gemini calls that still shows the
mechanism. Total calls in this notebook: ~8 (rate-limit safe on the free tier).
"""
from pathlib import Path

from _nb_utils import build_notebook, code, md

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorial" / "07_advanced_rag.ipynb"


cells = [
    md(r"""
    # 7. Advanced RAG patterns

    The notebook 3 pipeline is "fixed": retrieve → assemble → generate, once.
    That works when the retrieval is good and the question is well-formed.
    It breaks when either of those assumptions slips. This notebook walks
    through three patterns that add **structure** around the basic pipeline
    to handle those cases:

    | pattern              | what it adds                                     | when it helps                              |
    |----------------------|--------------------------------------------------|--------------------------------------------|
    | **Corrective RAG**   | grade retrieved context, retry if it's bad       | retriever sometimes misses                 |
    | **Multi-round RAG**  | decompose into sub-questions, retrieve each      | multi-hop / compound questions             |
    | **Agentic RAG**      | LLM picks the next tool call (retrieve / answer) | open-ended workflows, mixed-source queries |

    None of these are *different RAG algorithms* — they're all the
    `retrieve → prompt → generate` primitive of notebook 3, scheduled in a
    smarter way. We implement each one manually so you can see the loop.

    > **A note on API budget.** Each pattern below uses 2–3 Gemini calls,
    > total ~8 for the whole notebook. Re-runs are cheap.

    References:

    - Corrective RAG paper: <https://arxiv.org/abs/2401.15884>
    - LangGraph's [Adaptive RAG tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/) — same patterns, framework version.
    """),

    md(r"""
    ## 7.0 Shared setup — retriever + Gemini

    Same toy knowledge base as notebook 3 (mix of simple-wiki passages and
    a few synthetic Aurora Labs blog entries), same encoder, same Gemini
    client. We rebuild it once here so the notebook stands alone.
    """),

    code(r"""
    import os, re, time, json, warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")

    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    # --- knowledge base ---------------------------------------------------
    ds = load_dataset("sentence-transformers/simple-wiki", split="train",
                      streaming=True)
    sentences = [x["text"] for i, x in enumerate(ds) if i < 150]
    wiki_records = [
        {
            "doc_id": f"wiki-{g//5:03d}",
            "source": "simple-wiki",
            "text": " ".join(sentences[g:g+5]),
        }
        for g in range(0, len(sentences), 5)
    ]
    aurora_records = [
        {"doc_id": "auro-launch", "source": "blog", "text":
         "Aurora Labs released Helios-2 on July 30 2024. It produces "
         "768-dimensional embeddings and was trained with Matryoshka "
         "representation learning."},
        {"doc_id": "auro-pricing", "source": "blog", "text":
         "Aurora's API charges $0.02 per million tokens for Helios-2 and "
         "$0.05 per million tokens for the legacy Helios-1 endpoint, which "
         "is sunset on December 31 2025."},
        {"doc_id": "auro-rerank", "source": "blog", "text":
         "Aurora's 2025 roadmap includes Helios-Rank, a cross-encoder "
         "reranker for use on top of Helios-2 candidates."},
    ]
    KB = wiki_records + aurora_records
    print(f"KB: {len(KB)} records ({len(wiki_records)} wiki + {len(aurora_records)} blog)")
    """),

    code(r"""
    # --- encoder & retrieve() ---------------------------------------------
    bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = bi.encode([r["text"] for r in KB], normalize_embeddings=True,
                  convert_to_numpy=True, show_progress_bar=False)

    def retrieve(query: str, top_k: int = 3):
        q = bi.encode([query], normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)[0]
        scores = E @ q
        order = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), KB[i]) for i in order]

    print(retrieve("Helios-2 embedding size", top_k=2)[0][1])
    """),

    code(r"""
    # --- Gemini client ---------------------------------------------------
    # No retry wrapper: if Gemini hits a 429 / 503 the cell raises immediately
    # and you can re-run it after waiting ~30 seconds. A silent retry would
    # make a hung-looking cell that was actually just sleeping.
    KEY_FILE = Path("..") / "key.env"
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = KEY_FILE.read_text(encoding="utf-8").strip()

    from google import genai
    from google.genai import types

    client = genai.Client()
    GEN_MODEL = "gemini-3-flash-preview"

    def llm(prompt: str, *, temperature: float = 0.2) -> str:
        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return (resp.text or "").strip()

    print("LLM ready, model:", GEN_MODEL)
    """),

    md(r"""
    ## 7.1 Corrective RAG (CRAG)

    The motivating observation: the retriever sometimes misses. If the
    top-k chunks don't actually contain the answer, the generator either
    refuses or hallucinates. **Corrective RAG** adds a *grading* step
    after retrieval — if the context looks unreliable, the pipeline
    *rewrites the query* and retries.

    The paper's full pipeline grades each doc as Correct / Ambiguous /
    Incorrect and falls back to web search. Our minimal version captures
    the *idea* with a single judge call:

    ```
       query ──► retrieve ──► [grade context] ──► good?  ── yes ──► answer
                                                   │
                                                   no
                                                   │
                                                   ▼
                                         rewrite query ──► retrieve ──► answer
    ```

    Three Gemini calls in the worst case: grade, rewrite, answer.
    """),

    code(r"""
    def format_context(hits):
        return "\n\n".join(f"[{r['doc_id']}] {r['text']}" for _, r in hits)

    def grade_context(query: str, hits) -> str:
        '''Return 'good' or 'bad'. One call. Constrain to a single token.'''
        ctx = format_context(hits)
        verdict = llm(
            "You decide whether the retrieved CONTEXT contains the answer "
            "to the QUESTION.\n"
            "Reply with a single word: 'good' or 'bad'. Nothing else.\n\n"
            f"QUESTION: {query}\n\nCONTEXT:\n{ctx}",
            temperature=0.0,
        ).lower()
        return "good" if verdict.startswith("good") else "bad"

    def rewrite_query(query: str) -> str:
        return llm(
            "Rewrite the QUERY into a more search-friendly form. Use specific keywords. "
            "Reply with ONLY the rewritten query, no quotes, no preamble.\n\n"
            f"QUERY: {query}",
            temperature=0.0,
        )

    def answer_with_context(query: str, hits) -> str:
        ctx = format_context(hits)
        return llm(
            "Answer the QUESTION using ONLY the CONTEXT. If the CONTEXT is "
            "insufficient, say so. Cite chunks like [doc_id].\n\n"
            f"CONTEXT:\n{ctx}\n\nQUESTION: {query}",
            temperature=0.2,
        )

    def crag(query: str, *, top_k: int = 3, trace: bool = True):
        hits = retrieve(query, top_k=top_k)
        if trace:
            print(f"[step 1] retrieved {[r['doc_id'] for _, r in hits]}")

        verdict = grade_context(query, hits)
        if trace: print(f"[step 2] grade = {verdict}")
        if verdict == "good":
            return {"answer": answer_with_context(query, hits),
                    "retries": 0, "used_query": query}

        new_q = rewrite_query(query)
        if trace: print(f"[step 3] rewritten query = {new_q!r}")
        hits = retrieve(new_q, top_k=top_k)
        if trace:
            print(f"[step 4] re-retrieved {[r['doc_id'] for _, r in hits]}")
        return {"answer": answer_with_context(new_q, hits),
                "retries": 1, "used_query": new_q}

    # A query phrased awkwardly so the bi-encoder's first pass is shaky.
    res = crag("dimension count of the new model from that aurora startup")
    print("\n--- ANSWER ---\n", res["answer"])
    """),

    md(r"""
    The interesting line in the trace is `[step 2] grade = ...`. When the
    context is judged *bad*, the second retrieval (after rewriting the query)
    typically pulls in the right doc. In production, the grader is the
    bottleneck — a wrong "good" verdict leaves you with the original bad
    context; a wrong "bad" verdict wastes a roundtrip but doesn't break the
    answer.

    ## 7.2 Multi-round / multi-query RAG

    Some questions don't have a single relevant chunk — they need
    information from several. A retriever called once with the full
    question picks the *closest* chunk to the whole question, which often
    isn't *any* of the chunks needed.

    The fix: **decompose** the question into sub-questions, retrieve for
    each independently, then have the LLM synthesize using all retrieved
    chunks.

    ```
                ┌────► retrieve(sub-q 1) ─┐
       query ──►│────► retrieve(sub-q 2) ─├──► merged context ──► answer
                └────► retrieve(sub-q 3) ─┘
    ```

    Two Gemini calls: decompose + answer.
    """),

    code(r"""
    def decompose(question: str, n: int = 3) -> list[str]:
        '''Ask the LLM to break `question` into <= n sub-questions.'''
        raw = llm(
            f"Decompose this QUESTION into up to {n} smaller sub-questions, each "
            "of which can be answered by retrieving a single passage. "
            "Reply as a JSON array of strings ONLY.\n\n"
            f"QUESTION: {question}",
            temperature=0.0,
        )
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else [question]

    def multi_round_rag(question: str, *, per_sub_k: int = 2):
        sub_qs = decompose(question)
        print("sub-questions:")
        for s in sub_qs:
            print(f"  - {s}")

        # Retrieve top-k per sub-question, then dedupe by doc_id.
        seen, merged = set(), []
        for s in sub_qs:
            for score, r in retrieve(s, top_k=per_sub_k):
                if r["doc_id"] not in seen:
                    seen.add(r["doc_id"])
                    merged.append((score, r))
        print(f"\nmerged context ({len(merged)} unique chunks):")
        for s, r in merged:
            print(f"  cos={s:+.3f}  [{r['doc_id']}]  {r['text'][:80]}...")

        ans = answer_with_context(question, merged)
        return {"sub_qs": sub_qs, "answer": ans, "chunks": merged}

    res = multi_round_rag(
        "What size are the embeddings produced by Helios-2 and what "
        "reranker is Aurora planning to release next?"
    )
    print("\n--- ANSWER ---\n", res["answer"])
    """),

    md(r"""
    Decomposing the question is what gets `auro-launch` (embedding size) **and**
    `auro-rerank` (reranker plans) into the same context. A single retrieval
    on the full question would surface only the closer of those two.

    Decomposition has a real cost in surprise — the sub-questions the LLM
    invents are often *narrower or odder* than what you'd write by hand —
    so log them and inspect.

    ## 7.3 Agentic RAG (a minimal ReAct loop)

    Agentic RAG hands the *scheduling* over to the model. Instead of a
    fixed graph (retrieve → grade → maybe retry → answer), the model
    picks the next action at every step from a small set of tools:

    - `retrieve(query)` — call the retriever with this query.
    - `answer(text)` — terminate with this final answer.

    We expose those tools by giving the model a strict text grammar:
    every turn it has to emit either `ACTION: retrieve("...")` or
    `ACTION: answer("...")`. We parse the response, dispatch the tool,
    and feed the result back in the next turn.

    This is a stripped-down version of the [ReAct
    pattern](https://arxiv.org/abs/2210.03629) — the same idea behind
    every "agent" framework. We cap the loop at 3 steps so a single
    misbehaving call cannot melt your free-tier quota.
    """),

    code(r"""
    AGENT_SYSTEM = (
        "You are an information-gathering agent with access to a private "
        "knowledge base via a tool.\n"
        "On each turn you MUST output EXACTLY ONE line in one of these forms:\n"
        '  ACTION: retrieve("...your search query...")\n'
        '  ACTION: answer("...final answer with [doc_id] citations...")\n'
        "Choose 'retrieve' if you need more information; choose 'answer' once "
        "the OBSERVATIONS contain enough to answer the question.\n"
        "Never invent facts. If after retrieving the context is insufficient, "
        "answer with: 'I don't know based on the available documents.'"
    )

    ACTION_RE = re.compile(r'ACTION:\s*(retrieve|answer)\(\s*"(.+?)"\s*\)',
                           re.DOTALL)

    def run_agent(question: str, *, max_steps: int = 3, trace: bool = True):
        history = [f"QUESTION: {question}"]
        for step in range(1, max_steps + 1):
            prompt = AGENT_SYSTEM + "\n\n" + "\n\n".join(history) + "\n\nWhat do you do?"
            resp = llm(prompt, temperature=0.0)
            if trace: print(f"\n[step {step}] {resp.strip()}")
            m = ACTION_RE.search(resp)
            if not m:
                return {"answer": "(agent emitted unparseable output)",
                        "steps": step, "history": history}
            action, arg = m.group(1), m.group(2)
            if action == "answer":
                return {"answer": arg, "steps": step, "history": history}
            # otherwise it's a retrieve:
            hits = retrieve(arg, top_k=2)
            obs = " ; ".join(f"[{r['doc_id']}] {r['text'][:160]}" for _, r in hits)
            history.append(f"OBSERVATION (after retrieve {arg!r}): {obs}")
            if trace: print(f"           OBSERVATION: {obs[:200]}...")
        return {"answer": "(max steps reached)",
                "steps": max_steps, "history": history}

    out = run_agent("How much does Aurora charge for Helios-2 per million tokens?")
    print("\n--- FINAL ANSWER ---\n", out["answer"])
    """),

    md(r"""
    Watch the trace carefully. A well-behaved agent run looks like:

    1. The model picks a `retrieve(...)` call with a focused query.
    2. The observation comes back from the retriever (a small chunk).
    3. The model decides whether it has enough — if yes, `answer(...)`.

    A misbehaved run looks like: the model retrieves and retrieves but
    never produces an `answer(...)` line. That's why we cap the loop.

    ## 7.4 Which pattern when?

    Agentic RAG is fashionable but *expensive* and *non-deterministic*.
    Most production RAG systems do not need an agent — they need a
    well-tuned fixed pipeline with one or two of the simpler patterns
    above. Rough decision rule:

    | situation                                                       | use                       |
    |-----------------------------------------------------------------|---------------------------|
    | retrieval is reliable; questions are mostly atomic              | basic RAG (nb 3)           |
    | retrieval misses sometimes; latency budget allows a retry        | **corrective RAG**         |
    | questions are multi-hop (compare X and Y, list facts about Z)   | **multi-round RAG**        |
    | the same workflow needs to mix tools (KB + web search + calc)   | **agentic RAG**            |

    The patterns also compose. A common real-world stack is:
    *multi-round decomposition → corrective retrieval per sub-query → fixed
    synthesis*. Each layer is one of the simple primitives above.
    """),
]


if __name__ == "__main__":
    path = build_notebook(cells, OUT)
    print(f"wrote {path}")
