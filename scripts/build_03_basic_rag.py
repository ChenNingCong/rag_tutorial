"""Generate tutorial/03_basic_rag.ipynb.

Section 3: a hand-rolled RAG pipeline -- retrieval + prompt assembly + Gemini
generation. No LangChain, no agents. The knowledge base mixes real
sentence-transformers/simple-wiki passages with a few synthetic 'internal
blog posts' so the embedding-vs-metadata distinction and the
with/without-RAG hallucination demo both have something to show.
"""
from pathlib import Path

from _nb_utils import build_notebook, code, md

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorial" / "03_basic_rag.ipynb"


cells = [
    md(r"""
    # 3. Basic RAG — a hand-rolled pipeline

    RAG = **R**etrieval-**A**ugmented **G**eneration. The pattern is just three
    boxes wired in series:

    ```
        query ──► [retriever] ──► top-k chunks ──► [prompt builder] ──► [LLM] ──► answer
                                                                          ▲
                                                              system prompt + chat history
    ```

    No agents. No frameworks. Just three function calls. This notebook builds
    that pipeline once, end-to-end, so the only thing later notebooks have to
    do is *improve each box*.

    What you'll see in this notebook:

    - The crucial distinction between **embedding** (what the model sees) and
      **metadata** (what filtering / display code sees).
    - The **exact prompt string** that we send to the LLM, with retrieved
      chunks interpolated in.
    - A direct head-to-head: same question, **with** and **without** retrieval.

    For the knowledge base we mix two sources:

    - Real Wikipedia passages from
      [`sentence-transformers/simple-wiki`](https://huggingface.co/datasets/sentence-transformers/simple-wiki)
      (the `text` field — "regular" English, not the simplified version) —
      grouped into multi-sentence passages so each "document" is a useful
      retrieval unit.
    - A handful of synthetic "internal blog posts" about a fictional company,
      *Aurora Labs*. Those facts can't appear in the LLM's training data, so
      they make the with/without-RAG hallucination demo crisp.

    References:

    - HuggingFace cookbook: ["Building a RAG system with Hugging Face and Gemini"](https://huggingface.co/learn/cookbook/advanced_rag) — same prompt-assembly pattern.
    - The Google [`google-genai`](https://github.com/googleapis/python-genai) SDK.
    """),

    md(r"""
    ## 3.1 Embedding vs. metadata — they are not the same thing

    People conflate these constantly. They serve completely different roles:

    | dimension      | embedding                                | metadata                            |
    |----------------|------------------------------------------|-------------------------------------|
    | what it is     | a dense vector, learned by a model       | structured key/value fields         |
    | how it's used  | semantic similarity search (vector math) | exact filters / display / citation  |
    | typical fields | (none — it's opaque to a human)          | `source`, `title`, `author`, `date`, `page` |
    | storage        | float array in a vector index            | a row in a SQL/JSON store next to the vector |
    | symptom of confusion | "the embedding doesn't know it was published yesterday" | "I'm doing semantic search on a `year` field" |

    A practical rule:

    > **If a user can spell it, it's metadata. If the meaning matters, it's the embedding.**

    Use metadata for hard filters ("only papers from 2024", "only Spanish
    documents", "exclude internal drafts"). Use embeddings for the actual
    "find things that are *about* X" step. Cite from metadata. Rank with
    embeddings.

    Our knowledge base therefore stores *both* per record. Every entry has:

    - **`text`** — the content that gets embedded.
    - **`doc_id`, `source`, `date`, `title`** — metadata, never embedded.
    """),

    code(r"""
    import warnings; warnings.filterwarnings("ignore")
    from datasets import load_dataset

    # Stream the first N sentences from simple-wiki and group every G consecutive
    # ones into a passage. Consecutive sentences in this dataset come from the
    # same source article, so groups end up topically coherent.
    N_SENTENCES = 150
    GROUP_SIZE = 5

    ds = load_dataset("sentence-transformers/simple-wiki", split="train", streaming=True)
    raw_sentences = [x["text"] for i, x in enumerate(ds) if i < N_SENTENCES]
    print(f"streamed {len(raw_sentences)} sentences from simple-wiki")

    wiki_records = []
    for g in range(0, len(raw_sentences), GROUP_SIZE):
        passage = " ".join(raw_sentences[g:g + GROUP_SIZE])
        wiki_records.append({
            "doc_id": f"wiki-{g // GROUP_SIZE:03d}",
            "source": "simple-wiki",
            "date":   None,
            "title":  f"Simple-Wiki passage #{g // GROUP_SIZE}",
            "text":   passage,
        })

    print(f"-> {len(wiki_records)} passages of ~{GROUP_SIZE} sentences each")
    print("\nfirst passage:")
    print(f"  [{wiki_records[0]['doc_id']}]  {wiki_records[0]['title']}")
    print(f"  {wiki_records[0]['text'][:240]}...")
    """),

    code(r"""
    # A few synthetic 'internal blog posts' about a fictional company.
    # The LLM has NEVER seen these -- so RAG is the only way to answer questions
    # about Aurora Labs accurately.
    aurora_records = [
        {
            "doc_id": "auro-2024-launch",
            "source": "blog",
            "date":   "2024-08-02",
            "title":  "Helios-2 launch notes",
            "text": (
                "Aurora Labs released Helios-2, the follow-up to Helios-1, on July 30 2024. "
                "Helios-2 produces 768-dimensional embeddings and was trained with Matryoshka "
                "representation learning so users can truncate the vector to 256, 384, or 512 "
                "dimensions without losing much quality."
            ),
        },
        {
            "doc_id": "auro-faq-pricing",
            "source": "blog",
            "date":   "2025-01-10",
            "title":  "Aurora API pricing FAQ",
            "text": (
                "The hosted Aurora embedding API charges $0.02 per million tokens for Helios-2. "
                "Customers on the legacy Helios-1 endpoint are billed at $0.05 per million tokens "
                "until that endpoint is sunset on December 31 2025."
            ),
        },
        {
            "doc_id": "auro-faq-context",
            "source": "blog",
            "date":   "2025-02-21",
            "title":  "Context window FAQ",
            "text": (
                "Both Helios-1 and Helios-2 accept up to 8192 input tokens. "
                "Inputs longer than the context window are silently truncated server-side."
            ),
        },
        {
            "doc_id": "auro-blog-finetune",
            "source": "blog",
            "date":   "2025-03-17",
            "title":  "Finetuning Helios-2 on legal text",
            "text": (
                "We finetuned Helios-2 on 1.2 million pairs of legal questions and clauses. "
                "The finetuned checkpoint, Helios-2-Legal, improves recall@10 on contract "
                "retrieval from 0.61 to 0.78."
            ),
        },
    ]

    KB = wiki_records + aurora_records
    print(f"total knowledge base: {len(KB)} records "
          f"({len(wiki_records)} wiki + {len(aurora_records)} blog)")
    """),

    md(r"""
    Notice what we put in each record. `text` is what we'll *embed*. The
    other fields — `doc_id`, `title`, `date`, `source` — are pure metadata.
    The embedding model will never see them; they'll be used for citation
    and for the (optional) filter step.

    ## 3.2 Build the retriever

    Same recipe as notebook 2: embed every document once, embed the query,
    sort by cosine. The only new thing is that we keep the metadata next to
    the vectors so we can both **filter** and **cite**.
    """),

    code(r"""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = SentenceTransformer(EMBED_MODEL)

    # One embedding per record. Note: we only encode the `text` field.
    texts = [r["text"] for r in KB]
    E = encoder.encode(texts, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
    print("embeddings matrix:", E.shape, "dtype", E.dtype)
    """),

    code(r"""
    def retrieve(query: str, *, top_k: int = 3, source: str | None = None):
        '''Return the top_k records most similar to `query`.

        `source`, if given, is a metadata filter applied BEFORE ranking.
        That filter never touches the embedding model -- it just removes
        ineligible rows from the candidate set.
        '''
        candidate_idx = [i for i, r in enumerate(KB)
                         if (source is None or r["source"] == source)]

        q = encoder.encode([query], normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)[0]
        scores = E[candidate_idx] @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[j]), KB[candidate_idx[j]]) for j in order]

    print(">>> query about a synthetic fact, no filter:\n")
    for s, r in retrieve("how big is the embedding produced by Helios-2?", top_k=3):
        print(f"  cos={s:+.3f}  [{r['doc_id']:18s}] src={r['source']:12s}  {r['title']!r}")
        print(f"      -> {r['text'][:180]}...")
    """),

    code(r"""
    # Metadata filter in action -- restrict to Wikipedia passages only.
    print(">>> same query, source='simple-wiki' only:\n")
    for s, r in retrieve("how big is the embedding produced by Helios-2?",
                          top_k=3, source="simple-wiki"):
        print(f"  cos={s:+.3f}  [{r['doc_id']:18s}] src={r['source']:12s}  {r['title']!r}")
        print(f"      -> {r['text'][:180]}...")
    """),

    md(r"""
    Two things just happened that are worth highlighting:

    1. The retriever returned not just text but **the record**. We keep the
       metadata attached so the next step can both interpolate the *content*
       into the prompt *and* render a citation from the *metadata*.
    2. The `source` filter is a plain Python list comprehension. No vector
       math, no embedding lookup. Filters are *outside* the embedding space.
       In the second call we narrowed the candidate set to wiki passages
       only — and watch the scores collapse, because none of those passages
       are *about* Helios-2.

    ## 3.3 Assemble the prompt

    A RAG prompt has three pieces glued together by your code (not by the
    model):

    ```
    [ system instructions       ]   <-- you write this, fixed per app
    [ retrieved chunks (context)]   <-- inserted at runtime, per query
    [ user query                ]   <-- the actual question
    ```

    Two things every working RAG prompt does:

    1. **Tag every chunk with a stable identifier** (we'll use `[doc_id]`)
       so the model can quote it back in its answer. This is the cheapest
       form of citation.
    2. Explicitly tell the model what to do when the context is **silent on
       the question** — otherwise it will happily make something up.

    Below we build the prompt by string concatenation, with absolutely
    nothing hidden. *This is the whole "magic" of RAG.*
    """),

    code(r"""
    SYSTEM_PROMPT = (
        "You are a helpful assistant. Only use information from the CONTEXT block "
        "to answer the QUESTION.\n"
        "If the CONTEXT does not contain the answer, reply exactly: "
        "\"I don't know based on the provided documents.\"\n"
        "Cite each fact with the doc_id in square brackets, e.g. [auro-2024-launch]."
    )

    def build_prompt(query: str, hits) -> str:
        ctx_blocks = []
        for _, r in hits:
            block = (
                f"[{r['doc_id']}] (source={r['source']}, date={r['date']}, title={r['title']!r})\n"
                f"{r['text']}"
            )
            ctx_blocks.append(block)
        context = "\n\n".join(ctx_blocks)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n"
            f"-------\n"
            f"{context}\n"
            f"-------\n\n"
            f"QUESTION: {query}\n"
            f"ANSWER:"
        )

    query = "How big is the Helios-2 embedding, and what training trick lets you truncate it?"
    hits = retrieve(query, top_k=3)
    full_prompt = build_prompt(query, hits)
    print(full_prompt)
    """),

    md(r"""
    Take a moment with that printout. There is **no hidden state**, no
    "chains", no callbacks. The entire interaction with the LLM is a single
    string. Every byte that influences the model's output is visible above.

    When a RAG system misbehaves, this is the *first* thing you print and
    inspect — almost every "the model hallucinated" bug is actually a
    "the retrieved chunks did not contain the answer" bug, which is plainly
    visible in this string.

    ## 3.4 Generation with Gemini

    We use Google's [`google-genai`](https://github.com/googleapis/python-genai)
    SDK. The API key is stored as a single line in `../key.env`. We load
    it into the environment with no extra dependencies.
    """),

    code(r"""
    import os
    from pathlib import Path

    KEY_FILE = Path("..") / "key.env"   # ../key.env relative to the notebook
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = KEY_FILE.read_text(encoding="utf-8").strip()

    from google import genai
    from google.genai import types

    client = genai.Client()    # picks up GEMINI_API_KEY from the environment

    GEN_MODEL = "gemini-3-flash-preview"
    print("Gemini client ready. Model:", GEN_MODEL)
    """),

    code(r"""
    def generate(prompt: str, temperature: float = 0.2) -> str:
        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return resp.text

    answer = generate(full_prompt)
    print(answer)
    """),

    md(r"""
    The answer should cite the relevant `doc_id`s, which is the cheapest way
    to verify that the model is grounding its answer in the retrieved text
    rather than making things up.

    ## 3.5 The complete RAG function

    Put it all together. This is the simplest production-shaped RAG you can
    write — no frameworks needed:
    """),

    code(r"""
    def rag(query: str, *, top_k: int = 3, source: str | None = None,
            show_prompt: bool = False) -> dict:
        hits = retrieve(query, top_k=top_k, source=source)
        prompt = build_prompt(query, hits)
        if show_prompt:
            print("---- PROMPT ----")
            print(prompt)
            print("---- /PROMPT ----\n")
        answer = generate(prompt)
        return {
            "query": query,
            "answer": answer,
            "citations": [r["doc_id"] for _, r in hits],
            "retrieved": [(s, r["doc_id"], r["title"]) for s, r in hits],
        }

    res = rag("When is the Helios-1 API going away and what does it cost until then?")
    print("Q:", res["query"])
    print("A:", res["answer"])
    print("\nretrieved candidates (debug):")
    for s, d, t in res["retrieved"]:
        print(f"  cos={s:+.3f}  [{d}]  {t}")
    """),

    md(r"""
    ## 3.6 With RAG vs. without RAG

    The most concrete way to internalize what RAG buys you is to ask the
    exact same question both ways. We deliberately ask about Aurora Labs —
    a fictional company that does not appear anywhere in the LLM's training
    data — so any specific number it produces without retrieval is, by
    construction, a hallucination.
    """),

    code(r"""
    QUESTION = (
        "How many dimensions does the Helios-2 embedding produce, "
        "and what training trick lets you truncate it without much quality loss?"
    )

    # ----- Without RAG: just hand the question to the LLM.
    bare_answer = generate(
        f"You are a helpful assistant. Answer concisely.\n\nQUESTION: {QUESTION}\nANSWER:"
    )

    # ----- With RAG.
    rag_result = rag(QUESTION)

    print("============ NO RAG ============")
    print(bare_answer.strip())
    print("\n============ WITH RAG ============")
    print(rag_result["answer"].strip())
    print("\nciting:", rag_result["citations"])
    """),

    md(r"""
    Note how:

    - The bare LLM either refuses politely or invents an answer — Aurora Labs
      simply isn't in its training data, so any specific number it commits
      to is a guess.
    - The RAG answer is grounded in the retrieved chunk and pins the
      relevant `doc_id`, so a reader can verify the fact directly.

    For contrast, let's repeat the experiment on a *real* topic that the
    LLM probably already knows. The retrieved Wikipedia passage will mostly
    confirm what the model already had memorised — but the cited answer is
    still strictly better, because it tells the reader *where the fact came
    from*.
    """),

    code(r"""
    res2 = rag("Summarise what the wiki passages say about Patrick Hillery's career.",
                source="simple-wiki", top_k=4)
    print("Q:", res2["query"])
    print("\nA:", res2["answer"])
    print("\nciting:", res2["citations"])
    """),

    md(r"""
    ## 3.7 What we have, and what's missing

    Every part of this pipeline is improvable. Here's the map of where each
    later notebook fits:

    | step               | weakness today                        | covered in   |
    |--------------------|---------------------------------------|--------------|
    | encoder            | "what is an embedding model, really?" | notebook 4   |
    | top-k retrieval    | top-k is noisy on the boundary        | notebook 5 (reranker) |
    | "how good is it?"  | no metric yet                         | notebook 6 (eval) |
    | one-shot pipeline  | no self-correction, no fallback       | notebook 7 (advanced RAG) |
    | failure modes      | chunking, domain gap, query gap       | notebook 8   |
    """),
]


if __name__ == "__main__":
    path = build_notebook(cells, OUT)
    print(f"wrote {path}")
