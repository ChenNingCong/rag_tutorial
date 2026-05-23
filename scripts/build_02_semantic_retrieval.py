"""Generate tutorial/02_semantic_retrieval.ipynb.

Section 2: Semantic retrieval — from tokens to vectors, cosine ranking,
side-by-side with Jaccard, then chunking and preprocessing.
"""
from pathlib import Path

from _nb_utils import build_notebook, code, md

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorial" / "02_semantic_retrieval.ipynb"


cells = [
    md(r"""
    # 2. Semantic retrieval

    In notebook 1 we saw that every classical retrieval score is an inner product
    on a *hand-designed* vector (binary BoW, TF-IDF, BM25). They all share one
    problem: the coordinates are tied to surface tokens, so "cat" and "feline"
    look completely unrelated.

    The fix is conceptually simple — but algebraically the same: replace the
    sparse hand-designed vector with a **dense, learned** vector. Each
    coordinate no longer corresponds to a word; it corresponds to a *latent
    feature* the model learned to extract during pretraining. Two pieces of
    text that *mean* similar things end up at similar points in the embedding
    space, even if they share no words.

    What we'll do here:

    1. **From tokens to a vector** — the low-level transformer plumbing
       (tokenizer, hidden states, pooling) using `transformers` directly.
       Then the convenient `sentence-transformers` wrapper.
    2. **Ranking** a small corpus with cosine similarity.
    3. **Jaccard vs. cosine** on a query with synonyms — the classical
       method *cannot* answer it; the embedding model can.
    4. **Chunking and preprocessing** — how long inputs get cut into
       retrievable pieces.

    References:

    - HuggingFace blog: ["Getting started with embeddings"](https://huggingface.co/blog/getting-started-with-embeddings)
    - HuggingFace cookbook: ["Advanced RAG"](https://huggingface.co/learn/cookbook/advanced_rag)
    - sentence-transformers docs: <https://www.sbert.net/>
    """),

    md(r"""
    ## 2.1 From a sentence to a vector

    The model we'll use is
    [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    — a 22 M parameter MiniLM trained with contrastive learning on a billion
    sentence pairs. It's the standard "tutorial" embedder: tiny, CPU-friendly,
    and surprisingly strong.

    There are three steps that get a piece of text to a vector:

    1. **Tokenization** — text → list of subword IDs (plus an attention mask
       saying which positions are real vs. padding).
    2. **Forward pass** — the transformer produces one hidden vector *per
       token*. For an input of length $L$, the output tensor shape is
       `(batch, L, hidden_dim)`.
    3. **Pooling** — collapse the $L$ per-token vectors into *one* vector
       per sentence. The standard recipe for sentence-transformers is
       **mean pooling** over real (non-pad) tokens.

    Let's walk through each step explicitly first, then use the high-level
    wrapper that hides them.
    """),

    code(r"""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    import torch
    from transformers import AutoTokenizer, AutoModel

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    print(model.config.model_type, "hidden_size =", model.config.hidden_size)
    """),

    md(r"""
    ### Step 1 — tokenization

    The tokenizer is a *WordPiece* tokenizer with a vocab of ~30k subwords.
    Notice how rare words get split into multiple pieces, and how `[CLS]` /
    `[SEP]` special tokens get prepended/appended.
    """),

    code(r"""
    examples = [
        "the cat sat on the mat",
        "neural embeddings rule",
    ]

    enc = tokenizer(examples, padding=True, return_tensors="pt")
    print("input_ids shape   :", tuple(enc["input_ids"].shape),
          "  (batch, padded_seq_len)")
    print("attention_mask    :")
    print(enc["attention_mask"])
    print()
    for i, ex in enumerate(examples):
        ids = enc["input_ids"][i].tolist()
        print(f"  {ex!r}")
        print(f"   ids    = {ids}")
        print(f"   tokens = {tokenizer.convert_ids_to_tokens(ids)}")
    """),

    md(r"""
    Two things to look at:

    - `input_ids` has shape `(batch=2, seq_len=8)`. The shorter sentence was
      **padded** with token id `0` (`[PAD]`) so both fit in one tensor.
    - `attention_mask` is `1` for real tokens, `0` for padding. We need it for
      two reasons: (a) the transformer ignores padded positions, and
      (b) when we mean-pool we must average over real positions only.

    ### Step 2 — forward pass to per-token vectors
    """),

    code(r"""
    with torch.no_grad():
        out = model(**enc)

    H = out.last_hidden_state  # (batch, seq_len, hidden_dim)
    print("last_hidden_state shape:", tuple(H.shape))
    print(f"-> one {H.shape[-1]}-dim vector per token, for every sentence in the batch")

    # Snapshot of the first 5 dims of the [CLS] token vector of sentence 0:
    print("\nfirst 5 dims of token 0 ([CLS]) of sentence 0:")
    print(H[0, 0, :5])
    """),

    md(r"""
    ### Step 3 — mean pooling

    `all-MiniLM-L6-v2` was trained with **mean pooling**, not `[CLS]`. The
    standard recipe (it appears in every sentence-transformers config):

    1. Multiply the hidden states by the attention mask so padded positions
       contribute zero.
    2. Sum along the sequence dimension.
    3. Divide by the number of real tokens.
    4. **Normalize** the resulting vector to unit length — so the dot product
       *is* the cosine similarity.

    These four lines are the entire "embedding model API" once you peel back
    the wrappers.
    """),

    code(r"""
    import torch.nn.functional as F

    def mean_pool(hidden, attn_mask):
        mask = attn_mask.unsqueeze(-1).float()       # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)          # (B, hidden)
        counts = mask.sum(dim=1).clamp(min=1e-9)     # (B, 1)
        return summed / counts

    pooled = mean_pool(H, enc["attention_mask"])
    embeddings = F.normalize(pooled, p=2, dim=1)

    print("pooled shape  :", tuple(pooled.shape))
    print("normalized?    ", torch.linalg.norm(embeddings, dim=1))
    print("first 5 dims of the sentence embedding for example 0:")
    print(embeddings[0, :5])
    """),

    md(r"""
    ### Same thing, with the wrapper

    `sentence-transformers` packages the three steps above (tokenize → forward
    → pool → normalize) into a one-liner. From here on we'll use it.
    """),

    code(r"""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL_ID)
    v = encoder.encode(examples, normalize_embeddings=True, convert_to_numpy=True)
    print("encoder.encode -> numpy array of shape", v.shape)

    # Sanity: matches our manual pipeline.
    import numpy as np
    print("max abs difference vs manual pipeline:",
          float(np.abs(v - embeddings.numpy()).max()))
    """),

    md(r"""
    Two takeaways from §2.1:

    - A "sentence embedding" is just **one dense vector** per piece of text.
      For this model that vector lives in $\mathbb{R}^{384}$.
    - Embeddings are conventionally **L2-normalized** so cosine similarity
      reduces to a plain dot product:
      $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|} = \mathbf{a}\cdot\mathbf{b}$
      when $\|\mathbf{a}\| = \|\mathbf{b}\| = 1$.
    """),

    md(r"""
    ## 2.2 Ranking a small corpus

    Now we assume the documents are already chunked (we'll cover chunking in
    §2.4) and walk through the standard retrieve-by-cosine recipe:

    1. Embed every document **once**, store the matrix `E ∈ R^{N × d}`.
    2. At query time, embed the query → `q ∈ R^d`.
    3. Score = `E @ q` (one dot product per document).
    4. Sort, take top-k.

    That `E @ q` is the entire "vector database lookup" — modulo speed tricks
    (FAISS, HNSW, IVF) for large corpora.
    """),

    code(r"""
    # A toy corpus deliberately seeded with synonyms / paraphrases.
    corpus = [
        "the cat sat on the mat",                                # d0  pet, exact words
        "a feline rested on a rug",                              # d1  paraphrase of d0 with NO shared content words
        "the cat played with the dog",                           # d2  pet, two animals
        "a puppy and a kitten were chasing each other",          # d3  pet, full synonyms
        "neural networks learn vector representations",          # d4  ML
        "vector search uses dot products to rank documents",     # d5  ML / retrieval
        "transformers have revolutionised natural language processing",  # d6 ML
        "I went to the bakery and bought fresh bread",           # d7  totally off-topic
    ]

    import numpy as np
    E = encoder.encode(corpus, normalize_embeddings=True, convert_to_numpy=True)
    print("corpus embedding matrix E:", E.shape)
    """),

    code(r"""
    def search(query: str, top_k: int = 5):
        q = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = E @ q
        order = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i]), corpus[i]) for i in order]

    for q in [
        "a kitten napping on a carpet",
        "how do embeddings work",
        "what should I eat for breakfast",
    ]:
        print(f"\n>>> query: {q!r}")
        for rank, (i, s, t) in enumerate(search(q), 1):
            print(f"  {rank}. d{i}  cos={s:+.3f}   {t!r}")
    """),

    md(r"""
    The first query is the interesting one. *None* of its content words —
    "kitten", "napping", "carpet" — appear in `d0` ("the cat sat on the mat")
    or `d1` ("a feline rested on a rug"). Jaccard / BM25 would score both
    those documents at **zero**. The embedding model puts them at the top.

    That is the entire pitch of semantic retrieval, in one example.
    """),

    md(r"""
    ## 2.3 Side-by-side: Jaccard vs. cosine

    Let's make that pitch quantitative. For the query above, compute every
    score with both methods and look at them next to each other.
    """),

    code(r"""
    import re
    import numpy as np

    def tokenize(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def jaccard(a, b):
        A, B = set(a), set(b)
        return len(A & B) / max(len(A | B), 1)

    query = "a kitten napping on a carpet"
    q_emb = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    rows = []
    for i, d in enumerate(corpus):
        rows.append({
            "doc": f"d{i}",
            "jaccard": jaccard(tokenize(query), tokenize(d)),
            "cosine":  float(E[i] @ q_emb),
            "text": d,
        })

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("cosine", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    """),

    md(r"""
    - The two top semantic matches (`d1` "feline rested on a rug",
      `d3` "puppy and kitten chasing") have **Jaccard = 0**. They share no
      content words with the query. Classical lexical retrieval simply
      cannot find them.
    - The unrelated bakery sentence `d7` scores low under cosine —
      directionally aligned with the query in the embedding space matters
      more than incidental shared stopwords like "a".

    A practical aside that bites people: hybrid retrieval (BM25 ∪ embedding)
    is still the strongest baseline in production. BM25 catches the
    proper-noun / exact-identifier matches that embeddings sometimes blur.
    Don't throw classical retrieval away — make it complementary.
    """),

    md(r"""
    ## 2.4 Chunking, padding, and preprocessing

    Real documents are too long to embed whole. They have to be cut into
    *chunks* — overlapping windows of text — before they go through the
    encoder. Three things to keep in mind:

    1. **Model context limit.** MiniLM accepts at most 256 (sometimes 512)
       tokens. Anything longer gets *silently truncated*: text past the
       cutoff is dropped, not embedded.
    2. **Chunks are the unit of retrieval.** Two facts inside the *same* chunk
       are retrieved together — bad if they're unrelated. Two facts split
       across chunks are retrieved separately — bad if they need each other.
       Chunking is the most common knob that controls retrieval quality.
    3. **Padding** is a *batching* concern, not a *semantics* concern.
       Padding lets you stack variable-length sentences into one tensor; the
       attention mask makes sure the model ignores the pad tokens (we saw this
       in §2.1).

    Three common chunking strategies:

    - **Fixed-token windows** with overlap (simple, fast, language-agnostic).
    - **Sentence-based** (one chunk per sentence, optionally merged until a
      target length).
    - **Recursive / structural** — try to split on paragraph, then sentence,
      then word — the strategy popularized by LangChain's
      `RecursiveCharacterTextSplitter` and used in the HF cookbook.
    """),

    code(r"""
    # A sample document — the first paragraphs of a "tutorial" markdown file.
    # We'll treat the markdown structure as the natural boundary.
    DOC = '''# Introduction to Retrieval-Augmented Generation

    Retrieval-augmented generation (RAG) is a technique that combines a
    retrieval step with a language model. The retriever finds documents
    relevant to a user query, and the language model uses those documents
    as context when generating its response.

    ## Why use RAG?

    Language models trained on a fixed corpus cannot know about events that
    happened after their cutoff date, and they tend to hallucinate facts
    they don't know. RAG sidesteps both problems by injecting fresh,
    grounded text into the prompt at inference time.

    ## Components

    A RAG system has three moving parts. First, an embedding model that
    turns text into vectors. Second, a vector index that stores those
    vectors and supports fast nearest-neighbour search. Third, a generator
    (an LLM) that conditions its answer on the retrieved snippets.
    '''
    print(DOC[:200], "...")
    """),

    code(r"""
    # ---- (a) fixed token windows, with overlap ----
    def chunk_fixed_tokens(text, tokenizer, chunk_size=32, overlap=8):
        ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        step = chunk_size - overlap
        for start in range(0, len(ids), step):
            window = ids[start:start + chunk_size]
            if not window:
                break
            chunks.append(tokenizer.decode(window))
            if start + chunk_size >= len(ids):
                break
        return chunks

    fixed_chunks = chunk_fixed_tokens(DOC, tokenizer, chunk_size=32, overlap=8)
    print(f"fixed-window chunks: {len(fixed_chunks)} pieces\n")
    for i, c in enumerate(fixed_chunks):
        print(f"  [{i}] {c!r}")
    """),

    code(r"""
    # ---- (b) sentence-based ----
    # The naive splitter -- good enough for clean prose, breaks on Mr./etc.
    import re

    def chunk_sentences(text):
        # First split on blank lines (paragraph), then on sentence-ending punctuation.
        out = []
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            for s in re.split(r"(?<=[.!?])\s+", para):
                s = s.strip()
                if s:
                    out.append(s)
        return out

    sent_chunks = chunk_sentences(DOC)
    print(f"sentence chunks: {len(sent_chunks)} pieces\n")
    for i, c in enumerate(sent_chunks):
        print(f"  [{i}] {c}")
    """),

    code(r"""
    # ---- (c) recursive / structural ----
    # Try to split on the largest separator that fits, fall back to smaller ones.
    def chunk_recursive(text, max_chars=200, separators=("\n\n", "\n", ". ", " ")):
        if len(text) <= max_chars:
            return [text.strip()]
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks, buf = [], ""
                for p in parts:
                    candidate = (buf + sep + p) if buf else p
                    if len(candidate) <= max_chars:
                        buf = candidate
                    else:
                        if buf:
                            chunks.append(buf.strip())
                        if len(p) > max_chars:
                            chunks.extend(chunk_recursive(p, max_chars, separators))
                            buf = ""
                        else:
                            buf = p
                if buf:
                    chunks.append(buf.strip())
                return [c for c in chunks if c]
        # Fallback: hard slice.
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    rec_chunks = chunk_recursive(DOC, max_chars=160)
    print(f"recursive chunks: {len(rec_chunks)} pieces\n")
    for i, c in enumerate(rec_chunks):
        print(f"  [{i}] {c!r}")
    """),

    md(r"""
    ### Does the choice of chunker actually affect retrieval?

    Yes — sometimes a lot. Let's ask the same factual question against each
    chunking and see which strategy lets us recover the *right* paragraph.
    """),

    code(r"""
    import numpy as np

    question = "What are the three components of a RAG system?"

    def best_chunk(chunks, question):
        E_c = encoder.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        q   = encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = E_c @ q
        i = int(scores.argmax())
        return i, float(scores[i]), chunks[i]

    for name, chunks in [
        ("fixed-tokens",  fixed_chunks),
        ("sentences",     sent_chunks),
        ("recursive",     rec_chunks),
    ]:
        i, s, c = best_chunk(chunks, question)
        print(f"\n[{name}]  best chunk = #{i}  cosine = {s:+.3f}")
        print(f"  {c}")
    """),

    md(r"""
    All three strategies surface a chunk about the "three moving parts" of
    a RAG system — but with very different fidelity. The
    **fixed-window** chunker cuts mid-sentence, so its best chunk is a
    fragment. The **sentence** chunker gives the cleanest single sentence
    but loses the "first / second / third" structure unless you concatenate.
    The **recursive** chunker keeps the entire paragraph intact, which is
    usually the sweet spot for prose.

    Two production-level rules of thumb people learn the hard way:

    - Use enough **overlap** that information at chunk boundaries is not
      lost. Common defaults: chunk 200–800 tokens, overlap 50–100.
    - Match the chunker to the **document structure**. Code uses function
      boundaries; markdown uses headings; legal text uses sections. Generic
      RAG libraries (LangChain, LlamaIndex) ship dozens of structural
      splitters for this reason.

    We'll see in notebook 8 that bad chunking is the **single most common
    cause of RAG failure** in practice — much more so than the choice of
    embedding model.

    ---

    Next: **Notebook 3** wires this retriever to a language model to get our
    first end-to-end RAG pipeline.
    """),
]


if __name__ == "__main__":
    path = build_notebook(cells, OUT)
    print(f"wrote {path}")
