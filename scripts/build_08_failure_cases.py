"""Generate tutorial/08_failure_cases.ipynb.

Section 8: Common failure modes of a basic RAG system.

- Domain gap: general-purpose embedder underperforms on specialized terminology.
- Bad chunking: under-padded chunks lose context; duplicated boilerplate
  pulls unrelated docs together.
- Question/answer gap: bare queries are dissimilar to answer text; HyDE fixes it.

API budget: exactly 1 Gemini call (the HyDE demo). Failures raise immediately.
"""
from pathlib import Path

from _nb_utils import build_notebook, code, md

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorial" / "08_failure_cases.ipynb"


cells = [
    md(r"""
    # 8. Common failure cases of RAG

    By this point you can build a pipeline that retrieves, ranks, generates,
    and evaluates. What you can't yet do is *diagnose what went wrong* when
    the pipeline misbehaves. This notebook walks through the three failure
    modes that account for the overwhelming majority of "my RAG doesn't
    work" issues, with a small reproduction of each:

    1. **Domain gap** — the embedder was pretrained on general text; your
       corpus is specialized. Vocabulary mismatch causes scores to collapse.
    2. **Bad chunking** — chunks too short, or every chunk shares boilerplate
       that dominates the embedding.
    3. **Question / answer gap** — questions and answers don't share the
       same vocabulary, so cosine similarity is lower than it should be.
       Fixed by [HyDE](https://arxiv.org/abs/2212.10496).

    Each section uses the simple-wiki corpus where possible and the
    embedding model from notebook 2. Only §8.3 calls Gemini (once).
    """),

    code(r"""
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    # The same general-purpose embedder we've used throughout.
    bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # A handful of simple-wiki passages used as the background "in-domain" corpus.
    ds = load_dataset("sentence-transformers/simple-wiki", split="train",
                      streaming=True)
    wiki_sents = [x["text"] for i, x in enumerate(ds) if i < 50]
    wiki_passages = [" ".join(wiki_sents[g:g+5]) for g in range(0, len(wiki_sents), 5)]
    print(f"wiki passages: {len(wiki_passages)}")
    """),

    md(r"""
    ## 8.1 Domain gap

    Embedding models inherit the *vocabulary* of their training corpus.
    `all-MiniLM-L6-v2` was trained on a mix of internet text + paraphrase
    datasets — fine for "everyday English", weak for niche jargon
    (specialized law, advanced medicine, low-resource languages, finance,
    code, …).

    Symptom: when both the query and the documents are in a specialized
    domain that the embedder wasn't trained on, cosine scores **collapse**.
    Everything is moderately similar to everything else; the top-1 isn't
    much higher than rank 20.

    Reproduction: build a tiny "in-domain" subcorpus of medical-jargon
    sentences and ask a domain question. Look at the score distribution
    versus the same setup on plain English.
    """),

    code(r"""
    # A specialized "domain" corpus -- five passages dense in clinical jargon,
    # plus the wiki passages as background distractors.
    medical_corpus = [
        "Allogeneic hematopoietic stem cell transplantation remains the standard "
        "curative therapy for high-risk myelodysplastic syndromes, though "
        "transplant-related mortality limits its use in older patients.",
        "Endothelial dysfunction precedes overt atherosclerosis and is "
        "characterized by impaired nitric-oxide-mediated vasodilation.",
        "Direct oral anticoagulants are non-inferior to warfarin for stroke "
        "prevention in non-valvular atrial fibrillation and reduce intracranial "
        "haemorrhage.",
        "CAR-T cell therapy uses autologous T-lymphocytes engineered to express "
        "a chimeric antigen receptor targeting CD19 on malignant B cells.",
        "Pre-exposure prophylaxis with tenofovir/emtricitabine reduces HIV "
        "acquisition risk by over 90 percent in high-adherence populations.",
    ]
    corpus = medical_corpus + wiki_passages

    D = bi.encode(corpus, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=False)

    # A jargon-heavy query that matches the FIRST medical passage.
    domain_query = "stem cell transplant outcomes in high-risk MDS"

    # A "normal English" query whose gold is in the wiki section.
    plain_query  = "career of Patrick Hillery as President"

    for name, q in [("domain (jargon)", domain_query),
                    ("plain English ", plain_query)]:
        v = bi.encode([q], normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)[0]
        s = D @ v
        order = np.argsort(-s)
        top1 = s[order[0]]; top5 = s[order[4]]
        print(f"\n[{name}] query: {q!r}")
        print(f"  top-1 cosine  = {top1:+.3f}")
        print(f"  top-5 cosine  = {top5:+.3f}")
        print(f"  margin top1 - top5  = {top1 - top5:+.3f}")
        for r, i in enumerate(order[:3], 1):
            print(f"   {r}. cos={s[i]:+.3f}  {corpus[i][:90]}...")
    """),

    md(r"""
    Compare the two queries:

    - For the **plain-English** query, the top-1 score is decisively higher
      than rank 5 — the embedding model is confident.
    - For the **jargon** query, all five top hits cluster within a small
      cosine window. The model knows enough about each sentence to put
      vaguely-relevant ones near the top, but it cannot reliably
      distinguish *which* clinical passage is the right one.

    The signal that a domain-gap problem is present:

    > **The cosine margin between top-1 and top-k is small *and* the
    > absolute scores are low**. That's the model saying "I have no idea —
    > the top-1 is just the best of a bad lot."

    Fixes, in order of cost:

    1. **Use a domain-finetuned model.** `BAAI/bge-large-zh-v1.5` for
       Chinese, `medCPT` or `BiomedNLP-PubMedBERT` for biomed, etc.
    2. **Add classical retrieval (BM25) alongside.** Lexical matching is
       *invariant* to domain — `"CD19"` matches `"CD19"` regardless of how
       the embedder feels about it.
    3. **Finetune your own embedder** on a few thousand pairs from your
       domain. Cheap compared to pretraining, often moves recall by a lot.
    """),

    md(r"""
    ## 8.2 Bad chunking

    Two specific failure modes here:

    1. **Under-padded chunks** — chunks so short that they lose the
       surrounding context that disambiguates their meaning.
    2. **Boilerplate domination** — every chunk shares a long header /
       footer (copyright notice, page numbers, "click here to subscribe"),
       which dominates the embedding and makes all chunks look similar.
    """),

    md(r"""
    ### 8.2a Under-padded chunks lose context

    The same idea phrased as a sentence-fragment vs. a paragraph can land
    in very different places in the embedding space. Look at the cosine
    between a query and (i) a fragment of the gold doc, (ii) the full
    paragraph that fragment came from.
    """),

    code(r"""
    paragraph = (
        "Patrick John 'Paddy' Hillery (1923-2008) was an Irish Fianna Fáil "
        "politician and the sixth President of Ireland from 1976 until 1990. "
        "He had previously served as Minister for Education, Minister for "
        "Labour, Minister for Foreign Affairs, and as European Commissioner "
        "for Social Affairs."
    )
    fragment = "He had previously served as Minister for Education."
    query = "Patrick Hillery's political career and presidency"

    v_para  = bi.encode([paragraph], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)[0]
    v_frag  = bi.encode([fragment],  normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)[0]
    v_query = bi.encode([query],     normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)[0]

    print(f"cosine(query, full paragraph) = {v_para  @ v_query:+.3f}")
    print(f"cosine(query, fragment)       = {v_frag  @ v_query:+.3f}")
    """),

    md(r"""
    The fragment uses the pronoun *"He"* — without the surrounding
    sentence, the embedder has no idea *who* "he" is. The cosine
    plummets. Practical implication: always pad your chunks with enough
    surrounding text that pronouns and references resolve. Most
    production splitters add 50–100 token overlap for exactly this reason.

    ### 8.2b Boilerplate makes every chunk look the same

    Suppose every doc in your KB carries a long header like

    > *"Aurora Labs, Inc. — Confidential. Do not redistribute. Generated
    > on 2024-08-02. Document version 14.2. Customer support: …"*

    Each chunk's embedding is now dominated by tokens shared across every
    doc. Pairwise cosines collapse: even *unrelated* chunks look similar.
    """),

    code(r"""
    HEADER = (
        "AURORA LABS, INC. -- CONFIDENTIAL. Do not redistribute. "
        "Generated 2024-08-02. Document version 14.2. Internal use only. "
        "Contact support@aurora.example for questions. "
    )

    chunks_clean = [
        "Helios-2 produces 768-dimensional embeddings.",
        "Aurora's API charges $0.02 per million tokens for Helios-2.",
        "The cat sat on the mat.",
        "Patrick Hillery was the sixth President of Ireland.",
    ]
    chunks_with_header = [HEADER + c for c in chunks_clean]

    def pairwise_cos(texts):
        v = bi.encode(texts, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        return v @ v.T

    M_clean   = pairwise_cos(chunks_clean)
    M_header  = pairwise_cos(chunks_with_header)

    np.set_printoptions(precision=3, suppress=True)
    print("pairwise cosine, CLEAN chunks:")
    print(M_clean)
    print("\npairwise cosine, chunks with shared boilerplate header:")
    print(M_header)
    print(f"\nmean off-diagonal similarity:")
    print(f"  clean  : {(M_clean.sum() - np.trace(M_clean))/(M_clean.size - len(M_clean)):.3f}")
    print(f"  header : {(M_header.sum()- np.trace(M_header))/(M_header.size- len(M_header)):.3f}")
    """),

    md(r"""
    Notice the shift: with boilerplate, even the cat sentence and the
    Patrick Hillery sentence get pulled into a moderately similar cosine
    region — they "share most of their text". The retriever is now
    fighting against the boilerplate instead of the content.

    Practical implication: **strip boilerplate aggressively at ingest
    time**. PDF headers/footers, navigation menus on HTML pages, license
    banners on source files. Two minutes of preprocessing is worth a lot
    of retrieval quality.

    ## 8.3 The question / answer gap, and HyDE

    The third common failure: questions and answers don't sound alike.

    > Query: *"What did Patrick Hillery do for a living before politics?"*
    > Gold answer text: *"Upon his conferral in 1947 he returned to his native
    > town where he followed in his father's footsteps as a doctor."*

    Almost no vocabulary overlap. A bi-encoder trained on natural-pair
    data does some bridging, but for short keyword queries the gap is
    real.

    **HyDE** ([Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496))
    fixes this without retraining: instead of embedding the *question*,
    ask the LLM to write a *hypothetical answer* and embed *that*. The
    hypothetical answer naturally shares vocabulary with the real answer
    docs, so cosine similarity goes up.

    ```
       question  ──►  [LLM]  ──►  hypothetical answer  ──►  [encoder]  ──►  q'
                                                                            │
                              retrieve(q', corpus) ─► best chunks ─────────►│
    ```

    One Gemini call per query. We compare retrieval *with* and *without*
    HyDE on a single concrete example.
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
    # No retry: if the call hits a 429 / 503, re-run this cell after ~30 seconds.

    def hyde_sketch(question: str) -> str:
        '''Return a 1-2 sentence hypothetical answer to embed.'''
        resp = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=(
                "Write a 1-2 sentence ANSWER to the QUESTION, in the style of an "
                "encyclopedia entry. The answer may be FICTIONAL -- accuracy does "
                "not matter, only that it sounds like a real answer would.\n\n"
                f"QUESTION: {question}\n\nANSWER:"
            ),
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return (resp.text or "").strip()
    """),

    code(r"""
    # We need a gold passage on a topic the LLM has broad general knowledge
    # of, otherwise the hypothetical answer will drift to the wrong concept
    # and HyDE will HURT retrieval (see caveat below).
    # The corpus includes wiki passages PLUS the medical sentences from
    # section 8.1, so there are real medical distractors competing with
    # the sickle-cell gold passage.
    corpus = wiki_passages[:10] + medical_corpus
    gold_text = (
        "Sickle-cell disease (SCD), or sickle-cell anemia, is an autosomal "
        "recessive genetic blood disorder. It is characterized by red blood "
        "cells that assume an abnormal sickle shape, decreasing the cells' "
        "flexibility and resulting in many complications including anemia, "
        "infections, and vaso-occlusive crises."
    )
    corpus.append(gold_text)
    GOLD_IDX = len(corpus) - 1
    D = bi.encode(corpus, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=False)

    # A symptom-level question that uses NONE of the gold's clinical
    # vocabulary ("sickle", "anemia", "red blood cells", "disorder", ...).
    # Exactly the case HyDE was designed for -- bare cosine to the gold
    # passage is weak; the LLM hypothetical bridges the vocabulary gap.
    question = "what genetic condition causes pain crises and tiredness"

    # ---- bare query retrieval ----
    v_q = bi.encode([question], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)[0]
    bare_scores = D @ v_q
    bare_rank   = int(np.argsort(-bare_scores).tolist().index(GOLD_IDX))

    # ---- HyDE retrieval ----
    hypothetical = hyde_sketch(question)
    print("hypothetical answer:\n  ", hypothetical, "\n")
    v_h = bi.encode([hypothetical], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)[0]
    hyde_scores = D @ v_h
    hyde_rank   = int(np.argsort(-hyde_scores).tolist().index(GOLD_IDX))

    print(f"gold doc cosine + rank:")
    print(f"  bare query  : cos={bare_scores[GOLD_IDX]:+.3f}   rank = {bare_rank+1}")
    print(f"  HyDE query  : cos={hyde_scores[GOLD_IDX]:+.3f}   rank = {hyde_rank+1}")
    """),

    md(r"""
    Print the hypothetical answer carefully. It reads like real medical
    prose — *that's the trick*. Whether HyDE actually moves the gold
    doc's rank up in this tiny demo depends on chance details (the
    bare query may already rank the gold at 1 if the corpus is small,
    in which case there's nowhere to go). The point is to see the
    *mechanism*: the embedded hypothetical lives somewhere closer to
    real answer prose than the bare question does.

    **Caveats — HyDE is not free:**

    - **Latency.** One extra LLM call per query.
    - **It can *hurt* when the LLM doesn't know the topic.** An earlier
      draft of this notebook used the question *"What did Patrick
      Hillery do for a living before politics?"*. The LLM hallucinated
      him as a "structural engineer" (he was a doctor), so the
      hypothetical drifted to bridges and infrastructure and HyDE
      *dropped* the gold doc one rank. Lesson: HyDE inherits the LLM's
      mistakes.
    - **Verify with real metrics on real data.** Don't infer that HyDE
      helps your system from a single example; measure recall@k on a
      held-out set before deploying it.
    - It's a *retrieval* trick, not a *generation* trick — the final
      answer to the user is still produced by the standard RAG pipeline
      conditioned on the *real* retrieved docs.

    ## 8.4 Summary — diagnostic table

    Walk into a broken RAG system and check, in this order:

    | symptom                                                                  | likely cause                           | fix in this notebook |
    |--------------------------------------------------------------------------|----------------------------------------|----------------------|
    | low cosine across the board for queries in a niche topic                 | domain gap                             | §8.1 — pick a domain model, add BM25, finetune |
    | retrieved chunk is a sentence fragment with unresolved pronouns          | under-padded chunks                    | §8.2a — increase chunk size / overlap |
    | unrelated docs have suspiciously high pairwise similarity                | shared boilerplate                      | §8.2b — strip headers/footers at ingest |
    | answers exist in the corpus but bare-question retrieval ranks them low   | question/answer vocabulary gap         | §8.3 — HyDE |
    | the model invents facts despite a relevant top-k                         | generator hallucinates (not retrieval) | nb 6 faithfulness |
    | the model can't pull together two facts that exist in different chunks   | one-shot retrieval                     | nb 7 multi-round RAG |

    The discipline that ties all of this together: **measure first, then
    fix**. Notebook 6 gave you the metrics; this notebook gave you the
    failure-mode catalogue. Most "we need a bigger model" intuitions, when
    actually measured, turn out to be a chunking or boilerplate problem in
    disguise.
    """),
]


if __name__ == "__main__":
    path = build_notebook(cells, OUT)
    print(f"wrote {path}")
