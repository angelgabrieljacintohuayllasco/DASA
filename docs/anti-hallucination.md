# Anti-Hallucination Guarantee

## What is an AI Hallucination?

A hallucination occurs when a language model generates factually incorrect information with apparent confidence. This happens because LLMs are **probabilistic sequence predictors** — they predict the next token based on statistical patterns learned from training data, not from a ground-truth lookup.

If the model learned that "egg recipes usually include oregano", it may add oregano to an egg recipe even when asked to answer about a specific database that doesn't mention oregano.

## How DASA Eliminates Hallucinations

DASA's guarantee is **structural, not heuristic**. It does not use:
- Temperature=0 (still probabilistic)
- System prompts telling the model "don't hallucinate" (ignored by models)
- Output validation filters (catch-after-the-fact, not prevention)

Instead, DASA **architecturally prevents** the generation of non-retrieved content:

### Layer 1: Agent A cannot invent

Agent A performs vector similarity search. It computes `Fragment.text` by reading bytes from a JSON record or binary shard file. There is no generation step — only read operations.

### Layer 2: Agent B's vocabulary is locked

The `StatisticalRewriter` builds its output as follows:

1. It splits fragment texts into sentences.
2. It selects the highest-scoring sentences (by keyword overlap).
3. It connects those sentences with a **fixed, pre-approved connector list**.

The connector list is defined as a Python constant. No word outside this list or outside the fragment pool can appear in the output. This is enforced by the code structure, not by a prompt.

### Formal Property

Let $V_F$ be the vocabulary of all retrieved fragments and $V_C$ be the connector vocabulary. The output $O$ satisfies:

$$\text{words}(O) \subseteq V_F \cup V_C$$

Since $V_C$ contains only structural connectors (no facts), all factual content in $O$ comes from $V_F$, which comes directly from the database.

## Comparison of Approaches

| Approach | Mechanism | Hallucination Risk |
|---|---|---|
| Raw LLM (GPT-4, Gemma) | Probabilistic token sampling | **High** |
| RAG with LLM generator | Retrieval + probabilistic generation | **Medium** (model can drift) |
| RAG with temperature=0 | Greedy decoding, still generative | **Low-Medium** |
| DASA Statistical Mode | Deterministic sentence selection | **None** |
| DASA Grounded LLM Mode | Retrieval + anchored LLM prompt | **Very Low** |

## Limitations

The DASA guarantee applies only when:

1. The database is accurate. DASA propagates errors from the data source — it does not validate ground truth.
2. Statistical mode is used. Grounded LLM mode reduces hallucination risk but does not eliminate it.
3. `config.restricted_vocabulary = True`. Disabling this flag removes the vocabulary lock.

## The "Oregano Test"

The canonical sanity check for DASA: if your database contains an egg recipe with no mention of oregano, ask DASA for an egg recipe. If oregano appears in the response, the architecture contract has been violated. Passing this test is required for all contributions to Agent B.
