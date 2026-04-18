# Agent B — Synthesis Engine Specification

## Purpose

Agent B is the **synthesis layer** of DASA. It receives verified `Fragment` objects from Agent A and produces coherent natural language. It is **not a reasoner** — it is a deterministic formatter.

The key invariant: **Agent B's output vocabulary is bounded by the fragment vocabulary.**

## Interface

```python
class SynthesisEngine:
    def synthesize(self, query: str, fragments: List[Fragment]) -> str: ...
```

## Operating Modes

### 1. Statistical Mode (Default)

No neural network. Uses the `StatisticalRewriter`.

```
fragments: List[Fragment]
    │
    ▼ _extract_sentences(fragments)
sentences: List[str]
    │
    ▼ _score_sentences(sentences, query_keywords)
scored: Dict[str, float]    # score = keyword overlap ratio
    │
    ▼ sorted by score, take top N (config.max_output_sentences)
    │
    ▼ _chain_sentences(top_sentences)
response: str
```

**Why this mode is hallucination-proof:**
The output is constructed by:
1. Extracting sentences that already exist in the fragments.
2. Connecting them with a fixed connector vocabulary (defined as a constant in code).
3. No new words are generated — only rearranged and connected.

### 2. Grounded LLM Mode (Optional)

When `config.synthesis_model` is set, an LLM is invoked with a strict grounding prompt:

```
You are a text formatter. Your ONLY job is to rephrase the CONTEXT below.
CRITICAL RULES:
- Do NOT add any information not present in the CONTEXT.
- Do NOT reason, infer, or extrapolate.
- If the answer is not in the CONTEXT, say: 'The available information does not cover this topic.'

CONTEXT:
[fragments here]

QUERY: {query}

FORMATTED ANSWER:
```

This prompt forces the LLM into "Anchored Generation" mode — it acts as a syntax stylizer, not a knowledge source.

## Sentence Scoring Formula

For each candidate sentence $s$ extracted from the fragments:

$$\text{score}(s) = \frac{|\text{words}(s) \cap \text{keywords}(q)|}{|\text{words}(s)|}$$

Where $q$ is the original query and `keywords` excludes stopwords.

## Allowed Connectors

The only "invented" words Agent B can introduce are neutral discourse connectors:

- `Además,`
- `Asimismo,`
- `Por otro lado,`
- `En este sentido,`
- `Cabe destacar que`
- `De acuerdo con la información disponible,`
- `Finalmente,`

These connectors carry no factual claims and cannot introduce hallucinations.
