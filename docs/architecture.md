# DASA Architecture

## Overview

DASA (Deterministic Agent Synthesis Architecture) is a two-agent system designed to provide factual, hallucination-free answers from large datasets on minimal hardware.

## The Two-Agent Model

```
                        ┌─────────────────────────────────────┐
                        │          DASA Pipeline               │
                        │                                      │
   User Query ──────────▶   Agent A          Agent B          │
                        │   Retrieval   ──▶  Synthesis         │
                        │   (Agentic)        (Deterministic)   │
                        └────────────────────────┬────────────┘
                                                 │
                                         Grounded Response
```

### Agent A — Retrieval Agent

**Role:** Find truth. Return only what exists in the database.

- Accepts a natural-language query.
- Generates a query embedding on CPU using `sentence-transformers`.
- Computes cosine similarity against all pre-indexed document embeddings.
- Filters by `similarity_threshold` and returns the top `top_k_fragments` results.
- Each result is a `Fragment` object containing verified text and a relevance score.

Agent A has **no generative capability**. It cannot invent information.

### Agent B — Synthesis Engine

**Role:** Make the retrieved truth readable. Never add facts.

- Receives a list of `Fragment` objects from Agent A.
- Extracts sentences from the fragments.
- Scores sentences by keyword overlap with the original query.
- Chains the highest-scoring sentences using neutral connectors.
- Returns a coherent paragraph.

In statistical mode (default), Agent B uses **zero neural network calls**. The output vocabulary is a strict subset of the fragment vocabulary plus a fixed set of neutral connecting words.

## Data Flow

```
User Input: "¿Cómo preparo huevos fritos?"
     │
     ▼
EmbeddingEngine.encode(query)
     │
     ├─── query_vector (384-dim, normalized)
     │
     ▼
cosine_similarity_batch(query_vector, corpus_vectors)
     │
     ├─── scores[i] ∈ [-1, 1] for each record
     │
     ▼
filter_by_threshold + rank_fragments
     │
     ├─── fragments = [Fragment("huevo frito. Preparación...", score=0.78), ...]
     │
     ▼
StatisticalRewriter.rewrite(query, fragments)
     │
     ├─── Extract sentences
     ├─── Score by keyword overlap
     ├─── Select top N
     ├─── Chain with connectors
     │
     ▼
"Para preparar el huevo frito, calentar aceite en la sartén..."
```

## Anti-Hallucination Guarantee

The guarantee is **not heuristic** — it is structural:

1. Agent A only returns `Fragment` objects whose `.text` attribute was read directly from the database.
2. Agent B's `StatisticalRewriter` only uses words that appear in `Fragment.text` strings plus a fixed connector list.
3. There is no probabilistic sampling, no temperature parameter, no beam search.

Given a fixed database and a fixed query, DASA produces a deterministic output every time.

## Configuration

All parameters are exposed in `DASAConfig`:

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model (80 MB, CPU) |
| `top_k_fragments` | `5` | Max fragments passed to Agent B |
| `similarity_threshold` | `0.3` | Min cosine similarity to include a fragment |
| `device` | `cpu` | Inference device |
| `synthesis_model` | `None` | Optional LLM for Agent B (grounded mode) |
| `restricted_vocabulary` | `True` | Enforce vocabulary lock in synthesis |
| `max_output_sentences` | `4` | Max sentences in the response |

## Hardware Requirements

| Component | Minimum | Notes |
|---|---|---|
| RAM | 512 MB | For small datasets (< 10k records) |
| RAM | 2 GB | For medium datasets (100k records) |
| CPU | 1 vCPU | Single-threaded inference supported |
| GPU | Not required | All computation runs on CPU |
| Storage | Dataset size | Dataset stays on disk; only index in RAM |
