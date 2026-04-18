# Agent A — Retrieval Agent Specification

## Purpose

Agent A is the **search layer** of DASA. Its only job is to find the most relevant fragments of verified information in the database and return them as `Fragment` objects.

It is 100% agentic: it can use multiple tools (embeddings, filters, deduplication) to assemble its result set. It never generates content.

## Interface

```python
class RetrievalAgent:
    def load_dataset(self, path: str) -> None: ...
    def search(self, query: str) -> List[Fragment]: ...
```

## Fragment Object

The atomic unit of truth returned by Agent A:

```python
class Fragment:
    text: str        # The verified text from the database
    score: float     # Cosine similarity to the query [0, 1]
    source_id: str   # Record ID for traceability
```

## Search Pipeline

```
query: str
    │
    ▼ EmbeddingEngine.encode(query)
query_vector: ndarray (dim=384)
    │
    ▼ EmbeddingEngine.cosine_similarity_batch(query_vector, corpus_vectors)
scores: ndarray (N,)
    │
    ▼ filter_by_threshold(fragments, config.similarity_threshold)
    │
    ▼ rank_fragments(fragments)
    │
    ▼ fragments[:config.top_k_fragments]
List[Fragment]
```

## Embedding Strategy

- **Model:** `all-MiniLM-L6-v2` by default — 80 MB, 384-dimensional output.
- **Device:** CPU (no GPU dependency).
- **Normalization:** All vectors are L2-normalized before storage and at query time. This converts the cosine similarity computation into a simple dot product (`np.dot`), which is extremely fast with NumPy BLAS.
- **Batch encoding:** The full dataset is encoded once at load time. Query encoding is a single forward pass.

## Agentic Tools

Agent A has access to the following tools (defined in `tools.py`):

| Tool | Function |
|---|---|
| `filter_by_threshold` | Remove fragments below similarity threshold |
| `rank_fragments` | Sort by score descending |
| `deduplicate_fragments` | Remove near-duplicate fragments by Jaccard overlap |

## Extensibility

To add a new retrieval tool, add a function to `agent_a/tools.py` that takes and returns a `List[Fragment]`, then call it inside `RetrievalAgent.search()`.

## SHARD Backend (TB-scale)

For datasets larger than available RAM, set `config.use_shard_backend = True` and point `config.shard_db_path` to a SHARD database directory. Agent A will query SHARD's `MMapReader` and `IndexReader` instead of loading a JSON file into memory.
