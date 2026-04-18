from dataclasses import dataclass
from typing import Optional


@dataclass
class DASAConfig:
    # ── Agent A — Retrieval ───────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    """Sentence-Transformers model name. 'all-MiniLM-L6-v2' is 80 MB and
    runs well on CPU with 2 GB RAM."""

    top_k_fragments: int = 5
    """Maximum number of fragments Agent A returns to Agent B."""

    similarity_threshold: float = 0.3
    """Minimum cosine similarity score for a fragment to be considered
    relevant. Fragments below this threshold are discarded."""

    device: str = "cpu"
    """Inference device for embeddings. 'cpu' is the default; set to 'cuda'
    if a GPU is available."""

    # ── Agent B — Synthesis ───────────────────────────────────────────────────
    synthesis_model: Optional[str] = None
    """Optional LLM for Agent B. When None, the StatisticalRewriter is used
    (no neural network, zero hallucination risk). When set to a model name,
    LLM-guided synthesis is used with grounding constraints."""

    restricted_vocabulary: bool = True
    """Enforce that Agent B only uses vocabulary present in the retrieved
    fragments. This is the core anti-hallucination guarantee of DASA.
    Disable only if you fully understand the implications."""

    max_output_sentences: int = 4
    """Maximum number of sentences in Agent B's output."""

    # ── Storage backend ───────────────────────────────────────────────────────
    use_shard_backend: bool = False
    """When True, Agent A uses a SHARD binary database instead of loading
    the full JSON dataset into RAM. Required for TB-scale deployments."""

    shard_db_path: Optional[str] = None
    """Path to the SHARD database directory. Required when use_shard_backend
    is True."""

    shard_num_shards: int = 1000
    """Number of shards in the SHARD database. Must match the value used
    when the database was built."""

    # ── Misc ──────────────────────────────────────────────────────────────────
    demo_data_path: str = "data/demo_dataset.json"
    """Default dataset path used by DASAPipeline.load() when no path is given."""
