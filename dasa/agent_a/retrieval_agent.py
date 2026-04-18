import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

from dasa.agent_a.embeddings import EmbeddingEngine
from dasa.agent_a.tools import rank_fragments, filter_by_threshold
from dasa.config import DASAConfig


# ── Query-term utilities ──────────────────────────────────────────────────────

_QUERY_PREFIX_RE = re.compile(
    r"^(?:qu[eé]\s+(?:significa|es|son|quiere\s+decir)\s+(?:la\s+|el\s+|los\s+|las\s+|un\s+|una\s+)?"
    r"|defin[ei]\S*\s+(?:de\s+)?|significado\s+de\s+)",
    re.IGNORECASE,
)


def _normalize_str(text: str) -> str:
    """Lowercase and strip accents/diacritics for accent-insensitive comparison."""
    text = text.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _extract_query_term(query: str) -> str:
    """Extract main concept from a definition query.

    'que significa efimero'  → 'efimero'
    'que es la democracia'   → 'democracia'
    'define serendipia'      → 'serendipia'
    """
    cleaned = _QUERY_PREFIX_RE.sub("", query.strip())
    term = re.split(r"[,?.]", cleaned)[0].strip()
    return term if len(term) > 2 else ""


def _lev1_match(s1: str, s2: str) -> bool:
    """Return True if Levenshtein distance between s1 and s2 is at most 1.

    Used to catch minor spelling variants (e.g. 'efimero' vs 'efemero').
    Implemented in O(n) — safe to call in a tight loop over 66k keys.
    """
    if s1 == s2:
        return True
    n, m = len(s1), len(s2)
    if abs(n - m) > 1:
        return False
    if n < m:
        s1, s2, n, m = s2, s1, m, n   # ensure n >= m
    if n == m:
        return sum(a != b for a, b in zip(s1, s2)) == 1
    # n == m + 1: check if s2 == s1 with one char deleted
    i = 0
    while i < m and s1[i] == s2[i]:
        i += 1
    return s1[i + 1:] == s2[i:]


class Fragment:
    """
    A verified piece of information retrieved from the ground-truth database.

    Fragments are the atomic unit of truth in DASA.  Agent B can only use
    what is present in these objects — never anything outside them.
    """

    def __init__(self, text: str, score: float, source_id: Optional[str] = None) -> None:
        self.text = text
        self.score = score
        self.source_id = source_id

    def __repr__(self) -> str:
        return f"Fragment(score={self.score:.3f}, text={self.text[:60]!r})"


class RetrievalAgent:
    """
    Agent A — 100% agentic retrieval layer.

    Responsibilities:
    - Load and index a ground-truth dataset.
    - Accept a natural-language query.
    - Return a ranked, filtered list of Fragment objects.

    This agent NEVER generates content.  It only retrieves.
    All computation runs on CPU; no GPU is required.
    """

    def __init__(self, config: DASAConfig) -> None:
        self.config = config
        self.embedding_engine = EmbeddingEngine(config)
        self._records: List[Dict[str, Any]] = []
        self._embeddings = None  # np.ndarray, shape (N, dim), set after load

        # SHARD backend objects (only used when config.use_shard_backend=True)
        self._shard_reader = None    # MMapReader — exact key lookup
        self._shard_index = None     # IndexReader — MinHash similarity (fallback)
        self._embed_cache = None     # np.ndarray (N, dim) — pre-computed embeddings
        self._embed_keys: List[str] = []       # key list parallel to _embed_cache rows
        self._embed_keys_norm: Dict[str, int] = {}  # normalized_key → row index

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_dataset(self, path: str) -> None:
        """
        Load dataset.  Two modes:

        **JSON mode** (default, use_shard_backend=False):
          Loads the entire file into RAM and builds an in-memory embedding
          matrix.  Suitable for datasets up to ~500k records on 8 GB RAM.

        **SHARD mode** (use_shard_backend=True):
          Opens the SHARD binary database and its MinHash similarity index.
          No records are loaded into RAM: the OS pages in only the 4 KB blocks
          of the shard files that contain the requested data.
          Suitable for TB-scale datasets on machines with 2 GB RAM.

        Args:
            path: Path to a JSON file (JSON mode) or ignored (SHARD mode).
        """
        if self.config.use_shard_backend:
            self._load_shard_backend()
        else:
            self._load_json(path)

    def search(self, query: str) -> List[Fragment]:
        """
        Search for the top-k fragments most relevant to the query.

        In JSON mode:  cosine similarity on in-memory embedding matrix O(N·dim).
        In SHARD mode: MinHash similarity on the index → shard file lookup.
                       Only the matching shard file is read (mmap, zero full load).

        Args:
            query: Natural-language question or request.

        Returns:
            A ranked list of Fragment objects, capped at config.top_k_fragments.
        """
        if self.config.use_shard_backend:
            return self._search_shard(query)
        return self._search_json(query)

    # ── JSON backend ───────────────────────────────────────────────────────────

    def _load_json(self, path: str) -> None:
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(data_path, "r", encoding="utf-8") as f:
            self._records = json.load(f)

        if not isinstance(self._records, list):
            raise ValueError("Dataset must be a JSON array of objects.")

        import numpy as np
        texts = [self._record_to_text(r) for r in self._records]
        self._embeddings = self.embedding_engine.encode_batch(texts)

    def _search_json(self, query: str) -> List[Fragment]:
        if self._embeddings is None:
            raise RuntimeError(
                "Dataset not loaded. Call load_dataset() before search()."
            )

        query_embedding = self.embedding_engine.encode(query)
        scores = self.embedding_engine.cosine_similarity_batch(
            query_embedding, self._embeddings
        )

        fragments = [
            Fragment(
                text=self._record_to_text(self._records[i]),
                score=float(scores[i]),
                source_id=str(self._records[i].get("id", i)),
            )
            for i in range(len(self._records))
        ]

        fragments = filter_by_threshold(fragments, self.config.similarity_threshold)
        fragments = rank_fragments(fragments)
        return fragments[: self.config.top_k_fragments]

    # ── SHARD backend ──────────────────────────────────────────────────────────

    def _load_shard_backend(self) -> None:
        """
        Open the SHARD database and load auxiliary search structures.

        Priority:
          1. embeddings.npy + embedding_keys.json  → fast cosine similarity (best)
          2. index.minhash.bin                     → MinHash Jaccard (fallback)
          3. Neither                               → exact key lookup only

        The embedding cache (~96 MB for 66k records) is the recommended path.
        Build it once with: python tools/build_embedding_cache.py
        """
        db_path = self.config.shard_db_path
        if not db_path:
            raise ValueError(
                "config.shard_db_path must be set when use_shard_backend=True"
            )
        db_path = str(Path(db_path).resolve())
        _ensure_shard_importable()

        try:
            from shard.storage.mmap_reader import MMapReader
            from shard.index.index_reader import IndexReader
        except ImportError as exc:
            raise ImportError(
                "SHARD package not found. Install it with:\n"
                "  pip install shard-db\n"
                "or, if using the source repo:\n"
                "  pip install -e /path/to/SHARD-main"
            ) from exc

        self._shard_reader = MMapReader(db_path, num_shards=self.config.shard_num_shards)

        # ── Option 1: embedding cache (cosine similarity, most accurate) ───────
        embed_file = Path(db_path) / "embeddings.npy"
        keys_file  = Path(db_path) / "embedding_keys.json"
        if embed_file.exists() and keys_file.exists():
            import numpy as np
            self._embed_cache = np.load(str(embed_file))  # float32 (N, dim)
            with open(keys_file, encoding="utf-8") as f:
                self._embed_keys = json.load(f)
            self._embed_keys_norm = {
                _normalize_str(k): i for i, k in enumerate(self._embed_keys)
            }
            print(f"[Agent A] Embedding cache cargado: {len(self._embed_keys)} registros, "
                  f"{self._embed_cache.shape[1]} dims.")
            return

        # ── Option 2: MinHash index (Jaccard similarity, text overlap only) ────
        index_path = Path(db_path) / "index.meta.json"
        if index_path.exists():
            self._shard_index = IndexReader(db_path)
            self._shard_index.load()
            print("[Agent A] Indice MinHash cargado (modo fallback).")
            return

        print("[Agent A] Sin indice — solo busqueda por clave exacta.")

    def _search_shard(self, query: str) -> List[Fragment]:
        """
        Three-tier SHARD search (best available method wins):

        Tier 1 — Embedding cache (embeddings.npy):
          embed(query) → cosine similarity against all N pre-computed embeddings
          → top-k indices → fetch full record from SHARD by key
          RAM: embedding matrix (96 MB / 66k records) + one query embed (1.5 KB)
          Accuracy: identical to JSON mode

        Tier 2 — MinHash index (fallback if no .npy):
          query → shingles → Jaccard similarity → candidates → re-rank with cosine
          Less accurate for semantic queries, but works without embedding matrix.

        Tier 3 — Exact key lookup (no index at all):
          Only useful when the query IS the exact lemma.
        """
        if self._shard_reader is None:
            raise RuntimeError("SHARD backend not loaded.")

        import numpy as np

        # ── Tier 1: embedding cache ──────────────────────────────────────────
        if self._embed_cache is not None:
            query_emb = self.embedding_engine.encode(query)           # (dim,)
            # embeddings.npy rows are already L2-normalised → dot = cosine
            scores = self._embed_cache @ query_emb                    # (N,)
            # Exact-match boost: if the query names a specific term, guarantee
            # that term's record wins top-1 regardless of cosine drift.
            _qt = _extract_query_term(query)
            if _qt:
                _norm_qt = _normalize_str(_qt)
                _exact_idx = self._embed_keys_norm.get(_norm_qt)
                if _exact_idx is not None:
                    scores = scores.copy()  # don't mutate the cached matrix
                    scores[_exact_idx] = min(1.0, float(scores[_exact_idx]) + 0.40)
                elif len(_norm_qt) >= 5:
                    # Fuzzy fallback: catch spelling variants ≤1 edit distance
                    # (e.g. 'efimero' → 'efemero' in dataset)
                    scores = scores.copy()
                    for _nk, _ni in self._embed_keys_norm.items():
                        if _lev1_match(_norm_qt, _nk):
                            scores[_ni] = min(1.0, float(scores[_ni]) + 0.28)
                            break
            top_indices = np.argsort(scores)[::-1][: self.config.top_k_fragments * 3]

            fragments = []
            for idx in top_indices:
                key = self._embed_keys[idx]
                score = float(scores[idx])
                if score < self.config.similarity_threshold:
                    continue
                raw = self._shard_reader.find(key)
                if raw is None:
                    continue
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    record = {"text": str(raw)}
                fragments.append(Fragment(
                    text=self._record_to_text(record),
                    score=score,
                    source_id=key,
                ))
            fragments = rank_fragments(fragments)
            return fragments[: self.config.top_k_fragments]

        # ── Tier 2: MinHash index ────────────────────────────────────────────
        if self._shard_index is not None:
            candidates = self._shard_index.lookup(query, top_k=self.config.top_k_fragments * 4)
            fragments = []
            query_emb = self.embedding_engine.encode(query)
            for record_key, _ in candidates:
                raw = self._shard_reader.find(record_key)
                if raw is None:
                    continue
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    record = {"text": str(raw)}
                text = self._record_to_text(record)
                record_emb = self.embedding_engine.encode(text)
                cos_score = float(np.dot(query_emb, record_emb) /
                    (np.linalg.norm(query_emb) * np.linalg.norm(record_emb) + 1e-9))
                fragments.append(Fragment(text=text, score=cos_score, source_id=record_key))
            fragments = filter_by_threshold(fragments, self.config.similarity_threshold)
            fragments = rank_fragments(fragments)
            return fragments[: self.config.top_k_fragments]

        # ── Tier 3: exact key lookup ─────────────────────────────────────────
        raw = self._shard_reader.find(query)
        if raw is not None:
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                record = {"text": str(raw)}
            return [Fragment(text=self._record_to_text(record), score=1.0, source_id=query)]
        return []

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """Convert a record dict to a plain-text string for embedding."""
        parts = []
        for field in ("lemma", "term", "title", "name", "content", "text", "definition"):
            if field in record and record[field]:
                parts.append(str(record[field]))
        return ": ".join(parts) if parts else str(record)


def _ensure_shard_importable() -> None:
    """Add the SHARD repo root to sys.path if the package is not installed."""
    try:
        import shard  # noqa: F401
        return
    except ImportError:
        pass
    # Try common locations relative to this file
    candidates = [
        Path(__file__).parent.parent.parent.parent / "SHARD-main",  # sibling repo (packaged)
        Path(__file__).parent.parent.parent.parent / "SHARD",       # sibling repo (dev)
    ]
    for candidate in candidates:
        if (candidate / "shard" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
