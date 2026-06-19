"""
Integration: DASA RetrievalAgent Tier-0 path over a real SHARD + IVF-PQ index.

Uses a deterministic fake embedding function (so no MiniLM download is needed)
to build both the index and the query vectors in the same space. Verifies the
full wiring: query -> encode -> IVFPQReader.search -> MMapReader.find -> Fragment.

Run:  python -m tests.test_shard_ivf_integration     (from DASA-main/)
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make both packages importable when run from DASA-main/
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))                 # DASA-main
sys.path.insert(0, str(_HERE.parent.parent.parent / "SHARD-main"))  # sibling SHARD-main

from dasa.config import DASAConfig                            # noqa: E402
from dasa.agent_a.retrieval_agent import RetrievalAgent       # noqa: E402
from shard.storage.shard_writer import ShardWriter           # noqa: E402
from shard.index.ivfpq_builder import build_ivfpq            # noqa: E402

DIM = 384
NUM_SHARDS = 64


def fake_vec(text, dim=DIM):
    """Deterministic unit vector from text — distinct text -> distinct vector."""
    seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _build(td, n=3000):
    records = [{"id": i, "lemma": f"term{i}", "definition": f"definition number {i} about topic {i}"}
               for i in range(n)]
    texts = [f"{r['lemma']}: {r['definition']}" for r in records]   # matches _record_to_text
    keys = [r["lemma"] for r in records]
    embeddings = np.stack([fake_vec(t) for t in texts])

    db = Path(td) / "db"
    with ShardWriter(str(db), num_shards=NUM_SHARDS) as w:
        for r in records:
            w.write(r["lemma"], json.dumps(r, ensure_ascii=False))
    build_ivfpq(embeddings, keys, str(db / "ivf"), profile="low-ram", seed=1)
    return db, records, texts


def test_ivf_tier0_retrieval():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db, records, texts = _build(td)
        cfg = DASAConfig(use_shard_backend=True, shard_db_path=str(db),
                         shard_num_shards=NUM_SHARDS, top_k_fragments=5,
                         similarity_threshold=0.2)
        agent = RetrievalAgent(cfg)
        agent.embedding_engine.encode = fake_vec        # no real model needed
        agent.load_dataset("")                          # loads the IVF index

        assert agent._ivf is not None, "IVF index should be the active backend"

        rng = np.random.default_rng(7)
        hits = 0
        trials = 50
        for i in rng.integers(0, len(records), trials):
            frags = agent.search(texts[i])              # query == record text -> same vector
            assert frags, f"no fragments for record {i}"
            top_ids = [f.source_id for f in frags]
            if records[i]["lemma"] in top_ids:
                hits += 1
            # returned fragment text must round-trip through the SHARD record
            assert "definition number" in frags[0].text
        rate = hits / trials
        print(f"[integration] self-query top-{cfg.top_k_fragments} hit rate = {rate:.2f} ({hits}/{trials})")
        assert rate >= 0.90, f"Tier-0 retrieval hit rate too low: {rate:.2f}"
        try:
            agent._ivf.close()
        except Exception:
            pass


if __name__ == "__main__":
    test_ivf_tier0_retrieval()
    print("OK")
