# DASA — Deterministic Agent Synthesis Architecture

> **La IA que no alucina.** / *The AI that doesn't hallucinate.*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

DASA is a **dual-agent retrieval and synthesis architecture** that eliminates LLM hallucinations by design. Instead of asking a model to *generate* an answer, DASA retrieves verified fragments from a ground-truth database and mathematically reconstructs them into coherent language — using only words present in the retrieved context.

Runs on **2 GB RAM, 2 vCPU** with no GPU required.

---

## The Problem

Modern LLMs are probabilistic text predictors. They don't "know" facts — they generate statistically likely sequences of words, which sometimes look like facts but are invented (hallucinated). They also require industrial-scale VRAM/RAM to run.

## The Solution: Two Specialized Agents

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  AGENT A — Retrieval (Agentic)   │
│  · Embeds the query (CPU only)   │
│  · Searches the TB-scale DB      │
│  · Returns verified Fragments    │  ← Only real data crosses this boundary
└────────────────┬─────────────────┘
                 │  List[Fragment]
                 ▼
┌──────────────────────────────────┐
│  AGENT B — Synthesis (Rewriter)  │
│  · Receives verified fragments   │
│  · Applies statistical rewriting │
│  · Vocabulary LOCKED to context  │  ← Mathematical anti-hallucination
└────────────────┬─────────────────┘
                 │
                 ▼
         Grounded Response
```

**The core guarantee:** Agent B can only use words that appear in the fragments Agent A retrieved. If the database doesn't say "season with plutonium", Agent B can never say it — because it's mathematically impossible within this architecture.

---

## DASA vs. Traditional RAG

| Property | Standard RAG | DASA |
|---|---|---|
| Hallucination risk | Medium (model can still "drift") | **None** (vocabulary is locked) |
| GPU required | Yes (for generation) | **No** (CPU-only inference) |
| RAM footprint | 8–80 GB | **< 2 GB** |
| Agent autonomy | Passive retrieval | **Active agentic retrieval** |
| Output type | Probabilistic generation | **Deterministic synthesis** |
| Backend storage | Vector DB (Qdrant, Weaviate…) | **Any JSON / SHARD binary DB** |

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from dasa.pipeline import DASAPipeline
from dasa.config import DASAConfig

config = DASAConfig(
    embedding_model="all-MiniLM-L6-v2",  # 80 MB, runs on CPU
    top_k_fragments=5,
    similarity_threshold=0.3,
)

pipeline = DASAPipeline(config)
pipeline.load("my_dataset.json")   # JSON array of {"lemma": ..., "definition": ...}

response = pipeline.run("¿Cómo preparo huevos fritos?")
print(response)
```

Run the built-in demo (no dataset needed):

```bash
python examples/recipe_example.py
```

---

## Project Structure

```
dasa/
├── agent_a/
│   ├── retrieval_agent.py     # Agent A: search + fragment retrieval
│   ├── embeddings.py          # CPU-only sentence embeddings
│   └── tools.py               # Agentic tools: rank, filter, deduplicate
├── agent_b/
│   ├── synthesis_engine.py    # Agent B: grounded synthesis orchestrator
│   └── statistical_rewriter.py # Pure-math text reconstruction (no LLM needed)
├── pipeline.py                # DASAPipeline: connects A → B
└── config.py                  # All configuration parameters

docs/
├── architecture.md            # Full system design
├── agent-a.md                 # Agent A specification
├── agent-b.md                 # Agent B specification
└── anti-hallucination.md      # Why DASA cannot hallucinate

examples/
├── basic_query.py             # Minimal usage example
└── recipe_example.py          # The "egg recipe" demo from the DASA paper
```

---

## Dataset Format

DASA works with any JSON array. Minimal format:

```json
[
  {"id": "001", "lemma": "Python", "definition": "High-level, general-purpose programming language."},
  {"id": "002", "lemma": "ábaco",  "definition": "Manual calculating instrument using rows of beads."}
]
```

For TB-scale datasets, use **[SHARD](https://github.com/YOUR_ORG/shard)** — the purpose-built binary hash-sharded database designed for DASA.

---

## Why "Deterministic"?

Given the same query and the same database, DASA **always produces the same class of answer**: one constructed exclusively from retrieved truth. Unlike temperature-controlled LLMs where outputs vary run-to-run and hallucinations are random, DASA's output space is bounded by the database. This is the `D` in DASA.

---

## Roadmap

- [ ] SHARD backend integration (native connector)
- [ ] Multi-language embedding support
- [ ] Streaming response mode for large fragment sets
- [ ] Agent A tool extensions: math reasoning, date parsing, aggregation
- [ ] Lightweight LLM-guided synthesis mode (grounding-constrained)
- [ ] Docker image for Raspberry Pi / ARM64

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome contributions in:
- New agentic tools for Agent A
- Alternative synthesis strategies for Agent B
- Adapters for different dataset formats
- Benchmarks against standard RAG pipelines

---

## ⭐ Star History

<a href="https://star-history.com/#angelgabrieljacintohuayllasco-lgtm/DASA-IA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco-lgtm/DASA-IA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco-lgtm/DASA-IA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=angelgabrieljacintohuayllasco-lgtm/DASA-IA&type=Date" />
 </picture>
</a>

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE)

---

*DASA was designed to run on the hardware you already have: a $5 chip should be enough to answer any question correctly, without inventing facts.*
