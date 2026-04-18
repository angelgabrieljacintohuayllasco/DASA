# Contributing to DASA

Thank you for your interest in DASA! This document explains how to contribute.

## Ways to Contribute

- **Bug reports** — Open an issue with a minimal reproducible example
- **New agentic tools** — Extend `dasa/agent_a/tools.py` with new retrieval capabilities
- **Synthesis strategies** — Add alternative rewriting algorithms to `agent_b/`
- **Dataset adapters** — Support new input formats (JSONL, CSV, Parquet, SHARD)
- **Benchmarks** — Compare DASA against standard RAG pipelines
- **Documentation** — Fix typos, add examples, translate docs

## Development Setup

```bash
git clone https://github.com/YOUR_ORG/dasa.git
cd dasa
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Type hints on all public functions
- Docstrings on all public classes and methods
- No external dependencies beyond `requirements.txt` for the core package

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/my-feature`
2. Write tests for your change
3. Ensure `pytest tests/` passes
4. Open a pull request with a clear description of the change

## Architecture Contract

DASA has one hard rule that all contributions must respect:

> **Agent B must never produce output that contains information not present in the fragments delivered by Agent A.**

Any synthesis strategy that could introduce new facts (hallucinations) will be rejected.
