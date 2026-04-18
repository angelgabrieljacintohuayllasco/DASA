"""DASA Basic Query Example
========================
Demonstrates a simple query against the built-in demo dataset.

- No GPU required.
- Runs on CPU with 2 GB RAM.
- Uses the demo_dataset.json included in the repo.

Modes:
  Statistical (default): Agent B uses pure-math rewriting. No LLM.
  LLM mode: Agent B uses Qwen2.5-0.5B-Instruct as the synthesis engine.
            Set environment variable USE_LLM=1 to enable.

Run:
    python examples/basic_query.py           # statistical mode
    USE_LLM=1 python examples/basic_query.py # LLM-guided mode (Windows: set USE_LLM=1)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dasa.pipeline import DASAPipeline
from dasa.config import DASAConfig

USE_LLM = os.environ.get("USE_LLM", "0") == "1"


def main() -> None:
    config = DASAConfig(
        embedding_model="all-MiniLM-L6-v2",
        top_k_fragments=3,
        similarity_threshold=0.2,
        demo_data_path=os.path.join(
            os.path.dirname(__file__), "..", "data", "demo_dataset.json"
        ),
        synthesis_model="Qwen/Qwen2.5-0.5B-Instruct" if USE_LLM else None,
    )

    mode = "LLM-guided (Qwen2.5-0.5B-Instruct)" if USE_LLM else "Statistical (pure-math, no LLM)"
    print(f"Loading DASA pipeline — Agent B mode: {mode}")
    pipeline = DASAPipeline(config)
    pipeline.load()
    print(f"Pipeline ready: {pipeline}\n")
    print("=" * 60)

    queries = [
        "¿Qué es la inteligencia artificial?",
        "¿Cómo preparo huevos?",
        "¿Qué es Python?",
        "¿Qué es un embedding?",
    ]

    for query in queries:
        print(f"\nQuery   : {query}")
        response = pipeline.run(query)
        print(f"Response: {response}")
        print("-" * 60)


if __name__ == "__main__":
    main()
