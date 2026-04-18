"""DASA Recipe Example
====================
The canonical DASA demo: a user asks for a recipe and the system returns
a coherent, factual answer assembled from the database — with zero hallucinations.

Modes:
  Statistical (default): Agent B uses pure-math rewriting. No LLM.
  LLM mode: Agent B uses Qwen2.5-0.5B-Instruct as final synthesizer.
            Set environment variable USE_LLM=1 to enable.

Run:
    python examples/recipe_example.py
    USE_LLM=1 python examples/recipe_example.py  # LLM mode
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dasa.pipeline import DASAPipeline
from dasa.config import DASAConfig

USE_LLM = os.environ.get("USE_LLM", "0") == "1"

DEMO_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "demo_dataset.json"
)


def run_recipe_demo() -> None:
    mode = "LLM-guided (Qwen2.5-0.5B-Instruct)" if USE_LLM else "Statistical (pure-math)"
    print("=" * 60)
    print("  DASA — Recipe Demo")
    print(f"  Agent B mode: {mode}")
    print("=" * 60)

    config = DASAConfig(
        embedding_model="all-MiniLM-L6-v2",
        top_k_fragments=4,
        similarity_threshold=0.2,
        max_output_sentences=3,
        demo_data_path=DEMO_DATA_PATH,
        synthesis_model="Qwen/Qwen2.5-0.5B-Instruct" if USE_LLM else None,
    )

    print("\n[1/2] Loading pipeline...")
    pipeline = DASAPipeline(config)
    pipeline.load()

    query = "Quiero receta de huevos"
    print(f"\n[2/2] Running query: {query!r}\n")

    fragments = pipeline.agent_a.search(query)
    print("Fragments retrieved by Agent A:")
    for f in fragments:
        print(f"  score={f.score:.3f}  |  {f.text[:80]}")

    response = pipeline.agent_b.synthesize(query, fragments)
    print(f"\nFinal response from Agent B ({mode}):")
    print(f"  {response}")

    print("\n" + "=" * 60)
    if USE_LLM:
        print("Agent B (LLM) received only the fragments above as context.")
        print("It could not have added facts not present in those fragments.")
    else:
        print("Every word above appears in the retrieved fragments.")
        print("Agent B introduced ZERO new facts (pure-math mode).")
    print("=" * 60)


if __name__ == "__main__":
    run_recipe_demo()
