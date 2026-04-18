from typing import Optional

from dasa.config import DASAConfig
from dasa.agent_a.retrieval_agent import RetrievalAgent
from dasa.agent_b.synthesis_engine import SynthesisEngine
from dasa.agent_b.llm_connector import LLMConnector


class DASAPipeline:
    """
    Top-level orchestrator for the DASA system.

    Usage::

        pipeline = DASAPipeline()
        pipeline.load("my_dataset.json")
        response = pipeline.run("¿Cómo preparo huevos fritos?")

    The pipeline connects Agent A (retrieval) → Agent B (synthesis) and
    guarantees that the final response is grounded exclusively in the
    retrieved fragments — zero hallucinations by construction.
    """

    def __init__(self, config: Optional[DASAConfig] = None) -> None:
        self.config = config or DASAConfig()
        self.agent_a = RetrievalAgent(self.config)
        self.agent_b = SynthesisEngine(self.config)
        self._ready = False

    def load(self, dataset_path: Optional[str] = None) -> "DASAPipeline":
        """
        Load the dataset and prepare Agent A for querying.

        Args:
            dataset_path: Path to a JSON file (array of records). Falls back
                to config.demo_data_path when omitted.

        Returns:
            self, so calls can be chained: ``pipeline.load().run(...)``.
        """
        path = dataset_path or self.config.demo_data_path
        self.agent_a.load_dataset(path)

        # If a synthesis model is configured, load the LLM connector for Agent B
        if self.config.synthesis_model is not None and not self.agent_b.engine_loaded:
            connector = LLMConnector(
                model_name=self.config.synthesis_model,
                device=self.config.device,
            )
            connector.load()
            self.agent_b._llm_callable = connector

        self._ready = True
        return self

    def run(self, query: str) -> str:
        """
        Execute a full DASA query cycle:
            1. Agent A searches for relevant fragments.
            2. Agent B synthesizes a grounded response.

        Args:
            query: Natural-language question or request.

        Returns:
            A coherent, hallucination-free response string.

        Raises:
            RuntimeError: If the pipeline has not been loaded yet.
        """
        if not self._ready:
            raise RuntimeError(
                "Pipeline not loaded. Call .load() before .run()."
            )

        fragments = self.agent_a.search(query)

        if not fragments:
            return "No relevant information found in the database for this query."

        return self.agent_b.synthesize(query, fragments)

    def __repr__(self) -> str:
        status = "ready" if self._ready else "not loaded"
        return (
            f"DASAPipeline("
            f"model={self.config.embedding_model!r}, "
            f"top_k={self.config.top_k_fragments}, "
            f"status={status!r})"
        )
