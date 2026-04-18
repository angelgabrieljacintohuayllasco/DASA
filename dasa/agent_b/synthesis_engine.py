from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from dasa.config import DASAConfig
from dasa.agent_b.statistical_rewriter import StatisticalRewriter

if TYPE_CHECKING:
    from dasa.agent_a.retrieval_agent import Fragment


class SynthesisEngine:
    """
    Agent B — Deterministic synthesis layer.

    This agent receives verified Fragments from Agent A and produces
    coherent natural language.  It has two operating modes:

    1. **Statistical mode** (default, ``config.synthesis_model is None``):
       The StatisticalRewriter rearranges and connects fragment sentences
       using only vocabulary present in the fragments.  Zero LLM involved.
       This is the purest implementation of DASA's anti-hallucination guarantee.

    2. **Grounded LLM mode** (``config.synthesis_model`` is set):
       A language model is used, but it is constrained by a strict grounding
       prompt that explicitly forbids generating information outside the
       provided context.  The LLM acts as a *formatter*, not a *reasoner*.

    In both modes, ``config.restricted_vocabulary = True`` enforces the DASA
    contract: Agent B cannot introduce facts that Agent A did not retrieve.
    """

    def __init__(self, config: DASAConfig) -> None:
        self.config = config
        self.rewriter = StatisticalRewriter(config)
        self._llm_callable: Optional[object] = None
        # System prompt para modo libre (sin corpus). Se puede sobreescribir
        # externamente para respetar el system message del cliente (e.g., Jan).
        self._free_system_prompt: str = (
            "Eres DASA, un asistente inteligente. "
            "Responde de forma natural, útil y concisa."
        )

    @property
    def engine_loaded(self) -> bool:
        """True if an LLM callable has already been wired up."""
        return self._llm_callable is not None

    def synthesize(self, query: str, fragments: List[Fragment]) -> str:
        """
        Produce a grounded response from verified fragments.

        Args:
            query:     The original user query (used for scoring relevance).
            fragments: Verified fragments from Agent A. The output vocabulary
                       is mathematically bounded to these objects.

        Returns:
            A coherent, factual response string with no hallucinations.
        """
        # Umbral de relevancia real: si ningún fragmento supera este score,
        # el corpus no cubre el tema → el LLM responde libremente.
        _RELEVANCE_THRESHOLD = 0.40

        if not fragments or (
            self._llm_callable is not None
            and max(f.score for f in fragments) < _RELEVANCE_THRESHOLD
        ):
            # Sin fragmentos relevantes: si hay LLM disponible (Ollama/HuggingFace),
            # dejar que responda libremente — saludos, preguntas generales, etc.
            if self._llm_callable is not None:
                return self._llm_free(query)
            return ""

        # Activar ruta LLM si hay modelo configurado O si se inyectó un callable externo
        # (ej. OllamaConnector inyectado directamente en _llm_callable)
        if self.config.synthesis_model is not None or self._llm_callable is not None:
            return self._llm_guided_synthesis(query, fragments)

        return self.rewriter.rewrite(query, fragments)

    # ── LLM-guided mode (optional) ─────────────────────────────────────────────

    def _llm_free(self, query: str) -> str:
        """
        Llamada al LLM sin restricciones de corpus.
        Se usa cuando Agent A no encontró fragmentos suficientemente relevantes
        (saludos, preguntas generales, conversación).
        Usa _free_system_prompt que puede ser sobreescrito por el cliente (e.g., Jan).
        """
        messages = [
            {"role": "system", "content": self._free_system_prompt},
            {"role": "user", "content": query},
        ]
        return self._call_llm(messages)

    def _llm_guided_synthesis(self, query: str, fragments: List[Fragment]) -> str:
        """
        Constrain an LLM with a grounding prompt that prevents it from
        generating content beyond the fragment boundaries.

        Uses OpenAI-style chat messages so Instruct models apply their chat
        template (system/user roles) — critical for small models like Qwen2.5.
        """
        fragment_texts = "\n".join(f"[{i+1}] {f.text}" for i, f in enumerate(fragments))
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un reformateador de texto. Tu única tarea es combinar "
                    "la información del CONTEXTO en una respuesta fluida y natural. "
                    "REGLAS ABSOLUTAS:\n"
                    "1. SOLO usa información del CONTEXTO. Cero inventos.\n"
                    "2. NO añadas datos, ejemplos, ni explicaciones externas.\n"
                    "3. Si el CONTEXTO no responde la pregunta, di exactamente: "
                    "'La información disponible no cubre este tema.'\n"
                    "4. Respuesta breve: máximo 3 oraciones."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXTO:\n{fragment_texts}\n\nPREGUNTA: {query}",
            },
        ]
        return self._call_llm(messages)

    def _call_llm(self, prompt: str) -> str:
        if self._llm_callable is None:
            raise NotImplementedError(
                "LLM-guided synthesis requires a callable set via "
                "engine._llm_callable = my_llm_function(prompt) -> str"
            )
        return str(self._llm_callable(prompt))
