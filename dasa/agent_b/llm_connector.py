"""
LLM Connector for Agent B — Grounded Synthesis with a local language model.

Loads a small causal LM (default: Qwen2.5-0.5B-Instruct) via HuggingFace
Transformers and wraps it in a callable that the SynthesisEngine can use.

Design principles:
- Model is loaded lazily on first call (no startup cost if not used).
- Inference runs on CPU — no GPU dependency.
- The prompt enforces the DASA "Anchored Generation" contract:
  the LLM is instructed to act as a formatter, not a reasoner.
- Max new tokens is capped to avoid runaway generation.
- Temperature is set low (0.2) for near-deterministic outputs.
"""

from __future__ import annotations

from typing import Optional


class LLMConnector:
    """
    Wraps a HuggingFace causal language model for use as Agent B.

    Usage::

        connector = LLMConnector("Qwen/Qwen2.5-0.5B-Instruct")
        connector.load()
        # Pass the callable to SynthesisEngine:
        engine._llm_callable = connector

    The connector is callable: ``connector(prompt: str) -> str``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._pipeline = None

    def load(self) -> "LLMConnector":
        """
        Load the model and tokenizer.  Called once; subsequent calls are no-ops.

        Returns self for chaining: ``connector.load()(prompt)``
        """
        if self._pipeline is not None:
            return self

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for LLM mode. "
                "Install with: pip install transformers"
            ) from exc

        print(f"[Agent B] Loading LLM: {self.model_name} on {self.device}...")
        self._pipeline = hf_pipeline(
            "text-generation",
            model=self.model_name,
            device=self.device,
            dtype="auto",
        )
        print("[Agent B] LLM ready.")
        return self

    def __call__(self, messages: list | str) -> str:
        """
        Generate a completion.

        Args:
            messages: Either a list of OpenAI-style chat dicts
                      [{'role': 'system', 'content': '...'},
                       {'role': 'user',   'content': '...'}]
                      or a plain prompt string (legacy).

        Returns:
            The generated text, stripped.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "LLMConnector not loaded. Call .load() first."
            )

        # Use chat template if available (Instruct models need it)
        if isinstance(messages, list):
            tokenizer = self._pipeline.tokenizer
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback: concatenate roles manually
                prompt = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                ) + "\nASSISTANT:"
        else:
            prompt = messages

        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded into memory."""
        return self._pipeline is not None


class OllamaConnector:
    """
    Connector for Ollama (local LLM server).

    Requires Ollama to be running: https://ollama.com

    Usage::

        connector = OllamaConnector(model="gemma3:4b")
        engine._llm_callable = connector

    Accepts both OpenAI-style chat message lists and plain strings:
      - list  → POST /api/chat   (preserves system/user roles)
      - str   → POST /api/generate
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout

    def __call__(self, prompt: "list | str") -> str:
        """
        Generate a completion via Ollama.

        Args:
            prompt: Either an OpenAI-style list of chat dicts
                    [{'role': 'system', 'content': '...'},
                     {'role': 'user',   'content': '...'}]
                    or a plain string.

        Returns:
            The generated text, stripped.
        """
        import urllib.request
        import json as _json

        try:
            if isinstance(prompt, list):
                # Chat endpoint — preserves system/user role separation
                payload = _json.dumps({
                    "model": self.model,
                    "messages": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 256},
                }).encode()
                url = f"{self.host}/api/chat"
                key = "message"  # response is {"message": {"role": "assistant", "content": "..."}}
            else:
                # Generate endpoint — plain string prompt
                payload = _json.dumps({
                    "model": self.model,
                    "prompt": str(prompt),
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 256},
                }).encode()
                url = f"{self.host}/api/generate"
                key = "response"

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = _json.loads(resp.read())
                if key == "message":
                    return result.get("message", {}).get("content", "").strip()
                return result.get(key, "").strip()

        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed ({self.host}). "
                f"Is Ollama running with model '{self.model}'? Error: {exc}"
            ) from exc
