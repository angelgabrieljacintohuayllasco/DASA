"""
Statistical Rewriter — the mathematical heart of DASA's anti-hallucination system.

This module contains zero neural network calls.  It reconstructs coherent
text exclusively from the retrieved fragment pool, using:

1. Sentence extraction and scoring (keyword overlap).
2. Deterministic selection of top sentences.
3. Syntactic chaining with a fixed connector vocabulary.

The output vocabulary is a strict subset of the fragment vocabulary plus a
small set of neutral connectors.  This is what makes hallucination
mathematically impossible in statistical mode.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Set, TYPE_CHECKING

from dasa.config import DASAConfig

if TYPE_CHECKING:
    from dasa.agent_a.retrieval_agent import Fragment

# ── Constants ──────────────────────────────────────────────────────────────────

# Stopwords excluded from keyword extraction (common Spanish + English).
# These are neutral connective words that carry little semantic signal.
_STOPWORDS = {
    "de", "la", "el", "en", "a", "que", "y", "los", "las", "un", "una",
    "es", "se", "su", "con", "por", "para", "como", "del", "al", "lo",
    "the", "is", "of", "in", "a", "an", "to", "and", "or", "for",
}

# Connectors that Agent B is allowed to inject between sentences.
# These are the ONLY "invented" words allowed — and they carry no facts.
_CONNECTORS = [
    "Además,",
    "Asimismo,",
    "Por otro lado,",
    "En este sentido,",
    "Cabe destacar que",
    "De acuerdo con la información disponible,",
    "Finalmente,",
]


# ── Query-term utilities ──────────────────────────────────────────────────────

_QUERY_PREFIX_RE = re.compile(
    r"^(?:qu[eé]\s+(?:significa|es|son|quiere\s+decir)\s+(?:la\s+|el\s+|los\s+|las\s+|un\s+|una\s+)?"
    r"|defin[ei]\S*\s+(?:de\s+)?|significado\s+de\s+)",
    re.IGNORECASE,
)


def _normalize_str(text: str) -> str:
    """Lowercase and strip accents for accent-insensitive comparison."""
    text = text.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _extract_query_term(query: str) -> str:
    """Extract main concept word from a definition query.

    'que significa efimero'  → 'efimero'
    'que es la democracia'   → 'democracia'
    """
    cleaned = _QUERY_PREFIX_RE.sub("", query.strip())
    term = re.split(r"[,?.]", cleaned)[0].strip()
    return term if len(term) > 2 else ""


def _select_focused_fragments(
    fragments: "List",
    query_term: str,
) -> "List":
    """Return only the fragment whose source_id matches the query term.

    Prevents mixing definitions of related-but-wrong words (e.g., 'serpear'
    appearing in a response about 'serendipia').  Falls back to Fragment[0]
    (best cosine hit) when no exact match is found.
    """
    if not fragments:
        return []
    if not query_term:
        return fragments[:1]
    norm = _normalize_str(query_term)
    # Primary check: source_id (lemma key in SHARD mode)
    for frag in fragments:
        if frag.source_id and _normalize_str(frag.source_id) == norm:
            return [frag]
    # Secondary check: fragment text starts with the term (JSON mode)
    for frag in fragments:
        prefix = _normalize_str(frag.text[: len(query_term) + 3])
        if prefix.startswith(norm):
            return [frag]
    # No exact match — use only the top-ranked fragment to avoid mixing
    return fragments[:1]


class StatisticalRewriter:
    """
    Reconstructs a response by selecting and chaining the most relevant
    sentences from the retrieved fragments.

    No language model.  No probability distributions over vocabulary.
    No possibility of hallucination — only retrieved words appear in output.
    """

    def __init__(self, config: DASAConfig) -> None:
        self.config = config

    def rewrite(self, query: str, fragments: List[Fragment]) -> str:
        """
        Full rewrite pipeline:

        1. Extract individual sentences from all fragments.
        2. Score each sentence by keyword overlap with the query.
        3. Select the top N sentences (bounded by config.max_output_sentences).
        4. Chain them with neutral connectors.

        Args:
            query:     Original user query — drives sentence scoring.
            fragments: Retrieved fragments from Agent A.

        Returns:
            A coherent, fact-grounded response string.
        """
        query_keywords = self._extract_keywords(query)

        # Focus only on the fragment that matches the query term exactly.
        # This prevents sentences from related-but-wrong words from leaking in.
        query_term = _extract_query_term(query)
        focused = _select_focused_fragments(fragments, query_term)
        sentences = self._extract_sentences(focused)

        if not sentences:
            return ""

        scored = self._score_sentences(sentences, query_keywords)
        top = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        # Cap at 2 sentences for clean, focused definitions
        max_sents = min(2, self.config.max_output_sentences)
        top_sentences = [s for s, _ in top][:max_sents]

        return self._chain_sentences(top_sentences).strip()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful words from the query, removing stopwords."""
        words = re.findall(r"\b\w+\b", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 2}

    def _extract_sentences(self, fragments: List[Fragment]) -> List[str]:
        """Split all fragment texts into individual sentences."""
        sentences: List[str] = []
        for fragment in fragments:
            parts = re.split(r"[.!?;]+", fragment.text)
            sentences.extend(s.strip() for s in parts if len(s.strip()) > 8)
        return sentences

    def _score_sentences(
        self,
        sentences: List[str],
        keywords: Set[str],
    ) -> Dict[str, float]:
        """
        Score each sentence by the proportion of query keywords it contains.

        Score = |sentence_words ∩ query_keywords| / |sentence_words|

        Higher score = more relevant to the query.
        """
        scores: Dict[str, float] = {}
        for sentence in sentences:
            words = set(re.findall(r"\b\w+\b", sentence.lower()))
            if not words:
                scores[sentence] = 0.0
            else:
                scores[sentence] = len(words & keywords) / len(words)
        return scores

    def _chain_sentences(self, sentences: List[str]) -> str:
        """Join sentences with neutral connectors to form a readable paragraph."""
        if not sentences:
            return ""
        if len(sentences) == 1:
            s = sentences[0]
            return s if s.endswith(".") else s + "."

        result = sentences[0]
        if not result.endswith("."):
            result += "."

        for i, sentence in enumerate(sentences[1:]):
            connector = _CONNECTORS[i % len(_CONNECTORS)]
            s = sentence[0].lower() + sentence[1:] if sentence else sentence
            result += f" {connector} {s}"
            if not result.endswith("."):
                result += "."

        return result
