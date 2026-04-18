"""
Agentic tools for Agent A — retrieval utilities.

These functions are the "toolbox" available to the Retrieval Agent.
They operate on Fragment objects and apply mathematical logic
(sorting, filtering, deduplication) with no neural network involvement.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from dasa.agent_a.retrieval_agent import Fragment


def rank_fragments(fragments: List[Fragment]) -> List[Fragment]:
    """Sort fragments by relevance score, highest first."""
    return sorted(fragments, key=lambda f: f.score, reverse=True)


def filter_by_threshold(fragments: List[Fragment], threshold: float) -> List[Fragment]:
    """Discard fragments whose similarity score is below *threshold*."""
    return [f for f in fragments if f.score >= threshold]


def deduplicate_fragments(
    fragments: List[Fragment],
    overlap_threshold: float = 0.7,
) -> List[Fragment]:
    """
    Remove near-duplicate fragments using Jaccard overlap on word sets.

    When two fragments share more than *overlap_threshold* of their vocabulary,
    only the higher-scored one is kept.  This prevents Agent B from receiving
    redundant context that would bias its output.
    """
    seen_texts: List[str] = []
    result: List[Fragment] = []

    for fragment in fragments:
        is_duplicate = any(
            _jaccard_overlap(fragment.text, seen) >= overlap_threshold
            for seen in seen_texts
        )
        if not is_duplicate:
            result.append(fragment)
            seen_texts.append(fragment.text)

    return result


def _jaccard_overlap(a: str, b: str) -> float:
    """Jaccard similarity between word sets of two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)
