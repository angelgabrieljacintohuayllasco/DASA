import json
import os
import tempfile

import pytest

from dasa.config import DASAConfig
from dasa.agent_a.tools import (
    filter_by_threshold,
    rank_fragments,
    deduplicate_fragments,
    _jaccard_overlap,
)
from dasa.agent_b.statistical_rewriter import StatisticalRewriter


# ── Minimal Fragment stub (avoids loading real embedding models in tests) ──────

class MockFragment:
    def __init__(self, text: str, score: float, source_id: str = "mock"):
        self.text = text
        self.score = score
        self.source_id = source_id


# ── Agent A tools ──────────────────────────────────────────────────────────────

class TestFilterByThreshold:
    def test_removes_fragments_below_threshold(self):
        fragments = [
            MockFragment("high relevance", 0.9),
            MockFragment("medium relevance", 0.5),
            MockFragment("low relevance", 0.1),
        ]
        result = filter_by_threshold(fragments, threshold=0.4)
        assert len(result) == 2
        assert all(f.score >= 0.4 for f in result)

    def test_empty_input_returns_empty(self):
        assert filter_by_threshold([], 0.5) == []

    def test_all_above_threshold_returns_all(self):
        fragments = [MockFragment("x", 0.8), MockFragment("y", 0.6)]
        result = filter_by_threshold(fragments, 0.5)
        assert len(result) == 2

    def test_all_below_threshold_returns_empty(self):
        fragments = [MockFragment("x", 0.1), MockFragment("y", 0.2)]
        result = filter_by_threshold(fragments, 0.5)
        assert result == []


class TestRankFragments:
    def test_sorts_descending_by_score(self):
        fragments = [
            MockFragment("c", 0.3),
            MockFragment("a", 0.9),
            MockFragment("b", 0.6),
        ]
        ranked = rank_fragments(fragments)
        scores = [f.score for f in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_single_fragment_unchanged(self):
        f = MockFragment("only one", 0.7)
        result = rank_fragments([f])
        assert result[0].score == 0.7


class TestJaccardOverlap:
    def test_identical_strings_return_one(self):
        assert _jaccard_overlap("hello world", "hello world") == 1.0

    def test_disjoint_strings_return_zero(self):
        assert _jaccard_overlap("cat dog", "table chair") == 0.0

    def test_partial_overlap(self):
        score = _jaccard_overlap("cat dog bird", "cat fish")
        assert 0.0 < score < 1.0

    def test_empty_strings_return_zero(self):
        assert _jaccard_overlap("", "") == 0.0


class TestDeduplicateFragments:
    def test_removes_near_duplicates(self):
        fragments = [
            MockFragment("Freír el huevo en aceite caliente durante tres minutos", 0.9),
            MockFragment("Freír el huevo en aceite durante tres minutos exactos", 0.85),
            MockFragment("La capital de Francia es París", 0.7),
        ]
        result = deduplicate_fragments(fragments, overlap_threshold=0.6)
        assert len(result) == 2

    def test_unique_fragments_unchanged(self):
        fragments = [
            MockFragment("Python es un lenguaje de programación", 0.9),
            MockFragment("La Raspberry Pi es un microordenador", 0.7),
        ]
        result = deduplicate_fragments(fragments)
        assert len(result) == 2


# ── Agent B statistical rewriter ──────────────────────────────────────────────

class TestStatisticalRewriter:
    def setup_method(self):
        self.rewriter = StatisticalRewriter(DASAConfig())

    def test_output_not_empty_with_valid_fragment(self):
        fragments = [MockFragment("Freír el huevo en aceite durante tres minutos.", 0.9)]
        output = self.rewriter.rewrite("receta de huevos", fragments)
        assert len(output) > 0

    def test_empty_fragments_returns_empty_string(self):
        output = self.rewriter.rewrite("any query", [])
        assert output == ""

    def test_output_ends_with_period(self):
        fragments = [MockFragment("El huevo se fríe con aceite caliente", 0.9)]
        output = self.rewriter.rewrite("huevo frito", fragments)
        assert output.endswith(".")

    def test_keyword_extraction_removes_stopwords(self):
        keywords = self.rewriter._extract_keywords("receta de huevos fritos con aceite")
        assert "huevos" in keywords
        assert "receta" in keywords
        assert "de" not in keywords
        assert "con" not in keywords

    def test_output_uses_fragment_vocabulary(self):
        text = "Cocinar el huevo con agua y sal por diez minutos."
        fragments = [MockFragment(text, 0.9)]
        output = self.rewriter.rewrite("cómo cocinar huevos", fragments)
        fragment_words = set(text.lower().split())
        output_words = set(output.lower().split())
        overlap = fragment_words & output_words
        assert len(overlap) >= 2

    def test_multiple_fragments_combined(self):
        fragments = [
            MockFragment("Calentar aceite en la sartén.", 0.9),
            MockFragment("Añadir sal al gusto.", 0.7),
        ]
        output = self.rewriter.rewrite("receta huevos fritos", fragments)
        assert len(output) > 10


# ── DASAConfig ─────────────────────────────────────────────────────────────────

class TestDASAConfig:
    def test_default_device_is_cpu(self):
        assert DASAConfig().device == "cpu"

    def test_default_restricted_vocabulary_is_true(self):
        assert DASAConfig().restricted_vocabulary is True

    def test_default_synthesis_model_is_none(self):
        assert DASAConfig().synthesis_model is None

    def test_custom_values_are_set(self):
        config = DASAConfig(top_k_fragments=10, similarity_threshold=0.5)
        assert config.top_k_fragments == 10
        assert config.similarity_threshold == 0.5


# ── DASAPipeline integration (no embedding model loaded) ──────────────────────

class TestDASAPipelineContract:
    def test_run_raises_if_not_loaded(self):
        from dasa.pipeline import DASAPipeline
        pipeline = DASAPipeline()
        with pytest.raises(RuntimeError, match="not loaded"):
            pipeline.run("any query")

    def test_load_raises_on_missing_file(self):
        from dasa.pipeline import DASAPipeline
        pipeline = DASAPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.load("/nonexistent/path/dataset.json")

    def test_load_raises_on_invalid_json_structure(self):
        from dasa.pipeline import DASAPipeline
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"not": "an array"}, f)
            tmp_path = f.name
        try:
            pipeline = DASAPipeline()
            with pytest.raises((ValueError, Exception)):
                pipeline.load(tmp_path)
        finally:
            os.unlink(tmp_path)
