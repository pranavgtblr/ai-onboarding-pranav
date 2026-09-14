"""Unit and integration tests for Task 5.2: CI RAG Regression Suite."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from phase_5_production.regression_evaluator import (
    RAGRegressionSuite,
    RegressionThresholds,
    detect_rag_relevant_changes,
)


def test_detect_rag_relevant_changes_prompt_file() -> None:
    """Verify changes to prompt files trigger the evaluation suite."""
    fake_diff_names = "phase_3_rag/src/phase_3_rag/naive_rag.py\nREADME.md\n"
    fake_diff_content = "DEFAULT_RAG_SYSTEM_PROMPT = 'New prompt text'"

    with patch("subprocess.check_output") as mock_subp:
        mock_subp.side_effect = [fake_diff_names, fake_diff_content]
        should_run, categories, triggers = detect_rag_relevant_changes(
            base_ref="origin/main"
        )

        assert should_run is True
        assert "phase_3_rag/src/phase_3_rag/naive_rag.py" in triggers
        assert len(categories["prompts"]) > 0


def test_detect_rag_relevant_changes_chunking_file() -> None:
    """Verify changes to chunking parameters trigger the evaluation suite."""
    fake_diff_names = "phase_3_rag/src/phase_3_rag/chunking.py\n"
    fake_diff_content = "- chunk_size: int = 500\n+ chunk_size: int = 250\n"

    with patch("subprocess.check_output") as mock_subp:
        mock_subp.side_effect = [fake_diff_names, fake_diff_content]
        should_run, categories, triggers = detect_rag_relevant_changes(
            base_ref="origin/main"
        )

        assert should_run is True
        assert "phase_3_rag/src/phase_3_rag/chunking.py" in triggers
        assert len(categories["chunking"]) > 0


def test_detect_rag_relevant_changes_retrieval_file() -> None:
    """Verify changes to retrieval files trigger the evaluation suite."""
    fake_diff_names = "phase_3_rag/src/phase_3_rag/vector_store.py\n"
    fake_diff_content = "def similarity_search(): pass"

    with patch("subprocess.check_output") as mock_subp:
        mock_subp.side_effect = [fake_diff_names, fake_diff_content]
        should_run, categories, triggers = detect_rag_relevant_changes(
            base_ref="origin/main"
        )

        assert should_run is True
        assert "phase_3_rag/src/phase_3_rag/vector_store.py" in triggers
        assert len(categories["retrieval"]) > 0


def test_detect_rag_relevant_changes_content_keyword() -> None:
    """Verify diffs modifying chunk_size in unspecified python files trigger eval."""
    fake_diff_names = "some_pipeline/config.py\n"
    fake_diff_content = "chunk_size = 300"

    with patch("subprocess.check_output") as mock_subp:
        mock_subp.side_effect = [fake_diff_names, fake_diff_content]
        should_run, categories, triggers = detect_rag_relevant_changes(
            base_ref="origin/main"
        )

        assert should_run is True
        assert "some_pipeline/config.py" in triggers


def test_detect_rag_relevant_changes_unrelated_files() -> None:
    """Verify purely unrelated documentation or asset changes do not trigger eval."""
    fake_diff_names = "docs/overview.md\nassets/logo.png\n.gitignore\n"
    fake_diff_content = "Documentation updates"

    with patch("subprocess.check_output") as mock_subp:
        mock_subp.side_effect = [fake_diff_names, fake_diff_content]
        should_run, categories, triggers = detect_rag_relevant_changes(
            base_ref="origin/main"
        )

        assert should_run is False
        assert len(triggers) == 0
        assert len(categories["prompts"]) == 0
        assert len(categories["chunking"]) == 0
        assert len(categories["retrieval"]) == 0


def test_regression_evaluator_passes_quality_gates() -> None:
    """Verify standard Phase 3 retrieval and router pass all production gates."""
    thresholds = RegressionThresholds(
        min_hit_at_5=0.95,
        min_recall_at_5=0.90,
        min_context_precision=0.85,
        min_routing_accuracy=0.90,
        min_execution_success=0.95,
    )
    suite = RAGRegressionSuite(thresholds=thresholds, top_k=5)
    report = suite.evaluate_all(
        triggered_files=["phase_3_rag/src/phase_3_rag/naive_rag.py"],
        detected_categories={"prompts": ["phase_3_rag/src/phase_3_rag/naive_rag.py"]},
    )

    assert report.all_passed is True
    assert len(report.failure_reasons) == 0
    assert report.retrieval.hit_at_k >= 0.95
    assert report.retrieval.recall_at_k >= 0.90
    assert report.retrieval.context_precision >= 0.85
    assert report.router.routing_accuracy >= 0.90
    assert report.router.execution_success_rate >= 0.95


def test_regression_evaluator_fails_when_threshold_breached() -> None:
    """Verify the suite catches regressions and generates failure reasons."""
    # Set an impossible threshold to simulate a regression failure
    strict_thresholds = RegressionThresholds(
        min_hit_at_5=1.01,  # Impossible (max 1.0)
        min_routing_accuracy=1.01,  # Impossible (max 1.0)
    )
    suite = RAGRegressionSuite(thresholds=strict_thresholds, top_k=5)
    report = suite.evaluate_all()

    assert report.all_passed is False
    assert len(report.failure_reasons) >= 2
    assert any("Hit@5 regressed" in r for r in report.failure_reasons)
    assert any("Routing Accuracy regressed" in r for r in report.failure_reasons)


def test_regression_report_markdown_formatting(tmp_path: Path) -> None:
    """Verify generated markdown contains scorecard table and alerts."""
    suite = RAGRegressionSuite()
    report = suite.evaluate_all(
        triggered_files=["phase_3_rag/chunking.py"],
        detected_categories={"chunking": ["phase_3_rag/chunking.py"]},
    )

    md = report.to_markdown()
    assert "# 🧪 Phase 3 RAG Regression Suite (Task 5.2)" in md
    assert "Hit@5" in md
    assert "Context Precision (MRR)" in md
    assert "Routing Accuracy" in md
    assert "phase_3_rag/chunking.py" in md

    # Verify JSON export round-trip
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["all_passed"] is True
    assert "retrieval" in loaded
    assert "router" in loaded
