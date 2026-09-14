"""Unit and integration tests for Task 5.3: LLM-as-a-Judge and Human Calibration."""

from __future__ import annotations

import json
from pathlib import Path

from phase_5_production.llm_judge import (
    ComparisonResult,
    HumanSample,
    JudgeVerdict,
    LLMJudgeEvaluator,
    compute_agreement_metrics,
    interpret_cohens_kappa,
    run_calibration_suite,
)

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "human_benchmark_30.json"
)


def test_human_sample_schema_and_distribution() -> None:
    """Verify the 30-sample benchmark dataset exists and is balanced."""
    assert BENCHMARK_PATH.exists(), f"Benchmark dataset missing at {BENCHMARK_PATH}"
    raw_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    assert len(raw_data) == 30, f"Expected exactly 30 samples, found {len(raw_data)}"
    samples = [HumanSample.model_validate(item) for item in raw_data]

    pass_count = sum(1 for s in samples if s.human_label == "PASS")
    fail_count = sum(1 for s in samples if s.human_label == "FAIL")

    assert pass_count == 15, f"Expected 15 PASS samples, got {pass_count}"
    assert fail_count == 15, f"Expected 15 FAIL samples, got {fail_count}"

    for s in samples:
        assert s.id.startswith("SAMPLE-")
        assert len(s.query) > 5
        assert len(s.retrieved_context) > 10
        assert len(s.model_output) > 5
        assert len(s.human_rationale) > 5


def test_judge_verdict_normalization_and_validation() -> None:
    """Verify JudgeVerdict normalizes casing, floats, and scores."""
    v1 = JudgeVerdict(
        verdict="pass",  # type: ignore[arg-type]
        score=1.0,  # type: ignore[arg-type]
        faithfulness_score=5.0,  # type: ignore[arg-type]
        relevance_score=5.0,  # type: ignore[arg-type]
        reasoning="All claims match context.",
        detected_hallucinations=[],
    )
    assert v1.verdict == "PASS"
    assert v1.score == 1
    assert v1.faithfulness_score == 5
    assert v1.relevance_score == 5

    v2 = JudgeVerdict(
        verdict="fail",  # type: ignore[arg-type]
        score=0.0,  # type: ignore[arg-type]
        faithfulness_score=2,
        relevance_score=3,
        reasoning="Hallucinated claims.",
        detected_hallucinations=["invented_fact"],
    )
    assert v2.verdict == "FAIL"
    assert v2.score == 0
    assert len(v2.detected_hallucinations) == 1


def test_cohens_kappa_perfect_agreement() -> None:
    """Verify that perfect agreement yields Cohen's Kappa = 1.0."""
    raw_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))[:10]
    comparisons: list[ComparisonResult] = []

    for item in raw_data:
        s = HumanSample.model_validate(item)
        c_class = "TP" if s.human_label == "PASS" else "TN"
        comparisons.append(
            ComparisonResult(
                sample=s,
                judge=JudgeVerdict(
                    verdict=s.human_label,
                    score=s.human_score,
                    faithfulness_score=5 if s.human_label == "PASS" else 2,
                    relevance_score=5 if s.human_label == "PASS" else 2,
                    reasoning="Simulated perfect judge.",
                ),
                agreement=True,
                confusion_class=c_class,
            )
        )

    metrics = compute_agreement_metrics(comparisons)
    assert metrics.raw_agreement_rate == 1.0
    assert metrics.cohens_kappa == 1.0
    assert "Near-Perfect" in metrics.kappa_interpretation
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0


def test_cohens_kappa_chance_agreement() -> None:
    """Verify that random 50/50 prediction yields Cohen's Kappa near 0.0."""
    raw_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    comparisons: list[ComparisonResult] = []

    # Simulate a judge that always outputs PASS regardless of human label
    for item in raw_data:
        s = HumanSample.model_validate(item)
        c_class = "TP" if s.human_label == "PASS" else "FP"
        agreement = s.human_label == "PASS"
        comparisons.append(
            ComparisonResult(
                sample=s,
                judge=JudgeVerdict(
                    verdict="PASS",
                    score=1,
                    faithfulness_score=5,
                    relevance_score=5,
                    reasoning="Always passes.",
                ),
                agreement=agreement,
                confusion_class=c_class,
            )
        )

    metrics = compute_agreement_metrics(comparisons)
    assert metrics.raw_agreement_rate == 0.50
    # A degenerate all-PASS judge yields Kappa = 0.0 against balanced 50/50 dataset
    assert metrics.cohens_kappa == 0.0
    assert metrics.leniency_bias_rate == 0.50
    assert metrics.harshness_bias_rate == 0.0


def test_cohens_kappa_interpretation_tiers() -> None:
    """Verify standard Landis & Koch interpretation labels."""
    assert interpret_cohens_kappa(-0.1) == "Poor (Worse than Chance)"
    assert interpret_cohens_kappa(0.15) == "Slight Agreement"
    assert interpret_cohens_kappa(0.35) == "Fair Agreement"
    assert interpret_cohens_kappa(0.55) == "Moderate Agreement"
    assert interpret_cohens_kappa(0.75) == "Substantial Agreement"
    assert interpret_cohens_kappa(0.95) == "Near-Perfect / Strong Agreement"


def test_judge_detects_subtle_hallucinations() -> None:
    """Verify evaluator catches subtle ungrounded additions."""
    evaluator = LLMJudgeEvaluator()
    raw_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    # Sample 3 contains 'liquid hydrogen' which is absent from context
    s3 = HumanSample.model_validate(raw_data[2])
    v3 = evaluator.judge_sample(s3, use_mock=True)
    assert v3.verdict == "FAIL"
    assert v3.score == 0
    assert "liquid hydrogen" in v3.detected_hallucinations

    # Sample 11 invents 'bluetooth 5.4'
    s11 = HumanSample.model_validate(raw_data[10])
    v11 = evaluator.judge_sample(s11, use_mock=True)
    assert v11.verdict == "FAIL"
    assert "bluetooth 5.4" in v11.detected_hallucinations


def test_judge_approves_grounded_answers() -> None:
    """Verify evaluator approves faithful answers."""
    evaluator = LLMJudgeEvaluator()
    raw_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    s1 = HumanSample.model_validate(raw_data[0])
    v1 = evaluator.judge_sample(s1, use_mock=True)
    assert v1.verdict == "PASS"
    assert v1.score == 1
    assert v1.faithfulness_score == 5
    assert len(v1.detected_hallucinations) == 0


def test_full_calibration_benchmark_suite(tmp_path: Path) -> None:
    """Run full calibration benchmark across all 30 hand-labeled samples."""
    evaluator = LLMJudgeEvaluator()
    report = run_calibration_suite(
        dataset_path=BENCHMARK_PATH,
        evaluator=evaluator,
        use_mock=True,
    )

    assert report.metrics.total_samples == 30
    assert report.metrics.raw_agreement_rate >= 0.85
    assert report.metrics.cohens_kappa >= 0.70
    assert report.metrics.f1_score >= 0.85

    md = report.to_markdown()
    assert "# ⚖️ LLM-as-a-Judge Calibration & Agreement Report" in md
    assert "Raw Agreement Rate" in md
    assert "Cohen's Kappa (κ)" in md
    assert "Confusion Matrix" in md

    # Verify JSON report export
    json_path = tmp_path / "calibration.json"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["total_samples"] == 30
