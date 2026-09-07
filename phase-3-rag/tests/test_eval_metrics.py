"""Unit tests for retrieval and generation evaluation metrics."""

from phase_3_rag.eval_metrics import (
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
    compute_hit_at_k,
    compute_recall_at_k,
)


def test_compute_hit_at_k() -> None:
    """Verify Hit@K returns 1.0 when at least one expected doc is present."""
    assert compute_hit_at_k(["doc_01", "doc_02"], ["doc_02"]) == 1.0
    assert compute_hit_at_k(["doc_03", "doc_04"], ["doc_02"]) == 0.0
    # Negative questions (no expected doc) return 1.0
    assert compute_hit_at_k(["doc_01"], []) == 1.0


def test_compute_recall_at_k() -> None:
    """Verify Recall@K measures proportion of expected target documents found."""
    # 2 expected, both retrieved
    res = compute_recall_at_k(["doc_01", "doc_02", "doc_03"], ["doc_01", "doc_02"])
    assert res == 1.0

    # 2 expected, only 1 retrieved
    assert compute_recall_at_k(["doc_01", "doc_03"], ["doc_01", "doc_02"]) == 0.5

    # 2 expected, neither retrieved
    assert compute_recall_at_k(["doc_03", "doc_04"], ["doc_01", "doc_02"]) == 0.0


def test_compute_context_precision_mrr() -> None:
    """Verify Context Precision returns reciprocal rank of first relevant doc."""
    # First relevant doc is at rank 1: 1/1 = 1.0
    assert compute_context_precision(["doc_14", "doc_01"], ["doc_14"]) == 1.0

    # First relevant doc is at rank 2: 1/2 = 0.5
    assert compute_context_precision(["doc_01", "doc_14"], ["doc_14"]) == 0.5

    # First relevant doc is at rank 4: 1/4 = 0.25
    assert (
        compute_context_precision(["doc_01", "doc_02", "doc_03", "doc_14"], ["doc_14"])
        == 0.25
    )

    # Relevant doc not retrieved: 0.0
    assert compute_context_precision(["doc_01", "doc_02"], ["doc_14"]) == 0.0


def test_compute_answer_relevance() -> None:
    """Verify answer relevance measures coverage of ground-truth key facts."""
    answer = (
        "Odyssey Base stores 85 metric tons of liquid methane (LCH4) at -161.6°C "
        "and 340 metric tons of liquid oxygen (LOX) in Tank Farm Alpha."
    )
    key_facts = [
        "85 metric tons LCH4",
        "340 metric tons LOX",
        "-161.6°C",
    ]

    # All 3 facts present
    assert compute_answer_relevance(answer, key_facts) == 1.0

    # Only 1 fact present
    partial_answer = "The base stores 85 metric tons LCH4."
    assert abs(compute_answer_relevance(partial_answer, key_facts) - (1.0 / 3.0)) < 1e-4

    # Completely off-target answer
    off_target = "The rover has 6 wheels and suspension springs."
    assert compute_answer_relevance(off_target, key_facts) == 0.0


def test_compute_faithfulness() -> None:
    """Verify faithfulness measures factual grounding against retrieved context."""
    context = (
        "Odyssey Base stores 85 metric tons of liquid methane and 340 metric tons "
        "of liquid oxygen for the Earth Return Vehicle in double-walled tanks."
    )

    grounded_answer = (
        "The base stores 85 metric tons of liquid methane for the Earth Return Vehicle."
    )
    assert compute_faithfulness(grounded_answer, context) >= 0.9

    hallucinated_answer = (
        "The base purchases nuclear submarine propellants from civilian commercial "
        "contractors located in San Francisco California."
    )
    assert compute_faithfulness(hallucinated_answer, context) <= 0.2
