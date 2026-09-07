"""Unit tests for the evaluation Golden Set dataset and loader."""

from phase_3_rag.corpus import load_corpus
from phase_3_rag.golden_set import (
    GoldenQuestion,
    get_questions_by_type,
    get_questions_for_doc,
    load_golden_set,
)


def test_golden_set_count_range() -> None:
    """Verify golden set contains between 30 and 50 questions (task requirement)."""
    questions = load_golden_set()
    assert 30 <= len(questions) <= 50
    assert len(questions) == 40


def test_golden_set_unique_ids() -> None:
    """Verify all question IDs are unique."""
    questions = load_golden_set()
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))


def test_golden_set_fields_populated() -> None:
    """Verify non-empty strings and valid schema for every entry."""
    questions = load_golden_set()
    for q in questions:
        assert isinstance(q, GoldenQuestion)
        assert q.id.strip() != ""
        assert len(q.question.strip()) >= 10
        assert len(q.expected_answer.strip()) >= 10
        assert q.question_type in {
            "factoid",
            "numeric",
            "procedural",
            "multi_hop",
            "negative",
        }


def test_golden_set_expected_doc_ids_exist_in_corpus() -> None:
    """Verify all expected document IDs actually exist in the 20-doc corpus."""
    corpus = load_corpus()
    valid_doc_ids = {doc.id for doc in corpus}

    questions = load_golden_set()
    for q in questions:
        if q.question_type == "negative":
            assert q.expected_doc_ids == []
        else:
            assert len(q.expected_doc_ids) >= 1
            for doc_id in q.expected_doc_ids:
                assert doc_id in valid_doc_ids, (
                    f"Question {q.id} references invalid doc_id '{doc_id}'"
                )


def test_golden_set_corpus_coverage() -> None:
    """Verify every document in the 20-doc corpus is covered by a question."""
    corpus = load_corpus()
    questions = load_golden_set()

    referenced_docs: set[str] = set()
    for q in questions:
        referenced_docs.update(q.expected_doc_ids)

    # All 20 documents must be referenced
    assert len(referenced_docs) == 20
    for doc in corpus:
        assert doc.id in referenced_docs


def test_golden_set_filter_helpers() -> None:
    """Verify filtering questions by type and by document ID."""
    numerics = get_questions_by_type("numeric")
    assert len(numerics) >= 5
    assert all(q.question_type == "numeric" for q in numerics)

    doc_14_questions = get_questions_for_doc("doc_14")
    assert len(doc_14_questions) >= 2
    assert all("doc_14" in q.expected_doc_ids for q in doc_14_questions)
