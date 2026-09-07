"""Evaluation Golden Set loader and models for RAG benchmarking.

Loads and validates the curated 40-question golden dataset used for measuring
retrieval hit rate, MRR, citation precision, and answer faithfulness.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_GOLDEN_SET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "golden_set.json"
)


class GoldenQuestion(BaseModel):
    """Represents a single benchmark question with ground-truth target context."""

    id: str = Field(description="Unique question identifier, e.g. q01")
    question: str = Field(description="The user question string")
    expected_answer: str = Field(
        description="Ground-truth human-verified reference answer"
    )
    expected_doc_ids: list[str] = Field(
        description="Document IDs required to answer the question"
    )
    question_type: str = Field(
        description="Category: factoid, numeric, procedural, multi_hop, negative"
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Key phrases or factual nuggets required in a correct response",
    )


def load_golden_set(
    filepath: Path | None = None,
) -> list[GoldenQuestion]:
    """Load and validate the evaluation golden set from disk.

    Args:
        filepath: Optional custom path to golden_set.json.

    Returns:
        List of validated GoldenQuestion instances.
    """
    path = filepath or DEFAULT_GOLDEN_SET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Golden set file not found: {path}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Golden set must be a JSON array, got {type(raw_data)}")

    return [GoldenQuestion.model_validate(item) for item in raw_data]


def get_questions_by_type(
    question_type: str,
    filepath: Path | None = None,
) -> list[GoldenQuestion]:
    """Filter golden questions by their benchmark type."""
    questions = load_golden_set(filepath)
    return [q for q in questions if q.question_type == question_type]


def get_questions_for_doc(
    doc_id: str,
    filepath: Path | None = None,
) -> list[GoldenQuestion]:
    """Filter golden questions that target a specific source document."""
    questions = load_golden_set(filepath)
    return [q for q in questions if doc_id in q.expected_doc_ids]
