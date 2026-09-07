"""Unit and integration tests for citation extraction and validation."""

from unittest.mock import MagicMock, patch

import httpx

from phase_3_rag.bm25 import BM25Index
from phase_3_rag.chunking import Chunk
from phase_3_rag.citations import (
    extract_and_validate_citations,
    format_cited_prompt,
)
from phase_3_rag.cited_rag import execute_cited_rag


def _make_test_chunk(chunk_id: str, title: str, filename: str) -> Chunk:
    doc_id = chunk_id.split("_c")[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_title=title,
        category="Ops",
        source_url=f"https://odyssey.base.internal/docs/{filename}",
        chunk_index=0,
        text=f"Content for {chunk_id}",
        token_count=10,
    )


def test_extract_and_validate_citations_clean() -> None:
    """Verify inline citations are extracted, mapped to URLs, and footnoted."""
    c1 = _make_test_chunk(
        "doc_14_c0",
        "Cryogenic Fuel Storage",
        "doc_14_cryogenic_storage.md",
    )
    c2 = _make_test_chunk(
        "doc_01_c0",
        "Oxygen Generation Systems",
        "doc_01_oxygen_generation.md",
    )

    raw_text = (
        "Odyssey Base stores 85 metric tons of LCH4 [doc_14_c0]. "
        "Oxygen is recovered through solid polymer electrolysis [doc_01_c0]."
    )

    result = extract_and_validate_citations(raw_text, [c1, c2], query="test query")

    assert len(result.citations) == 2
    assert result.unverified_citations == []
    assert result.is_fully_verified is True

    # Check citation 1
    assert result.citations[0].chunk_id == "doc_14_c0"
    assert result.citations[0].source_title == "Cryogenic Fuel Storage"
    assert "doc_14_cryogenic_storage.md" in result.citations[0].source_url

    # Check citation 2
    assert result.citations[1].chunk_id == "doc_01_c0"
    assert result.citations[1].source_title == "Oxygen Generation Systems"
    assert "doc_01_oxygen_generation.md" in result.citations[1].source_url

    # Check footnote conversion in clean_answer
    assert "[1]" in result.clean_answer
    assert "[2]" in result.clean_answer
    assert "[doc_14_c0]" not in result.clean_answer


def test_detect_hallucinated_citations() -> None:
    """Verify unretrieved/invented chunk IDs are flagged as unverified citations."""
    c1 = _make_test_chunk(
        "doc_14_c0",
        "Cryogenic Storage",
        "doc_14_cryogenic_storage.md",
    )

    # Model hallucinates doc_99_c0 and doc_42_c1
    raw_text = (
        "Valid statement [doc_14_c0]. "
        "Invented claim [doc_99_c0] and another hallucination [doc_42_c1]."
    )

    result = extract_and_validate_citations(raw_text, [c1])

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "doc_14_c0"
    assert "doc_99_c0" in result.unverified_citations
    assert "doc_42_c1" in result.unverified_citations
    assert result.is_fully_verified is False

    display_output = result.format_display()
    assert "UNVERIFIED / HALLUCINATED" in display_output
    assert "doc_99_c0" in display_output


def test_format_cited_prompt_contains_rules_and_urls() -> None:
    """Verify prompt explicitly contains citation rules and canonical source URLs."""
    c = _make_test_chunk(
        "doc_06_c0",
        "Energy Storage",
        "doc_06_battery_storage.md",
    )
    prompt = format_cited_prompt("Battery runtime?", [c])

    assert "MANDATORY CITATION RULES:" in prompt
    assert "doc_06_c0" in prompt
    assert "https://odyssey.base.internal/docs/doc_06_battery_storage.md" in prompt


def test_execute_cited_rag_mocked() -> None:
    """Verify end-to-end cited RAG pipeline with mocked LLM response."""
    c = _make_test_chunk(
        "doc_14_c0",
        "Cryogenic Storage",
        "doc_14_cryogenic_storage.md",
    )
    bm25 = BM25Index([c])
    mock_client = MagicMock(spec=httpx.Client)

    mock_llm_text = "Odyssey Base stores 85 metric tons of liquid methane [doc_14_c0]."

    with patch(
        "phase_3_rag.cited_rag.generate_llm_response",
        return_value=(mock_llm_text, 180, 22),
    ):
        cited_answer, p_toks, c_toks = execute_cited_rag(
            "liquid methane storage",
            [c],
            bm25,
            client=mock_client,
            top_k=1,
        )

        assert cited_answer.is_fully_verified is True
        assert len(cited_answer.citations) == 1
        assert cited_answer.citations[0].chunk_id == "doc_14_c0"
        assert p_toks == 180
        assert c_toks == 22
