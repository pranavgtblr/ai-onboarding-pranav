"""Tests for multilingual tokenization and cost analysis."""

import pytest

from phase_1_llm_fundamentals.tokenizer import (
    calculate_cost_impact,
    calculate_token_stats,
    compare_tokenization,
    decode_tokens_to_pieces,
    tokenize_text,
)


def test_tokenize_text_basic():
    """Verify basic English tokenization produces non-empty token IDs."""
    text = "Hello, world!"
    tokens = tokenize_text(text, encoding_name="cl100k_base")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_decode_tokens_to_pieces():
    """Verify token IDs can be decoded back into subword pieces."""
    text = "Tokenization works well"
    tokens = tokenize_text(text, encoding_name="cl100k_base")
    pieces = decode_tokens_to_pieces(tokens, encoding_name="cl100k_base")
    assert "".join(pieces) == text


def test_calculate_token_stats():
    """Verify token stats computation."""
    text = "Artificial intelligence models process text efficiently."
    stats = calculate_token_stats(
        language="English",
        text=text,
        baseline_tokens=10,
        encoding_name="cl100k_base",
    )
    assert stats.language == "English"
    assert stats.char_count == len(text)
    assert stats.byte_count == len(text.encode("utf-8"))
    assert stats.word_count == 6
    assert stats.token_count > 0
    assert stats.fertility_ratio > 0


def test_compare_tokenization_fertility():
    """Verify comparing multiple languages correctly establishes fertility ratio."""
    corpus = {
        "English": "The weather is sunny today.",
        "Spanish": "El clima está soleado hoy.",
        "Malayalam": "ഇന്ന് കാലാവസ്ഥ വെയിൽ നിറഞ്ഞതാണ്.",
    }
    results = compare_tokenization(corpus, baseline_language="English")
    assert "English" in results
    assert "Spanish" in results
    assert "Malayalam" in results

    # English baseline should have fertility ratio 1.0
    assert results["English"].fertility_ratio == 1.0

    # Non-Latin script like Malayalam typically exhibits higher token count / fertility
    assert results["Malayalam"].token_count > results["English"].token_count
    assert results["Malayalam"].fertility_ratio > 1.0


def test_compare_tokenization_missing_baseline():
    """Verify error raised if baseline language is missing."""
    corpus = {"Spanish": "Hola"}
    with pytest.raises(ValueError, match="Baseline language 'English' not found"):
        compare_tokenization(corpus, baseline_language="English")


def test_cost_calculation():
    """Verify cost calculation scales proportionally with token fertility."""
    corpus = {
        "English": "Hello world",
        "Malayalam": "ഹലോ ലോകം",
    }
    stats = compare_tokenization(corpus, baseline_language="English")
    costs = calculate_cost_impact(
        stats,
        input_cost_per_million=2.0,
        output_cost_per_million=8.0,
        request_volume=1_000_000,
    )

    assert "English" in costs
    assert "Malayalam" in costs
    assert (
        costs["Malayalam"]["monthly_total_cost_usd"]
        > costs["English"]["monthly_total_cost_usd"]
    )
