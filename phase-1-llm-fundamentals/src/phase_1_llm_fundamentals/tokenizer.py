"""Tokenization utilities and multi-language token analysis."""

from __future__ import annotations

from typing import Any

import tiktoken
from pydantic import BaseModel, Field


class TokenStats(BaseModel):
    """Statistics for tokenized text in a specific language."""

    language: str
    text: str
    token_count: int
    char_count: int
    byte_count: int
    word_count: int
    tokens: list[int]
    token_pieces: list[str]
    tokens_per_word: float = Field(
        description="Average number of tokens required per word."
    )
    chars_per_token: float = Field(
        description="Average number of characters encoded per token."
    )
    bytes_per_token: float = Field(
        description="Average number of bytes encoded per token."
    )
    fertility_ratio: float = Field(
        default=1.0,
        description="Token count relative to the English baseline.",
    )


def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Retrieve tiktoken Encoding object."""
    return tiktoken.get_encoding(encoding_name)


def tokenize_text(text: str, encoding_name: str = "cl100k_base") -> list[int]:
    """Tokenize a string into token IDs using the specified tiktoken encoding."""
    enc = get_encoding(encoding_name)
    return enc.encode(text)


def decode_tokens_to_pieces(
    tokens: list[int], encoding_name: str = "cl100k_base"
) -> list[str]:
    """Decode individual token IDs into their subword string pieces."""
    enc = get_encoding(encoding_name)
    pieces: list[str] = []
    for token_id in tokens:
        raw_bytes = enc.decode_single_token_bytes(token_id)
        try:
            pieces.append(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            pieces.append(repr(raw_bytes))
    return pieces


def calculate_token_stats(
    language: str,
    text: str,
    baseline_tokens: int | None = None,
    encoding_name: str = "cl100k_base",
) -> TokenStats:
    """Compute comprehensive token metrics for a given text snippet."""
    tokens = tokenize_text(text, encoding_name=encoding_name)
    token_pieces = decode_tokens_to_pieces(tokens, encoding_name=encoding_name)
    token_count = len(tokens)
    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    words = text.strip().split()
    word_count = len(words) if words else 0

    tokens_per_word = round(token_count / word_count, 3) if word_count > 0 else 0.0
    chars_per_token = round(char_count / token_count, 3) if token_count > 0 else 0.0
    bytes_per_token = round(byte_count / token_count, 3) if token_count > 0 else 0.0

    if baseline_tokens and baseline_tokens > 0:
        fertility_ratio = round(token_count / baseline_tokens, 3)
    else:
        fertility_ratio = 1.0

    return TokenStats(
        language=language,
        text=text,
        token_count=token_count,
        char_count=char_count,
        byte_count=byte_count,
        word_count=word_count,
        tokens=tokens,
        token_pieces=token_pieces,
        tokens_per_word=tokens_per_word,
        chars_per_token=chars_per_token,
        bytes_per_token=bytes_per_token,
        fertility_ratio=fertility_ratio,
    )


def compare_tokenization(
    corpus: dict[str, str],
    baseline_language: str = "English",
    encoding_name: str = "cl100k_base",
) -> dict[str, TokenStats]:
    """Compare tokenization across multiple translations of the same text."""
    if baseline_language not in corpus:
        msg = f"Baseline language '{baseline_language}' not found in corpus."
        raise ValueError(msg)

    baseline_tokens = len(tokenize_text(corpus[baseline_language], encoding_name))
    results: dict[str, TokenStats] = {}

    for lang, text in corpus.items():
        results[lang] = calculate_token_stats(
            language=lang,
            text=text,
            baseline_tokens=baseline_tokens,
            encoding_name=encoding_name,
        )

    return results


def calculate_cost_impact(
    token_stats: dict[str, TokenStats],
    input_cost_per_million: float = 2.50,
    output_cost_per_million: float = 10.00,
    request_volume: int = 1_000_000,
) -> dict[str, dict[str, Any]]:
    """Calculate cost projections for each language at a given request volume."""
    cost_projections: dict[str, dict[str, Any]] = {}

    for lang, stats in token_stats.items():
        total_input_tokens = stats.token_count * request_volume
        input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
        total_output_tokens = stats.token_count * request_volume
        output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost

        cost_projections[lang] = {
            "tokens_per_request": stats.token_count,
            "fertility_ratio": stats.fertility_ratio,
            "monthly_input_cost_usd": round(input_cost, 2),
            "monthly_output_cost_usd": round(output_cost, 2),
            "monthly_total_cost_usd": round(total_cost, 2),
        }

    return cost_projections
