"""Tests for temperature sampling and variance experiment."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from phase_1_llm_fundamentals.temperature_experiment import (
    DEFAULT_PROMPT,
    ExperimentOutput,
    PromptRunResult,
    Settings,
    TemperatureSummary,
    compute_group_jaccard_average,
    compute_jaccard_similarity,
    execute_temperature_experiment,
    run_single_prompt,
    save_results,
)


def test_jaccard_similarity():
    """Verify Jaccard similarity metrics."""
    # Identical text
    assert (
        compute_jaccard_similarity("the quick brown fox", "the quick brown fox") == 1.0
    )

    # Completely disjoint
    assert compute_jaccard_similarity("apple orange", "banana grape") == 0.0

    # Partial overlap: union = 5 words, intersection = 2 words -> 2/5 = 0.4
    sim = compute_jaccard_similarity("the ocean is blue", "the ocean deep")
    assert sim == 0.4


def test_compute_group_jaccard_average():
    """Verify group Jaccard average across runs."""
    runs = [
        PromptRunResult(
            temperature=0.0,
            iteration=1,
            text="deep ocean",
            latency_seconds=0.5,
            char_count=10,
            word_count=2,
        ),
        PromptRunResult(
            temperature=0.0,
            iteration=2,
            text="deep ocean",
            latency_seconds=0.5,
            char_count=10,
            word_count=2,
        ),
    ]
    assert compute_group_jaccard_average(runs) == 1.0


def test_missing_api_key_raises():
    """Verify error raised if API key is not present."""
    empty_settings = Settings(GEMINI_API_KEY="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        execute_temperature_experiment(settings=empty_settings)


def test_save_results(tmp_path: Path):
    """Verify results persistence to JSON and Markdown."""
    dummy_run = PromptRunResult(
        temperature=0.0,
        iteration=1,
        text="Submarine discovers underwater ruins.",
        latency_seconds=0.42,
        char_count=38,
        word_count=4,
    )
    dummy_summary = TemperatureSummary(
        temperature=0.0,
        total_runs=1,
        unique_outputs_count=1,
        is_fully_deterministic=True,
        average_latency_seconds=0.42,
        average_word_count=4.0,
        jaccard_similarity_average=1.0,
        runs=[dummy_run],
    )
    dummy_exp = ExperimentOutput(
        prompt=DEFAULT_PROMPT,
        model="gemini-3.6-flash",
        timestamp_utc="2026-09-02T00:00:00Z",
        temperatures_tested=[0.0],
        runs_per_temperature=1,
        summaries={"temp_0.0": dummy_summary},
        all_runs=[dummy_run],
    )

    json_file, md_file = save_results(dummy_exp, output_dir=tmp_path)
    assert json_file.exists()
    assert md_file.exists()
    assert json_file.name == "temperature_experiment_results.json"
    assert md_file.name == "temperature_experiment_results.md"


@patch("google.genai.Client")
def test_run_single_prompt_mock(mock_client_class: MagicMock):
    """Verify run_single_prompt calls GenAI client properly."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Ancient glowing spires emerged from the abyssal darkness."
    mock_client.models.generate_content.return_value = mock_response

    result = run_single_prompt(
        client=mock_client,
        model="gemini-3.6-flash",
        prompt="Describe discovery",
        temperature=0.7,
        iteration=1,
    )

    assert result.temperature == 0.7
    assert result.iteration == 1
    assert "Ancient glowing spires" in result.text
    assert result.word_count == 8
    mock_client.models.generate_content.assert_called_once()
