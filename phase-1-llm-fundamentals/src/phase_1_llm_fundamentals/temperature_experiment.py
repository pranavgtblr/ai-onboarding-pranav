"""Temperature sampling experiment analyzing LLM determinism and variance."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", alias="GEMINI_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class PromptRunResult(BaseModel):
    """Result of a single prompt generation run."""

    temperature: float
    iteration: int
    text: str
    latency_seconds: float
    char_count: int
    word_count: int


class TemperatureSummary(BaseModel):
    """Aggregate statistics for runs at a single temperature setting."""

    temperature: float
    total_runs: int
    unique_outputs_count: int
    is_fully_deterministic: bool
    average_latency_seconds: float
    average_word_count: float
    jaccard_similarity_average: float = Field(
        description="Average word-level Jaccard similarity across pairwise runs."
    )
    runs: list[PromptRunResult]


class ExperimentOutput(BaseModel):
    """Complete dataset for temperature experiment across all temperatures."""

    prompt: str
    model: str
    timestamp_utc: str
    temperatures_tested: list[float]
    runs_per_temperature: int
    summaries: dict[str, TemperatureSummary]
    all_runs: list[PromptRunResult]


DEFAULT_PROMPT = (
    "In exactly 3 sentences, describe what happens when a deep-sea submarine "
    "discovers an uncharted ancient underwater civilization."
)


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().strip().split())
    words_b = set(text_b.lower().strip().split())
    if not words_a and not words_b:
        return 1.0
    union = words_a.union(words_b)
    if not union:
        return 0.0
    intersection = words_a.intersection(words_b)
    return round(len(intersection) / len(union), 4)


def compute_group_jaccard_average(runs: list[PromptRunResult]) -> float:
    """Calculate average pairwise Jaccard similarity across all runs in a group."""
    if len(runs) <= 1:
        return 1.0
    similarities: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            sim = compute_jaccard_similarity(runs[i].text, runs[j].text)
            similarities.append(sim)
    return round(sum(similarities) / len(similarities), 4) if similarities else 1.0


def run_single_prompt(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
    iteration: int,
    max_retries: int = 5,
) -> PromptRunResult:
    """Execute a single LLM request with specific temperature and retry backoff."""
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=1024,
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.perf_counter()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            elapsed = round(time.perf_counter() - start_time, 3)
            text = (response.text or "").strip()

            return PromptRunResult(
                temperature=temperature,
                iteration=iteration,
                text=text,
                latency_seconds=elapsed,
                char_count=len(text),
                word_count=len(text.split()),
            )
        except (ServerError, ClientError) as e:
            last_error = e
            err_str = str(e)
            # Check for rate limit / quota message
            retry_match = re.search(r"retry in (\d+\.?\d*)s", err_str)
            wait_time = float(retry_match.group(1)) if retry_match else (2**attempt)
            print(
                f"  [Attempt {attempt}/{max_retries}] Rate/Server wait on "
                f"Temp {temperature}, Run {iteration}. Waiting {wait_time:.1f}s..."
            )
            time.sleep(wait_time + 1.0)
        except Exception as e:
            last_error = e
            time.sleep(2.0)

    if last_error:
        raise last_error
    msg = "Failed to generate content after max retries."
    raise RuntimeError(msg)


def execute_temperature_experiment(
    prompt: str = DEFAULT_PROMPT,
    temperatures: list[float] | None = None,
    runs_per_temperature: int = 5,
    settings: Settings | None = None,
) -> ExperimentOutput:
    """Run temperature experiment across specified temperatures and iterations."""
    if temperatures is None:
        temperatures = [0.0, 0.7, 1.2]
    if settings is None:
        settings = Settings()

    if not settings.gemini_api_key:
        msg = "GEMINI_API_KEY is not configured in .env or environment."
        raise ValueError(msg)

    client = genai.Client(api_key=settings.gemini_api_key)
    all_runs: list[PromptRunResult] = []
    summaries: dict[str, TemperatureSummary] = {}

    for temp in temperatures:
        print(f"Executing {runs_per_temperature} runs for Temperature = {temp}...")
        temp_runs: list[PromptRunResult] = []
        for i in range(1, runs_per_temperature + 1):
            run_result = run_single_prompt(
                client=client,
                model=settings.gemini_model,
                prompt=prompt,
                temperature=temp,
                iteration=i,
            )
            print(
                f"  -> Temp {temp} | Run {i}/{runs_per_temperature} done "
                f"({run_result.latency_seconds}s, {run_result.word_count} words)"
            )
            temp_runs.append(run_result)
            all_runs.append(run_result)
            time.sleep(0.4)

        unique_texts = {r.text for r in temp_runs}
        avg_latency = round(
            sum(r.latency_seconds for r in temp_runs) / len(temp_runs), 3
        )
        avg_words = round(sum(r.word_count for r in temp_runs) / len(temp_runs), 1)
        jaccard_avg = compute_group_jaccard_average(temp_runs)

        summaries[f"temp_{temp}"] = TemperatureSummary(
            temperature=temp,
            total_runs=runs_per_temperature,
            unique_outputs_count=len(unique_texts),
            is_fully_deterministic=(len(unique_texts) == 1),
            average_latency_seconds=avg_latency,
            average_word_count=avg_words,
            jaccard_similarity_average=jaccard_avg,
            runs=temp_runs,
        )

    return ExperimentOutput(
        prompt=prompt,
        model=settings.gemini_model,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        temperatures_tested=temperatures,
        runs_per_temperature=runs_per_temperature,
        summaries=summaries,
        all_runs=all_runs,
    )


def save_results(
    experiment: ExperimentOutput,
    output_dir: Path | str = "outputs",
) -> tuple[Path, Path]:
    """Persist all 15 outputs to both JSON and formatted Markdown files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "temperature_experiment_results.json"
    md_file = out_path / "temperature_experiment_results.md"

    # Save structured JSON
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(experiment.model_dump(), f, indent=2, ensure_ascii=False)

    # Save human-readable Markdown
    with md_file.open("w", encoding="utf-8") as f:
        f.write("# Temperature Sampling Experiment Results\n\n")
        f.write(f"- **Model:** `{experiment.model}`\n")
        f.write(f"- **Timestamp (UTC):** `{experiment.timestamp_utc}`\n")
        f.write(f'- **Prompt:** *"{experiment.prompt}"*\n\n')
        f.write("---\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(
            "| Temperature | Total Runs | Unique Outputs | Deterministic? | "
            "Avg Words | Pairwise Jaccard Sim |\n"
        )
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for key, summary in experiment.summaries.items():
            det_label = "✅ Yes (100%)" if summary.is_fully_deterministic else "❌ No"
            f.write(
                f"| **{summary.temperature}** | {summary.total_runs} | "
                f"{summary.unique_outputs_count} / {summary.total_runs} | "
                f"{det_label} | {summary.average_word_count} | "
                f"{summary.jaccard_similarity_average:.4f} |\n"
            )

        f.write("\n---\n\n## All 15 Generated Outputs\n\n")

        for key, summary in experiment.summaries.items():
            f.write(f"### Temperature = {summary.temperature}\n\n")
            for run in summary.runs:
                f.write(
                    f"#### Run {run.iteration} "
                    f"({run.word_count} words, {run.latency_seconds}s)\n\n"
                )
                f.write(f"> {run.text}\n\n")

    return json_file, md_file


def print_experiment_report(experiment: ExperimentOutput) -> None:
    """Print readable CLI report of results."""
    print("=" * 80)
    print(f" TEMPERATURE SAMPLING EXPERIMENT (Model: {experiment.model})")
    print("=" * 80)
    print(f'Prompt: "{experiment.prompt}"\n')

    for key, summary in experiment.summaries.items():
        print(f"--- Temperature: {summary.temperature} ---")
        print(
            f"Unique Outputs: {summary.unique_outputs_count}/{summary.total_runs} | "
            f"Deterministic: {summary.is_fully_deterministic} | "
            f"Avg Jaccard Similarity: {summary.jaccard_similarity_average:.4f}"
        )
        for r in summary.runs:
            print(f"  [Run {r.iteration} | {r.latency_seconds}s]: {r.text[:85]}...")
        print()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Temperature sampling variance & determinism benchmark"
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to test across temperatures",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save JSON and Markdown results",
    )
    args = parser.parse_args()

    print("Running temperature experiment (3 temperatures x 5 runs = 15 calls)...")
    results = execute_temperature_experiment(prompt=args.prompt)
    json_path, md_path = save_results(results, output_dir=args.output_dir)
    print_experiment_report(results)
    print(f"Results saved to:\n- {json_path}\n- {md_path}")


if __name__ == "__main__":
    main()
