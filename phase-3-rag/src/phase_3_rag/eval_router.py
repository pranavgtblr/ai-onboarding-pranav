"""Evaluation suite for 4-route Adaptive Query Router (Task 3.22).

Evaluates routing classification accuracy, execution success, latency,
and citation compliance across all four modalities:
1. DIRECT_LLM
2. LOCAL_CORPUS
3. STRUCTURED_DB
4. WEB_SEARCH

Generates and exports JSON and Markdown benchmark reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from phase_3_rag.router import (
    AdaptiveRAGRouter,
    RoutedRAGResponse,
    RouteTarget,
)
from phase_3_rag.web_search import MockSearchProvider, SearchResult, WebSearchRAG

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


class BenchmarkQuestion(BaseModel):
    """Definition of a benchmark question across the 4 routes."""

    question_id: str
    question: str
    expected_route: RouteTarget
    expected_source: str
    category_note: str


class QuestionRunResult(BaseModel):
    """Result for a single query in the evaluation harness."""

    question_id: str
    question: str
    expected_route: RouteTarget
    predicted_route: RouteTarget
    routing_match: bool
    confidence: float
    source_used: str
    retrieval_time_ms: float
    execution_success: bool
    has_valid_citations: bool
    citations: list[str]
    answer_preview: str


class RouterEvaluationReport(BaseModel):
    """Comprehensive benchmark report across all 4 knowledge routes."""

    pipeline_name: str = "Adaptive Omni-Router (Task 3.21 & 3.22)"
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    total_questions: int
    routing_accuracy: float
    execution_success_rate: float
    citation_compliance_rate: float
    per_route_accuracy: dict[str, float]
    per_route_avg_latency_ms: dict[str, float]
    results: list[QuestionRunResult]


BENCHMARK_DATASET: list[BenchmarkQuestion] = [
    # 1. DIRECT_LLM (Coding, Math, Language, Logic, Chit-chat)
    BenchmarkQuestion(
        question_id="DIRECT-01",
        question="Write a Python function to check if a word is a palindrome.",
        expected_route=RouteTarget.DIRECT_LLM,
        expected_source="Direct Parametric LLM",
        category_note="General programming algorithm task",
    ),
    BenchmarkQuestion(
        question_id="DIRECT-02",
        question="Calculate 128 * 4 + 56 - 12.",
        expected_route=RouteTarget.DIRECT_LLM,
        expected_source="Direct Parametric LLM",
        category_note="Pure arithmetic calculation",
    ),
    BenchmarkQuestion(
        question_id="DIRECT-03",
        question="Hello! How are you doing today?",
        expected_route=RouteTarget.DIRECT_LLM,
        expected_source="Direct Parametric LLM",
        category_note="Conversational greeting pleasantry",
    ),
    BenchmarkQuestion(
        question_id="DIRECT-04",
        question=(
            "Explain the difference between mutable and immutable types in Python."
        ),
        expected_route=RouteTarget.DIRECT_LLM,
        expected_source="Direct Parametric LLM",
        category_note="Established computer science concept",
    ),
    BenchmarkQuestion(
        question_id="DIRECT-05",
        question="Translate 'Good morning, my friend' into Spanish.",
        expected_route=RouteTarget.DIRECT_LLM,
        expected_source="Direct Parametric LLM",
        category_note="Natural language translation",
    ),
    # 2. LOCAL_CORPUS (Project Odyssey Mars Engineering & ECLSS)
    BenchmarkQuestion(
        question_id="LOCAL-01",
        question="What is the nominal cabin atmospheric pressure limit for the ECLSS?",
        expected_route=RouteTarget.LOCAL_CORPUS,
        expected_source="Local Engineering Corpus (Documents)",
        category_note="Project Odyssey ECLSS pressure operating limits",
    ),
    BenchmarkQuestion(
        question_id="LOCAL-02",
        question="What propellant mixture does the MDAS propulsion system use?",
        expected_route=RouteTarget.LOCAL_CORPUS,
        expected_source="Local Engineering Corpus (Documents)",
        category_note="Propulsion specifications from technical manual",
    ),
    BenchmarkQuestion(
        question_id="LOCAL-03",
        question="What is the maximum safe operating voltage of the primary power bus?",
        expected_route=RouteTarget.LOCAL_CORPUS,
        expected_source="Local Engineering Corpus (Documents)",
        category_note="Telemetry matrix and bus voltage thresholds",
    ),
    BenchmarkQuestion(
        question_id="LOCAL-04",
        question="What fluid mixture is used in the habitat thermal control subsystem?",
        expected_route=RouteTarget.LOCAL_CORPUS,
        expected_source="Local Engineering Corpus (Documents)",
        category_note="HTCS fluid loop specifications",
    ),
    BenchmarkQuestion(
        question_id="LOCAL-05",
        question="What is the emergency minimum cabin atmospheric pressure limit?",
        expected_route=RouteTarget.LOCAL_CORPUS,
        expected_source="Local Engineering Corpus (Documents)",
        category_note="Emergency life support threshold",
    ),
    # 3. STRUCTURED_DB (Text-to-SQL over customers, orders, products, appointments)
    BenchmarkQuestion(
        question_id="DB-01",
        question="How many total customers are in the database?",
        expected_route=RouteTarget.STRUCTURED_DB,
        expected_source="Structured Relational Database (SQL)",
        category_note="Customers table aggregation query",
    ),
    BenchmarkQuestion(
        question_id="DB-02",
        question="Show all pending orders and their total amounts.",
        expected_route=RouteTarget.STRUCTURED_DB,
        expected_source="Structured Relational Database (SQL)",
        category_note="Orders table filtering query",
    ),
    BenchmarkQuestion(
        question_id="DB-03",
        question="List all appointments scheduled with doctor name and status.",
        expected_route=RouteTarget.STRUCTURED_DB,
        expected_source="Structured Relational Database (SQL)",
        category_note="Appointments relational table query",
    ),
    BenchmarkQuestion(
        question_id="DB-04",
        question="What is the stock quantity of all products currently in inventory?",
        expected_route=RouteTarget.STRUCTURED_DB,
        expected_source="Structured Relational Database (SQL)",
        category_note="Products inventory query",
    ),
    BenchmarkQuestion(
        question_id="DB-05",
        question="How many customers are located in New York or Chicago?",
        expected_route=RouteTarget.STRUCTURED_DB,
        expected_source="Structured Relational Database (SQL)",
        category_note="Customer location filter and count",
    ),
    # 4. WEB_SEARCH (Live internet, breaking news, latest software releases)
    BenchmarkQuestion(
        question_id="WEB-01",
        question="What are the latest features released in Python 3.13?",
        expected_route=RouteTarget.WEB_SEARCH,
        expected_source="Live Web Search Engine",
        category_note="Recent software release notes",
    ),
    BenchmarkQuestion(
        question_id="WEB-02",
        question="What is the current weather forecast for Tokyo today?",
        expected_route=RouteTarget.WEB_SEARCH,
        expected_source="Live Web Search Engine",
        category_note="Volatile real-time weather query",
    ),
    BenchmarkQuestion(
        question_id="WEB-03",
        question="Who won the latest Arsenal football match yesterday?",
        expected_route=RouteTarget.WEB_SEARCH,
        expected_source="Live Web Search Engine",
        category_note="Recent sports scores",
    ),
    BenchmarkQuestion(
        question_id="WEB-04",
        question="What is the latest stock price of NVIDIA today?",
        expected_route=RouteTarget.WEB_SEARCH,
        expected_source="Live Web Search Engine",
        category_note="Volatile market financial query",
    ),
    BenchmarkQuestion(
        question_id="WEB-05",
        question="What are the breaking technology news headlines this week?",
        expected_route=RouteTarget.WEB_SEARCH,
        expected_source="Live Web Search Engine",
        category_note="Breaking current news",
    ),
]


def setup_mock_web_provider() -> WebSearchRAG:
    """Provide deterministic canned search hits for offline and eval stability."""
    provider = MockSearchProvider()
    provider.add_canned_results(
        "Python 3.13 release features",
        [
            SearchResult(
                title="Python 3.13 Release Notes",
                url="https://docs.python.org/3.13/whatsnew/3.13.html",
                snippet=(
                    "Python 3.13 introduces free-threading and an experimental JIT."
                ),
                rank=1,
                published_date="2024-10-07",
            )
        ],
    )
    return WebSearchRAG(search_provider=provider, fetch_pages=False)


def run_evaluation_suite(
    router: AdaptiveRAGRouter | None = None,
    *,
    use_mock_search: bool = True,
) -> RouterEvaluationReport:
    """Execute evaluation benchmark across all 20 benchmark questions."""
    if router is None:
        web_rag = setup_mock_web_provider() if use_mock_search else None
        router = AdaptiveRAGRouter(web_search_rag=web_rag)

    results: list[QuestionRunResult] = []
    route_correct_counts: dict[str, int] = {r.value: 0 for r in RouteTarget}
    route_total_counts: dict[str, int] = {r.value: 0 for r in RouteTarget}
    route_latency_totals: dict[str, float] = {r.value: 0.0 for r in RouteTarget}
    citation_compliant_count = 0

    for bq in BENCHMARK_DATASET:
        route_total_counts[bq.expected_route.value] += 1
        resp: RoutedRAGResponse = router.route_and_execute(bq.question)

        is_match = resp.source_used == bq.expected_route
        if is_match:
            route_correct_counts[bq.expected_route.value] += 1

        route_latency_totals[resp.source_used.value] += resp.retrieval_time_ms

        has_valid_citations = False
        if resp.source_used == RouteTarget.DIRECT_LLM:
            has_valid_citations = True  # Direct LLM does not require external citations
        else:
            has_valid_citations = len(resp.citations) > 0

        if has_valid_citations:
            citation_compliant_count += 1

        exec_success = bool(resp.answer and not resp.answer.startswith("Error"))

        results.append(
            QuestionRunResult(
                question_id=bq.question_id,
                question=bq.question,
                expected_route=bq.expected_route,
                predicted_route=resp.source_used,
                routing_match=is_match,
                confidence=resp.decision.confidence,
                source_used=resp.source_label,
                retrieval_time_ms=resp.retrieval_time_ms,
                execution_success=exec_success,
                has_valid_citations=has_valid_citations,
                citations=resp.citations,
                answer_preview=resp.answer[:120].replace("\n", " ") + "...",
            )
        )

    total_q = len(BENCHMARK_DATASET)
    overall_correct = sum(route_correct_counts.values())
    accuracy = (overall_correct / total_q) * 100.0
    success_rate = (sum(1 for r in results if r.execution_success) / total_q) * 100.0
    citation_rate = (citation_compliant_count / total_q) * 100.0

    per_route_accuracy: dict[str, float] = {}
    for r in RouteTarget:
        tot = route_total_counts[r.value]
        per_route_accuracy[r.value] = (
            (route_correct_counts[r.value] / tot * 100.0) if tot > 0 else 0.0
        )

    per_route_avg_latency: dict[str, float] = {}
    for r in RouteTarget:
        actual_count = sum(1 for res in results if res.predicted_route == r)
        per_route_avg_latency[r.value] = (
            round(route_latency_totals[r.value] / actual_count, 2)
            if actual_count > 0
            else 0.0
        )

    return RouterEvaluationReport(
        total_questions=total_q,
        routing_accuracy=round(accuracy, 2),
        execution_success_rate=round(success_rate, 2),
        citation_compliance_rate=round(citation_rate, 2),
        per_route_accuracy=per_route_accuracy,
        per_route_avg_latency_ms=per_route_avg_latency,
        results=results,
    )


def export_reports(
    report: RouterEvaluationReport,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Export evaluation report to JSON and Markdown artifacts."""
    reports_dir = output_dir or (
        Path(__file__).resolve().parent.parent.parent / "reports"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "eval_3_22_router.json"
    md_path = reports_dir / "eval_3_22_router.md"

    # 1. Export JSON Report
    json_data = report.model_dump()
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    # 2. Export Markdown Report
    status_routing = "PASS" if report.routing_accuracy >= 90 else "FAIL"
    status_exec = "PASS" if report.execution_success_rate == 100 else "REVIEW"
    status_cit = "PASS" if report.citation_compliance_rate == 100 else "REVIEW"

    md_lines = [
        "# Benchmark Report: Adaptive 4-Route Omni-Router (Task 3.22)",
        "",
        f"**Date:** {report.timestamp}  ",
        f"**Pipeline:** `{report.pipeline_name}`  ",
        (
            f"**Total Evaluated Questions:** {report.total_questions} "
            "(Balanced across 4 routes)  "
        ),
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Score | Target | Status |",
        "| :--- | :---: | :---: | :---: |",
        (
            f"| **Overall Routing Accuracy** | **{report.routing_accuracy}%** | "
            f">= 90.0% | {status_routing} |"
        ),
        (
            f"| **Execution Success Rate** | **{report.execution_success_rate}%** | "
            f"100.0% | {status_exec} |"
        ),
        (
            f"| **Citation Compliance Rate** | "
            f"**{report.citation_compliance_rate}%** | 100.0% | "
            f"{status_cit} |"
        ),
        "",
        "## 2. Per-Route Performance Breakdown",
        "",
        (
            "| Knowledge Source Modality | Questions | Route Accuracy | "
            "Avg Retrieval Latency |"
        ),
        "| :--- | :---: | :---: | :---: |",
    ]

    for route_key, acc in report.per_route_accuracy.items():
        lat = report.per_route_avg_latency_ms.get(route_key, 0.0)
        md_lines.append(f"| **`{route_key}`** | 5 | {acc:.1f}% | {lat} ms |")

    md_lines.extend(
        [
            "",
            "## 3. Detailed Question-by-Question Results",
            "",
            (
                "| ID | Question | Expected | Predicted | Match | Citations | "
                "Source Used |"
            ),
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for r in report.results:
        match_icon = "PASS" if r.routing_match else "FAIL"
        cit_desc = f"{len(r.citations)} cited" if r.citations else "Parametric"
        md_lines.append(
            f"| `{r.question_id}` | {r.question} | `{r.expected_route.value}` | "
            f"`{r.predicted_route.value}` | {match_icon} | {cit_desc} | "
            f"{r.source_used} |"
        )

    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, md_path


def main() -> None:
    """CLI runner to execute 4-route evaluation suite and commit reports."""
    parser = argparse.ArgumentParser(
        description="Run 4-route router evaluation benchmark (Task 3.22)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live web search instead of canned mock search.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(" EXECUTING 4-ROUTE OMNI-ROUTER EVALUATION SUITE (TASK 3.22)")
    print("=" * 75)

    report = run_evaluation_suite(use_mock_search=not args.live)
    json_path, md_path = export_reports(report)

    print(f"\n[+] Total Questions Tested : {report.total_questions}")
    print(f"[+] Routing Accuracy       : {report.routing_accuracy}%")
    print(f"[+] Execution Success Rate : {report.execution_success_rate}%")
    print(f"[+] Citation Compliance    : {report.citation_compliance_rate}%")
    print("\n[+] Per-Route Accuracies:")
    for r_name, acc in report.per_route_accuracy.items():
        lat = report.per_route_avg_latency_ms.get(r_name, 0.0)
        print(f"    - {r_name:<15} : {acc:5.1f}% (Avg Latency: {lat:.1f}ms)")

    print("\n[+] Exported Reports:")
    print(f"    - JSON: {json_path}")
    print(f"    - MD  : {md_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
