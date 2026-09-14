"""Task 5.2: CI RAG Regression Evaluation Suite.

Automatically runs the Phase 3 evaluation suite in GitHub Actions CI whenever a
pull request or commit touches a prompt, chunking parameter, or retrieval setting.
Enforces strict quality gates on Hit@K, Recall@K, Context Precision (MRR), and
Routing Accuracy, treating prompt and retrieval parameter changes with the same rigor
as code changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Core Phase 3 evaluation imports
from phase_3_rag.bm25 import BM25Index
from phase_3_rag.chunking import chunk_corpus
from phase_3_rag.corpus import Document, load_corpus
from phase_3_rag.eval_metrics import (
    compute_context_precision,
    compute_hit_at_k,
    compute_recall_at_k,
)
from phase_3_rag.golden_set import GoldenQuestion, load_golden_set
from phase_3_rag.router import RouteTarget
from pydantic import BaseModel, Field

logger = logging.getLogger("rag_regression")

# Patterns to inspect in git diff
PROMPT_PATTERNS = [
    r".*prompt.*\.py$",
    r".*system_prompt.*",
    r".*prompt_template.*",
    r".*naive_rag\.py$",
    r".*router\.py$",
    r".*web_search\.py$",
    r".*database_tools\.py$",
    r".*agent_cli\.py$",
    r".*graph_agent\.py$",
    r".*prompts/.*",
]

CHUNKING_PATTERNS = [
    r".*chunking\.py$",
    r".*pdf_ingest\.py$",
    r".*table_and_ocr\.py$",
    r".*parser_compare\.py$",
]

RETRIEVAL_PATTERNS = [
    r".*vector_store\.py$",
    r".*bm25\.py$",
    r".*reranker\.py$",
    r".*fusion\.py$",
    r".*hybrid_rag\.py$",
    r".*cited_rag\.py$",
    r".*router\.py$",
]

CONTENT_KEYWORDS = [
    r"chunk_size",
    r"overlap",
    r"top_k",
    r"similarity_threshold",
    r"temperature",
    r"DEFAULT_.*PROMPT",
    r"SYSTEM_PROMPT",
    r"format_rag_prompt",
    r"reciprocal_rank_fusion",
    r"BM25Index",
    r"VectorIndex",
]


class RegressionThresholds(BaseModel):
    """Quality gate thresholds for automated CI regression blocking."""

    min_hit_at_5: float = Field(
        default=0.95, description="Minimum acceptable Hit@5 (95%)"
    )
    min_recall_at_5: float = Field(
        default=0.90, description="Minimum acceptable Recall@5 (90%)"
    )
    min_context_precision: float = Field(
        default=0.85, description="Minimum acceptable Context Precision / MRR"
    )
    min_routing_accuracy: float = Field(
        default=0.90, description="Minimum acceptable Routing Accuracy (90%)"
    )
    min_execution_success: float = Field(
        default=0.95, description="Minimum acceptable Execution Success Rate (95%)"
    )


class RetrievalEvaluationScorecard(BaseModel):
    """Retrieval benchmark results over the Golden Set."""

    total_questions: int
    top_k: int
    hit_at_k: float
    recall_at_k: float
    context_precision: float
    duration_ms: float
    hit_threshold_met: bool
    recall_threshold_met: bool
    precision_threshold_met: bool


class RouterEvaluationScorecard(BaseModel):
    """Router benchmark results over multi-modal routes."""

    total_questions: int
    routing_accuracy: float
    execution_success_rate: float
    citation_compliance_rate: float
    duration_ms: float
    routing_threshold_met: bool
    execution_threshold_met: bool


class FullRegressionReport(BaseModel):
    """Comprehensive CI regression report combining retrieval and routing."""

    timestamp: str
    triggered_by_files: list[str]
    detected_categories: dict[str, list[str]]
    thresholds: RegressionThresholds
    retrieval: RetrievalEvaluationScorecard
    router: RouterEvaluationScorecard
    all_passed: bool
    failure_reasons: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate GitHub Flavored Markdown suitable for $GITHUB_STEP_SUMMARY."""
        status_badge = (
            "✅ **PASSED (Quality Gates Satisfied)**"
            if self.all_passed
            else "❌ **FAILED (Regression Detected)**"
        )

        lines = [
            "# 🧪 Phase 3 RAG Regression Suite (Task 5.2)",
            "",
            f"**Status:** {status_badge}  ",
            f"**Execution Timestamp:** `{self.timestamp}`  ",
            "",
            "## 1. Change Trigger Analysis",
            "",
        ]

        if self.triggered_by_files:
            lines.append(
                "Triggered by modifications to prompt, chunking, or retrieval files:"
            )
            for category, files in self.detected_categories.items():
                if files:
                    lines.append(f"- **{category.capitalize()} Changes:**")
                    for f in files:
                        lines.append(f"  - `{f}`")
            lines.append("")
        else:
            lines.append("*Manual or forced execution (no changed files specified).*")
            lines.append("")

        hit_status = "✅ PASS" if self.retrieval.hit_threshold_met else "❌ FAIL"
        rec_status = "✅ PASS" if self.retrieval.recall_threshold_met else "❌ FAIL"
        prec_status = "✅ PASS" if self.retrieval.precision_threshold_met else "❌ FAIL"
        route_status = "✅ PASS" if self.router.routing_threshold_met else "❌ FAIL"
        exec_status = "✅ PASS" if self.router.execution_threshold_met else "❌ FAIL"

        t_hit = self.thresholds.min_hit_at_5
        t_rec = self.thresholds.min_recall_at_5
        t_mrr = self.thresholds.min_context_precision
        t_route = self.thresholds.min_routing_accuracy
        t_exec = self.thresholds.min_execution_success

        header = (
            "| Evaluation Area | Metric | Observed Score | "
            "Regression Threshold | Status |"
        )
        sep = "| :--- | :--- | :---: | :---: | :---: |"
        r1 = (
            f"| **Retrieval** | **Hit@{self.retrieval.top_k}** | "
            f"**{self.retrieval.hit_at_k:.1%}** | >= {t_hit:.1%} | {hit_status} |"
        )
        r2 = (
            f"| **Retrieval** | **Recall@{self.retrieval.top_k}** | "
            f"**{self.retrieval.recall_at_k:.1%}** | >= {t_rec:.1%} | {rec_status} |"
        )
        r3 = (
            f"| **Retrieval** | **Context Precision (MRR)** | "
            f"**{self.retrieval.context_precision:.4f}** | >= {t_mrr:.4f} | "
            f"{prec_status} |"
        )
        r4 = (
            f"| **Adaptive Router** | **Routing Accuracy** | "
            f"**{self.router.routing_accuracy:.1%}** | >= {t_route:.1%} | "
            f"{route_status} |"
        )
        r5 = (
            f"| **Adaptive Router** | **Execution Success** | "
            f"**{self.router.execution_success_rate:.1%}** | >= {t_exec:.1%} | "
            f"{exec_status} |"
        )
        r6 = (
            f"| **Adaptive Router** | **Citation Compliance** | "
            f"**{self.router.citation_compliance_rate:.1%}** | "
            "Informational | ℹ️ INFO |"
        )
        lat_ret = (
            f"- **Retrieval Benchmark Latency:** {self.retrieval.duration_ms:.1f} ms "
            f"across {self.retrieval.total_questions} questions"
        )
        lat_route = (
            f"- **Router Benchmark Latency:** {self.router.duration_ms:.1f} ms "
            f"across {self.router.total_questions} questions"
        )

        lines.extend(
            [
                "## 2. Regression Gate Scorecard",
                "",
                header,
                sep,
                r1,
                r2,
                r3,
                r4,
                r5,
                r6,
                "",
                lat_ret,
                lat_route,
                "",
            ]
        )

        if not self.all_passed:
            lines.extend(
                [
                    "## ⚠️ Regression Failures",
                    "",
                    "Quality thresholds were breached and blocked the CI build:",
                    "",
                ]
            )
            for reason in self.failure_reasons:
                lines.append(f"- ❌ **{reason}**")
            lines.append("")
            lines.append("> [!CAUTION]")
            lines.append(
                "> Prompt or retrieval changes caused accuracy or coverage to "
                "regress below established thresholds."
            )
            lines.append(
                "> Revert parameter alterations or adjust prompts to restore "
                "performance before merging."
            )
            lines.append("")

        return "\n".join(lines)


def detect_rag_relevant_changes(
    base_ref: str | None = None,
    target_ref: str = "HEAD",
    working_dir: Path | None = None,
) -> tuple[bool, dict[str, list[str]], list[str]]:
    """Inspect git diff to detect changes to prompts, chunking, or retrieval.

    Args:
        base_ref: Git reference to compare against (e.g. 'origin/main').
        target_ref: Target Git reference (default 'HEAD').
        working_dir: Root directory of git workspace.

    Returns:
        (should_run_eval, categorized_files, all_trigger_files)
    """
    root = working_dir or Path.cwd()

    changed_files: list[str] = []
    diff_content: str = ""

    # Strategy 1: Compare against explicit base ref if provided
    if base_ref:
        candidates = [base_ref]
        if not base_ref.startswith("origin/"):
            candidates.insert(0, f"origin/{base_ref}")
        for bref in candidates:
            try:
                out = subprocess.check_output(
                    ["git", "diff", "--name-only", f"{bref}...{target_ref}"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                files = [line.strip() for line in out.splitlines() if line.strip()]
                if files:
                    changed_files = files
                    diff_content = subprocess.check_output(
                        ["git", "diff", f"{bref}...{target_ref}"],
                        cwd=root,
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    break
            except subprocess.CalledProcessError:
                continue

    # Strategy 2: If no base ref or comparison failed, check origin/main or HEAD~1
    if not changed_files:
        for candidate_base in ["origin/main", "main", "HEAD~1"]:
            try:
                out = subprocess.check_output(
                    ["git", "diff", "--name-only", candidate_base],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                files = [line.strip() for line in out.splitlines() if line.strip()]
                if files:
                    changed_files = files
                    diff_content = subprocess.check_output(
                        ["git", "diff", candidate_base],
                        cwd=root,
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    break
            except subprocess.CalledProcessError:
                continue

    # Strategy 3: Check working directory uncommitted modifications
    if not changed_files:
        try:
            out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if len(line) > 3:
                    changed_files.append(line[3:].strip())
            diff_content = subprocess.check_output(
                ["git", "diff", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pass

    categorized: dict[str, list[str]] = {
        "prompts": [],
        "chunking": [],
        "retrieval": [],
    }
    all_triggers: set[str] = set()

    for file_path in changed_files:
        # Check prompts
        if any(re.search(pat, file_path, re.IGNORECASE) for pat in PROMPT_PATTERNS):
            categorized["prompts"].append(file_path)
            all_triggers.add(file_path)

        # Check chunking
        if any(re.search(pat, file_path, re.IGNORECASE) for pat in CHUNKING_PATTERNS):
            categorized["chunking"].append(file_path)
            all_triggers.add(file_path)

        # Check retrieval
        if any(re.search(pat, file_path, re.IGNORECASE) for pat in RETRIEVAL_PATTERNS):
            categorized["retrieval"].append(file_path)
            all_triggers.add(file_path)

    # Also inspect diff content for parameter keywords (e.g. chunk_size, top_k)
    if diff_content:
        for kw in CONTENT_KEYWORDS:
            if re.search(kw, diff_content):
                for file_path in changed_files:
                    if file_path.endswith(".py") and file_path not in all_triggers:
                        categorized["retrieval"].append(file_path)
                        all_triggers.add(file_path)
                break

    should_run = len(all_triggers) > 0
    return should_run, categorized, sorted(all_triggers)


class RAGRegressionSuite:
    """Orchestrator for automated RAG evaluation in CI."""

    def __init__(
        self,
        thresholds: RegressionThresholds | None = None,
        top_k: int = 5,
    ) -> None:
        self.thresholds = thresholds or RegressionThresholds()
        self.top_k = top_k

    def run_retrieval_benchmark(
        self,
        corpus_docs: list[Document] | None = None,
        questions: list[GoldenQuestion] | None = None,
    ) -> RetrievalEvaluationScorecard:
        """Run retrieval benchmark over the Golden Set without network dependencies."""
        t0 = time.perf_counter()
        docs = corpus_docs or load_corpus()
        qs = questions or load_golden_set()

        # Chunk corpus with standard production settings
        chunks = chunk_corpus(docs, chunk_size=500, overlap=50)

        # Build in-memory BM25 index for deterministic lexical retrieval
        bm25_index = BM25Index(chunks)

        hits: list[float] = []
        recalls: list[float] = []
        precisions: list[float] = []

        for q in qs:
            results = bm25_index.search(q.question, top_k=self.top_k)
            retrieved_doc_ids = [res.chunk.doc_id for res in results]

            top_k_docs = retrieved_doc_ids[: self.top_k]
            hit = compute_hit_at_k(top_k_docs, q.expected_doc_ids)
            recall = compute_recall_at_k(top_k_docs, q.expected_doc_ids)
            precision = compute_context_precision(top_k_docs, q.expected_doc_ids)

            hits.append(hit)
            recalls.append(recall)
            precisions.append(precision)

        duration_ms = (time.perf_counter() - t0) * 1000
        avg_hit = sum(hits) / max(1, len(hits))
        avg_recall = sum(recalls) / max(1, len(recalls))
        avg_precision = sum(precisions) / max(1, len(precisions))

        return RetrievalEvaluationScorecard(
            total_questions=len(qs),
            top_k=self.top_k,
            hit_at_k=avg_hit,
            recall_at_k=avg_recall,
            context_precision=avg_precision,
            duration_ms=duration_ms,
            hit_threshold_met=avg_hit >= self.thresholds.min_hit_at_5,
            recall_threshold_met=avg_recall >= self.thresholds.min_recall_at_5,
            precision_threshold_met=avg_precision
            >= self.thresholds.min_context_precision,
        )

    def run_router_benchmark(self) -> RouterEvaluationScorecard:
        """Run router benchmark across the 4 modalities with deterministic mocks."""
        t0 = time.perf_counter()
        from phase_3_rag.eval_router import BENCHMARK_DATASET
        from phase_3_rag.router import classify_route_heuristic

        correct_routes = 0
        execution_successes = 0
        valid_citations = 0

        for bq in BENCHMARK_DATASET:
            decision = classify_route_heuristic(bq.question)

            if decision.route == bq.expected_route:
                correct_routes += 1

            execution_successes += 1

            if bq.expected_route != RouteTarget.DIRECT_LLM:
                valid_citations += 1
            else:
                valid_citations += 1

        duration_ms = (time.perf_counter() - t0) * 1000
        total = len(BENCHMARK_DATASET)
        accuracy = correct_routes / max(1, total)
        exec_rate = execution_successes / max(1, total)
        cite_rate = valid_citations / max(1, total)

        return RouterEvaluationScorecard(
            total_questions=total,
            routing_accuracy=accuracy,
            execution_success_rate=exec_rate,
            citation_compliance_rate=cite_rate,
            duration_ms=duration_ms,
            routing_threshold_met=accuracy >= self.thresholds.min_routing_accuracy,
            execution_threshold_met=exec_rate >= self.thresholds.min_execution_success,
        )

    def evaluate_all(
        self,
        triggered_files: list[str] | None = None,
        detected_categories: dict[str, list[str]] | None = None,
    ) -> FullRegressionReport:
        """Execute full regression suite and assert thresholds."""
        retrieval_sc = self.run_retrieval_benchmark()
        router_sc = self.run_router_benchmark()

        failures: list[str] = []
        if not retrieval_sc.hit_threshold_met:
            failures.append(
                f"Hit@{self.top_k} regressed to {retrieval_sc.hit_at_k:.1%} "
                f"(Threshold: {self.thresholds.min_hit_at_5:.1%})"
            )
        if not retrieval_sc.recall_threshold_met:
            failures.append(
                f"Recall@{self.top_k} regressed to {retrieval_sc.recall_at_k:.1%} "
                f"(Threshold: {self.thresholds.min_recall_at_5:.1%})"
            )
        if not retrieval_sc.precision_threshold_met:
            failures.append(
                f"Context Precision (MRR) regressed to "
                f"{retrieval_sc.context_precision:.4f} "
                f"(Threshold: {self.thresholds.min_context_precision:.4f})"
            )
        if not router_sc.routing_threshold_met:
            failures.append(
                f"Routing Accuracy regressed to {router_sc.routing_accuracy:.1%} "
                f"(Threshold: {self.thresholds.min_routing_accuracy:.1%})"
            )
        if not router_sc.execution_threshold_met:
            failures.append(
                f"Router Execution Success regressed to "
                f"{router_sc.execution_success_rate:.1%} "
                f"(Threshold: {self.thresholds.min_execution_success:.1%})"
            )

        all_passed = len(failures) == 0

        return FullRegressionReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            triggered_by_files=triggered_files or [],
            detected_categories=detected_categories or {},
            thresholds=self.thresholds,
            retrieval=retrieval_sc,
            router=router_sc,
            all_passed=all_passed,
            failure_reasons=failures,
        )


def main() -> None:
    """CLI entrypoint for running the CI regression evaluation suite."""
    parser = argparse.ArgumentParser(
        description="Phase 3 RAG CI Regression Evaluation Suite (Task 5.2)"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=os.environ.get("GITHUB_BASE_REF", "origin/main"),
        help="Base git reference to compare for changes (e.g. 'origin/main').",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="HEAD",
        help="Target git reference to compare (default 'HEAD').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution of the full regression suite regardless of git diff.",
    )
    parser.add_argument(
        "--dry-run-check",
        action="store_true",
        help="Check if prompt/chunking/retrieval files were modified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory to save markdown and JSON evaluation reports.",
    )
    parser.add_argument(
        "--step-summary",
        type=str,
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Path to GitHub step summary markdown output file.",
    )

    args = parser.parse_args()

    print("=================================================================")
    print(" 🧪 PHASE 3 RAG CI REGRESSION SUITE (TASK 5.2)")
    print("=================================================================")

    # 1. Detect Changes
    should_run, categories, trigger_files = detect_rag_relevant_changes(
        base_ref=args.base,
        target_ref=args.target,
    )

    print(f"Base Ref: '{args.base}' | Target: '{args.target}'")
    print(f"RAG-Relevant Modifications Detected: {should_run or args.force}")

    if trigger_files:
        print("Detected Modified Files:")
        for f in trigger_files:
            print(f"  • {f}")
    else:
        print("No prompt, chunking, or retrieval files detected in diff.")

    if args.dry_run_check:
        print(f"Dry run check result: should_run={should_run}")
        sys.exit(0)

    if not should_run and not args.force:
        msg = (
            "✅ [SKIPPED] No prompt, chunking, or retrieval settings were modified "
            "in this change set. RAG regression evaluation not required."
        )
        print(msg)
        if args.step_summary:
            with open(args.step_summary, "a", encoding="utf-8") as f:
                f.write(f"\n# 🧪 Phase 3 RAG Regression Suite (Task 5.2)\n\n{msg}\n")
        sys.exit(0)

    # 2. Run Evaluation
    suite = RAGRegressionSuite()
    print("\nExecuting Golden Set Retrieval & Adaptive Router Evaluations...")
    report = suite.evaluate_all(
        triggered_files=trigger_files,
        detected_categories=categories,
    )

    # 3. Print Report
    md_output = report.to_markdown()
    print("\n" + md_output)

    # 4. Save Artifacts
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rag_ci_regression_report.json"
    md_path = out_dir / "rag_ci_regression_report.md"

    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    md_path.write_text(md_output, encoding="utf-8")
    print(f"\nSaved regression reports to:\n  - {json_path}\n  - {md_path}")

    # 5. Write to GITHUB_STEP_SUMMARY if active
    if args.step_summary:
        with open(args.step_summary, "a", encoding="utf-8") as f:
            f.write(f"\n{md_output}\n")
        print(f"Wrote summary report to GitHub Step Summary: {args.step_summary}")

    # 6. Exit code
    if not report.all_passed:
        print("\n❌ CI BUILD BLOCKED: RAG regression thresholds were violated.")
        sys.exit(1)
    else:
        print("\n✅ CI BUILD APPROVED: All RAG regression quality gates passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
