"""Unit and integration tests for comparative PDF parser harness (Task 3.8).

Verifies:
1. Multi-column horizontal interleaving failure in pypdf vs. column-order in PyMuPDF.
2. Running header/footer pollution in pypdf vs. margin suppression in PyMuPDF.
3. Table flattening into unstructured word stream vs. GitHub Flavored Markdown tables.
4. Benchmark comparison report generation.
"""

from pathlib import Path

from phase_3_rag.parser_compare import (
    DEFAULT_BENCHMARK_DIR,
    analyze_header_footer_pollution,
    analyze_multi_column_interleaving,
    analyze_table_flattening,
    parse_with_pymupdf_layout,
    parse_with_pypdf,
    run_parser_comparison,
)


def test_benchmark_files_exist() -> None:
    """Verify all 3 targeted challenge PDFs exist in data/benchmarks/."""
    assert DEFAULT_BENCHMARK_DIR.exists()
    assert (DEFAULT_BENCHMARK_DIR / "multi_column_sample.pdf").exists()
    assert (DEFAULT_BENCHMARK_DIR / "polluted_header_footer_sample.pdf").exists()
    assert (DEFAULT_BENCHMARK_DIR / "table_sample.pdf").exists()


def test_multi_column_interleaving_behavior() -> None:
    """Verify pypdf interleaves columns horizontally while PyMuPDF isolates columns."""
    pdf_path = DEFAULT_BENCHMARK_DIR / "multi_column_sample.pdf"

    p_res = parse_with_pypdf(pdf_path)
    m_res = parse_with_pymupdf_layout(pdf_path, detect_multi_column=True)

    analysis = analyze_multi_column_interleaving(p_res, m_res)

    # pypdf must exhibit the interleaving bug
    assert analysis.pypdf_exhibits_bug is True, "pypdf should interleave column lines"
    # PyMuPDF must cleanly separate the columns
    assert analysis.pymupdf_exhibits_bug is False, "PyMuPDF should isolate columns"

    # Column 1 topic: Rover exploration, Column 2 topic: Nuclear reactor
    # In PyMuPDF, all rover text should appear before reactor text
    rover_idx = m_res.full_text.find("ROVER EXPLORATION")
    reactor_idx = m_res.full_text.find("NUCLEAR REACTOR")
    assert rover_idx != -1 and reactor_idx != -1
    assert rover_idx < reactor_idx


def test_header_footer_pollution_behavior() -> None:
    """Verify pypdf duplicates running headers while PyMuPDF suppresses them."""
    pdf_path = DEFAULT_BENCHMARK_DIR / "polluted_header_footer_sample.pdf"

    p_res = parse_with_pypdf(pdf_path)
    m_res = parse_with_pymupdf_layout(pdf_path, suppress_header_footer=True)

    analysis = analyze_header_footer_pollution(p_res, m_res)

    # pypdf includes repeated headers on every page
    assert analysis.pypdf_exhibits_bug is True, "pypdf should repeat running headers"
    # PyMuPDF suppresses margin noise
    assert analysis.pymupdf_exhibits_bug is False, "PyMuPDF should suppress headers"
    assert "RESTRICTED // ODYSSEY BASE" not in m_res.full_text


def test_table_flattening_behavior() -> None:
    """Verify pypdf collapses tables while PyMuPDF preserves markdown structure."""
    pdf_path = DEFAULT_BENCHMARK_DIR / "table_sample.pdf"

    p_res = parse_with_pypdf(pdf_path)
    m_res = parse_with_pymupdf_layout(pdf_path)

    analysis = analyze_table_flattening(p_res, m_res)

    # pypdf produces no markdown table headers or pipes
    assert analysis.pypdf_exhibits_bug is True, "pypdf should fail to format table"
    # PyMuPDF produces markdown table with delimiter rows
    assert analysis.pymupdf_exhibits_bug is False, (
        "PyMuPDF should produce markdown table"
    )
    assert "| Subsystem Category |" in m_res.full_text
    assert "| :--- |" in m_res.full_text
    assert "| 101.3 |" in m_res.full_text


def test_run_parser_comparison_generates_report(tmp_path: Path) -> None:
    """Verify end-to-end execution generates valid Markdown report."""
    report_file = tmp_path / "test_report.md"
    analyses, report = run_parser_comparison(
        benchmark_dir=DEFAULT_BENCHMARK_DIR,
        output_path=report_file,
    )

    assert len(analyses) == 3
    assert report_file.exists()
    assert "# Comparative PDF Parser Benchmark" in report
    assert "Multi-Column" in report
    assert "Header" in report
    assert "Table" in report
    assert "Quantitative Performance & Throughput" in report
