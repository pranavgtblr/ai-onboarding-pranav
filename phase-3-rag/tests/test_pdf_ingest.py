"""Unit and integration tests for PDF ingestion pipeline (Task 3.7).

Verifies:
1. 20+ real PDFs exist in manifest.
2. At least one scanned document and one table-heavy document exist.
3. Structured table extraction into Markdown format.
4. Scanned page detection heuristic.
5. PDF extraction pipeline execution and corpus loading.
"""

import json
from pathlib import Path

import pymupdf

from phase_3_rag.corpus import Document, load_pdf_corpus
from phase_3_rag.pdf_ingest import (
    DEFAULT_PDF_DIR,
    ExtractedTable,
    extract_single_pdf,
    is_page_scanned,
)


def test_manifest_contains_20_or_more_pdfs() -> None:
    """Verify at least 20 real PDFs are cataloged in data/pdfs/manifest.json."""
    manifest_path = DEFAULT_PDF_DIR / "manifest.json"
    assert manifest_path.exists(), "manifest.json must exist"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) >= 20, f"Expected >= 20 PDFs, found {len(manifest)}"

    # Check that each physical file exists on disk
    for entry in manifest:
        pdf_file = DEFAULT_PDF_DIR / entry["filename"]
        assert pdf_file.exists(), f"PDF file missing on disk: {pdf_file}"
        assert pdf_file.stat().st_size > 500, f"PDF file {pdf_file} is too small"


def test_manifest_has_scanned_and_table_documents() -> None:
    """Verify manifest contains at least one scanned doc and one table-heavy doc."""
    manifest_path = DEFAULT_PDF_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    scanned_docs = [e for e in manifest if e.get("is_scanned") is True]
    table_docs = [e for e in manifest if e.get("mostly_tables") is True]

    assert len(scanned_docs) >= 1, "Must contain at least one scanned document"
    assert len(table_docs) >= 1, "Must contain at least one table-heavy document"
    assert scanned_docs[0]["filename"] == "pdf_04_archival_log_scanned.pdf"
    assert table_docs[0]["filename"] == "pdf_07_telemetry_matrix_tables.pdf"


def test_extracted_table_to_markdown() -> None:
    """Verify ExtractedTable formats cleanly into GitHub Flavored Markdown."""
    table = ExtractedTable(
        page_number=1,
        headers=["Subsystem", "Telemetry", "Unit"],
        rows=[
            ["Reactor Core", "800.0", "degC"],
            ["Chamber Pressure", "101.3", "kPa"],
        ],
    )
    md = table.to_markdown()
    lines = md.strip().splitlines()

    assert len(lines) == 4
    assert lines[0] == "| Subsystem | Telemetry | Unit |"
    assert lines[1] == "| :--- | :--- | :--- |"
    assert lines[2] == "| Reactor Core | 800.0 | degC |"
    assert lines[3] == "| Chamber Pressure | 101.3 | kPa |"


def test_extract_digital_pdf() -> None:
    """Verify extraction of standard digital text PDF."""
    pdf_path = DEFAULT_PDF_DIR / "pdf_01_atmospheric_entry.pdf"
    extracted = extract_single_pdf(pdf_path)

    assert extracted.id == "pdf_01_atmospheric_entry"
    assert extracted.total_pages >= 1
    assert extracted.is_scanned is False
    assert extracted.word_count > 50
    assert "Mars Atmospheric Entry" in extracted.markdown_content
    assert "PICA-X" in extracted.markdown_content


def test_extract_table_heavy_pdf() -> None:
    """Verify table extraction from table-heavy document (pdf_07)."""
    pdf_path = DEFAULT_PDF_DIR / "pdf_07_telemetry_matrix_tables.pdf"
    extracted = extract_single_pdf(pdf_path)

    assert extracted.id == "pdf_07_telemetry_matrix_tables"
    assert extracted.mostly_tables is True
    # Verify table formatting in markdown
    assert (
        "| Subsystem Sensor |" in extracted.markdown_content
        or "| Tank Identifier |" in extracted.markdown_content
    )
    assert "| :--- |" in extracted.markdown_content
    assert "101.3" in extracted.markdown_content


def test_is_page_scanned_detection() -> None:
    """Verify scanned page detection accurately flags image-only bitmap pages."""
    scanned_path = DEFAULT_PDF_DIR / "pdf_04_archival_log_scanned.pdf"
    doc_scanned = pymupdf.open(scanned_path)
    assert is_page_scanned(doc_scanned[0]) is True
    doc_scanned.close()

    digital_path = DEFAULT_PDF_DIR / "pdf_01_atmospheric_entry.pdf"
    doc_digital = pymupdf.open(digital_path)
    assert is_page_scanned(doc_digital[0]) is False
    doc_digital.close()


def test_load_pdf_corpus_reads_markdown(tmp_path: Path) -> None:
    """Verify load_pdf_corpus successfully parses exported markdown files."""
    dummy_file = tmp_path / "pdf_01_sample.md"
    dummy_file.write_text(
        "# Sample Technical Manual\n\nSample text content for RAG retrieval testing.",
        encoding="utf-8",
    )

    docs = load_pdf_corpus(data_dir=tmp_path)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].id == "pdf_01_sample"
    assert docs[0].title == "Sample Technical Manual"
    assert "Sample text content" in docs[0].text
