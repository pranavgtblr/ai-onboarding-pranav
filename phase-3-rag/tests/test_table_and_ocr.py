"""Unit tests for Deliberate Table Handling and Scanned PDF OCR (Task 3.9).

Verifies:
1. ExtractedTable structured row dict and key-value record formatting.
2. Table exclusion from prose: page text never treats table cells as flattened prose.
3. Table formatting modes (markdown, structured_rows, both).
4. Scanned PDF OCR transcription on pdf_04_archival_log_scanned.pdf.
"""

import pymupdf

from phase_3_rag.pdf_ingest import (
    DEFAULT_PDF_DIR,
    ExtractedTable,
    extract_page_content_layout_aware,
    extract_single_pdf,
    perform_gemini_ocr,
    transcribe_scanned_pdf,
)


def test_extracted_table_to_structured_rows() -> None:
    """Verify to_structured_rows maps column headers to row cell values."""
    table = ExtractedTable(
        page_number=1,
        headers=["Subsystem", "Metric", "Nominal Value", "Unit"],
        rows=[
            ["Life Support", "Habitat Pressure", "101.3", "kPa"],
            ["Life Support", "O2 Partial Pressure", "21.0", "kPa"],
        ],
    )
    rows = table.to_structured_rows()

    assert len(rows) == 2
    assert rows[0] == {
        "Subsystem": "Life Support",
        "Metric": "Habitat Pressure",
        "Nominal Value": "101.3",
        "Unit": "kPa",
    }
    assert rows[1]["Metric"] == "O2 Partial Pressure"
    assert rows[1]["Nominal Value"] == "21.0"


def test_extracted_table_to_row_records_text() -> None:
    """Verify to_row_records_text formats self-contained key-value strings."""
    table = ExtractedTable(
        page_number=1,
        headers=["Sensor", "Limit", "Status"],
        rows=[
            ["Temp Sensor A", "< 150 C", "NOMINAL"],
            ["Voltage Bus B", "> 110 V", "WARNING"],
        ],
    )
    records_text = table.to_row_records_text()
    lines = records_text.strip().splitlines()

    assert len(lines) == 2
    assert lines[0] == (
        "- [Row 1] Sensor: Temp Sensor A | Limit: < 150 C | Status: NOMINAL"
    )
    assert lines[1] == (
        "- [Row 2] Sensor: Voltage Bus B | Limit: > 110 V | Status: WARNING"
    )


def test_table_prose_deduplication_on_pdf_07() -> None:
    """Verify table cells are not flattened into unstructured prose on pdf_07."""
    pdf_path = DEFAULT_PDF_DIR / "pdf_07_telemetry_matrix_tables.pdf"
    assert pdf_path.exists(), f"Benchmark PDF missing: {pdf_path}"

    doc = pymupdf.open(pdf_path)
    page = doc[0]

    # Layout-aware extraction with both markdown and structured rows
    combined_text, tables = extract_page_content_layout_aware(
        page, 1, table_format="both"
    )
    doc.close()

    assert len(tables) == 3

    # 1. Verify structured Markdown tables exist
    assert (
        "| Subsystem Sensor | Nominal | Min Alarm | Max Alarm | Unit |" in combined_text
    )
    assert "| Habitat Pressure | 101.3 | 95.0 | 105.0 | kPa |" in combined_text

    # 2. Verify structured key-value rows exist
    assert (
        "- [Row 1] Subsystem Sensor: Habitat Pressure | Nominal: 101.3 |"
        in combined_text
    )
    assert "Tank Identifier: TK-LOX-01 | Cryogen: Liquid Oxygen" in combined_text

    # 3. Verify naive prose concatenation is absent:
    # In naive parsers, table headers and cells appear as raw vertical lines
    assert "Subsystem Sensor\nNominal\nMin Alarm" not in combined_text, (
        "Table headers should not be dumped as raw prose"
    )


def test_table_format_options() -> None:
    """Verify table_format parameter correctly filters markdown vs structured rows."""
    pdf_path = DEFAULT_PDF_DIR / "pdf_07_telemetry_matrix_tables.pdf"
    doc = pymupdf.open(pdf_path)
    page = doc[0]

    # Format: markdown only
    text_md, _ = extract_page_content_layout_aware(page, 1, table_format="markdown")
    assert "| Subsystem Sensor |" in text_md
    assert "*Structured Rows" not in text_md
    assert "- [Row 1]" not in text_md

    # Format: structured_rows only
    text_rows, _ = extract_page_content_layout_aware(
        page, 1, table_format="structured_rows"
    )
    assert "- [Row 1] Subsystem Sensor: Habitat Pressure" in text_rows
    assert "| :--- |" not in text_rows

    doc.close()


def test_scanned_pdf_ocr_execution() -> None:
    """Verify OCR transcription on the scanned archival log PDF."""
    pdf_path = DEFAULT_PDF_DIR / "pdf_04_archival_log_scanned.pdf"
    assert pdf_path.exists(), f"Scanned PDF missing: {pdf_path}"

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    ocr_text = perform_gemini_ocr(page)
    doc.close()

    assert len(ocr_text) > 100
    assert (
        "DECLASSIFIED TECHNICAL FIELD LOG" in ocr_text
        or "SOL 01" in ocr_text
        or "ARCHIVAL RECORD" in ocr_text
    )
    assert (
        "Atmospheric sampling" in ocr_text
        or "Regolith core drill" in ocr_text
        or "Stirling engine" in ocr_text
    )


def test_transcribe_scanned_pdf_convenience() -> None:
    """Verify transcribe_scanned_pdf processes the entire scanned document."""
    pdf_path = DEFAULT_PDF_DIR / "pdf_04_archival_log_scanned.pdf"
    transcript = transcribe_scanned_pdf(pdf_path)

    assert "--- [Page 1 (OCR Scanned)] ---" in transcript
    assert "SOL 04" in transcript or "COMMANDER MISSION LOG" in transcript


def test_extract_single_pdf_with_deliberate_tables() -> None:
    """Verify extract_single_pdf exports clean structured document for pdf_07."""
    pdf_path = DEFAULT_PDF_DIR / "pdf_07_telemetry_matrix_tables.pdf"
    extracted = extract_single_pdf(pdf_path, table_format="both")

    assert extracted.id == "pdf_07_telemetry_matrix_tables"
    assert extracted.mostly_tables is True
    assert len(extracted.pages[0].tables) == 3
    assert "| Subsystem Sensor |" in extracted.markdown_content
    assert "*Structured Rows" in extracted.markdown_content
