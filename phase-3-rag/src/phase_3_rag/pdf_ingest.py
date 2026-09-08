"""PDF Ingestion Pipeline for Phase 3 RAG.

Extracts text, structured tables, and scanned pages from real PDF documents:
1. Digital PDFs: Preserves headings, sections, and paragraphs.
2. Tables: Extracts table structures with PyMuPDF/pdfplumber into clean Markdown tables.
3. Scanned PDFs: Detects image-only pages and triggers Gemini Vision OCR.
4. Exports structured Markdown documents into data/pdf_corpus/.
"""

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pymupdf
from pydantic import BaseModel, Field

from phase_3_rag.config import get_settings

DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "pdfs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "pdf_corpus"


class ExtractedTable(BaseModel):
    """Structured table extracted from a PDF page."""

    page_number: int
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render table as a GitHub Flavored Markdown table."""
        if not self.headers and not self.rows:
            return ""

        # Normalize column count
        col_count = max(
            len(self.headers),
            max((len(r) for r in self.rows), default=0),
        )
        if col_count == 0:
            return ""

        padded_headers = (
            self.headers + [""] * (col_count - len(self.headers))
            if self.headers
            else [f"Col {i + 1}" for i in range(col_count)]
        )

        header_line = (
            "| "
            + " | ".join(h.replace("\n", " ").strip() for h in padded_headers)
            + " |"
        )
        sep_line = "| " + " | ".join(":---" for _ in range(col_count)) + " |"

        row_lines: list[str] = []
        for r in self.rows:
            padded_row = r + [""] * (col_count - len(r))
            row_str = (
                "| "
                + " | ".join(
                    str(cell).replace("\n", " ").strip() for cell in padded_row
                )
                + " |"
            )
            row_lines.append(row_str)

        return "\n".join([header_line, sep_line] + row_lines)


class PDFPage(BaseModel):
    """Extracted data from a single PDF page."""

    page_number: int
    text: str
    tables: list[ExtractedTable] = Field(default_factory=list)
    is_scanned: bool = False
    ocr_applied: bool = False


class ExtractedPDF(BaseModel):
    """Complete extraction result for a PDF document."""

    id: str
    title: str
    filename: str
    category: str
    total_pages: int
    is_scanned: bool = False
    mostly_tables: bool = False
    pages: list[PDFPage] = Field(default_factory=list)
    markdown_content: str = ""
    word_count: int = 0


def is_page_scanned(page: pymupdf.Page) -> bool:
    """Determine if a page is a scanned document (lacks digital text layer)."""
    text_val = page.get_text()
    raw_text = text_val.strip() if isinstance(text_val, str) else ""
    images = page.get_images()
    # If text is nearly absent (< 25 chars) and contains embedded images/drawings
    return len(raw_text) < 25 and len(images) > 0


def extract_tables_from_page(page: pymupdf.Page, page_num: int) -> list[ExtractedTable]:
    """Extract structured tables from a PDF page using PyMuPDF table finder."""
    extracted_tables: list[ExtractedTable] = []
    try:
        tabs = page.find_tables()
        if tabs is not None and hasattr(tabs, "tables"):
            for tab in tabs.tables:
                raw_matrix = tab.extract()
                if not raw_matrix or len(raw_matrix) < 2:
                    continue
                headers = [str(c or "").strip() for c in raw_matrix[0]]
                rows = [
                    [str(cell or "").strip() for cell in row] for row in raw_matrix[1:]
                ]
                extracted_tables.append(
                    ExtractedTable(
                        page_number=page_num,
                        headers=headers,
                        rows=rows,
                    )
                )
    except Exception as err:
        print(f"  ⚠️  [Table Extraction Warning] Page {page_num}: {err}")
    return extracted_tables


def perform_gemini_ocr(
    page: pymupdf.Page,
    *,
    client: httpx.Client | None = None,
    dpi: int = 150,
) -> str:
    """Transcribe a scanned page image using Gemini Multimodal Vision API."""
    settings = get_settings()
    pix = page.get_pixmap(dpi=dpi)
    png_bytes = pix.tobytes("png")
    b64_data = base64.b64encode(png_bytes).decode("utf-8")

    url = f"{settings.gemini_base_url}/models/{settings.gemini_model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.gemini_api_key,
    }
    prompt = (
        "Transcribe all text from this scanned document page verbatim. "
        "Preserve headings, lists, numbering, and paragraph breaks. "
        "Return the extracted content in clean Markdown without commentary."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": b64_data}},
                ]
            }
        ]
    }

    should_close = False
    if client is None:
        client = httpx.Client(timeout=30.0)
        should_close = True

    try:
        for attempt in range(5):
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code in (429, 503) and attempt < 4:
                wait_time = min(30.0, 3.0 * (2**attempt))
                print(f"  ⚠️  [OCR Rate Limit] Backing off {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if "text" in p).strip()
        raise RuntimeError("OCR transcription failed after retries")
    finally:
        if should_close:
            client.close()


def extract_single_pdf(
    pdf_path: Path,
    manifest_info: dict[str, Any] | None = None,
    *,
    client: httpx.Client | None = None,
    force_ocr: bool = False,
) -> ExtractedPDF:
    """Extract text, tables, and OCR content from a single PDF document."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    doc_id = pdf_path.stem
    raw_title = manifest_info.get("title") if manifest_info else None
    title = str(raw_title) if raw_title else doc_id.replace("_", " ").title()
    category = manifest_info.get("category", "General") if manifest_info else "General"
    is_scanned_doc = manifest_info.get("is_scanned", False) if manifest_info else False
    mostly_tables = (
        manifest_info.get("mostly_tables", False) if manifest_info else False
    )

    pages: list[PDFPage] = []
    markdown_sections: list[str] = [f"# {title}\n"]

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        is_scanned = (
            is_page_scanned(page) or force_ocr or (is_scanned_doc and page_idx == 0)
        )

        if is_scanned:
            ocr_text = perform_gemini_ocr(page, client=client)
            pages.append(
                PDFPage(
                    page_number=page_num,
                    text=ocr_text,
                    tables=[],
                    is_scanned=True,
                    ocr_applied=True,
                )
            )
            markdown_sections.append(
                f"\n--- [Page {page_num} (OCR Scanned)] ---\n{ocr_text}\n"
            )
        else:
            tables = extract_tables_from_page(page, page_num)
            text_val = page.get_text()
            page_text = text_val.strip() if isinstance(text_val, str) else ""

            page_blocks: list[str] = []
            if page_text:
                page_blocks.append(page_text)

            if tables:
                page_blocks.append("\n### Extracted Tables\n")
                for t_idx, tab in enumerate(tables, start=1):
                    md_table = tab.to_markdown()
                    if md_table:
                        page_blocks.append(
                            f"**Table {t_idx} (Page {page_num})**\n\n{md_table}\n"
                        )

            combined_page_text = "\n\n".join(page_blocks)
            pages.append(
                PDFPage(
                    page_number=page_num,
                    text=combined_page_text,
                    tables=tables,
                    is_scanned=False,
                    ocr_applied=False,
                )
            )
            markdown_sections.append(
                f"\n--- [Page {page_num}] ---\n{combined_page_text}\n"
            )

    doc.close()

    full_markdown = "\n".join(markdown_sections)
    word_count = len(full_markdown.split())

    return ExtractedPDF(
        id=doc_id,
        title=title,
        filename=pdf_path.name,
        category=category,
        total_pages=total_pages,
        is_scanned=any(p.is_scanned for p in pages),
        mostly_tables=mostly_tables or sum(len(p.tables) for p in pages) >= 3,
        pages=pages,
        markdown_content=full_markdown,
        word_count=word_count,
    )


def ingest_all_pdfs(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    client: httpx.Client | None = None,
) -> list[ExtractedPDF]:
    """Ingest all PDFs in pdf_dir and export Markdown files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_map: dict[str, dict[str, Any]] = {}
    manifest_path = pdf_dir / "manifest.json"
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_map = {e["filename"]: e for e in entries}
        except Exception:
            manifest_map = {}

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    results: list[ExtractedPDF] = []
    print(f"\n📂 [PDF Ingestion] Found {len(pdf_files)} PDFs in {pdf_dir}...")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        info = manifest_map.get(pdf_path.name)
        status_tag = ""
        if info and info.get("is_scanned"):
            status_tag = " [🔍 SCANNED - OCR]"
        elif info and info.get("mostly_tables"):
            status_tag = " [📊 TABLES]"

        print(
            f"  [{idx:02d}/{len(pdf_files):02d}] "
            f"Ingesting {pdf_path.name}{status_tag}...",
            flush=True,
        )

        extracted = extract_single_pdf(pdf_path, manifest_info=info, client=client)
        results.append(extracted)

        # Export Markdown to output_dir
        md_filename = pdf_path.stem + ".md"
        out_file = output_dir / md_filename
        out_file.write_text(extracted.markdown_content, encoding="utf-8")

    print(f"\n✅ [PDF Ingestion Complete] Ingested {len(results)} PDFs to {output_dir}")
    return results


def main() -> None:
    """CLI entrypoint for PDF ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest 20+ real PDFs (scanned/tables) into RAG Markdown corpus."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing source PDFs (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir for extracted Markdown (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    client = httpx.Client(timeout=30.0)
    try:
        results = ingest_all_pdfs(args.pdf_dir, args.output_dir, client=client)

        total_words = sum(r.word_count for r in results)
        scanned_docs = [r for r in results if r.is_scanned]
        table_docs = [r for r in results if r.mostly_tables]
        total_tables = sum(sum(len(p.tables) for p in r.pages) for r in results)

        scanned_eg = scanned_docs[0].filename if scanned_docs else "None"
        table_eg = table_docs[0].filename if table_docs else "None"

        print("\n" + "=" * 70)
        print("📊 PDF INGESTION SUMMARY REPORT")
        print("=" * 70)
        print(f"  • Total PDFs Ingested:        {len(results)}")
        print(f"  • Total Word Count:           {total_words:,} words")
        print(
            f"  • Scanned Documents (OCR):    {len(scanned_docs)} (e.g. {scanned_eg})"
        )
        print(f"  • Table-Heavy Documents:      {len(table_docs)} (e.g. {table_eg})")
        print(f"  • Total Tables Extracted:     {total_tables}")
        print(f"  • Markdown Destination:       {args.output_dir}")
        print("=" * 70 + "\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
