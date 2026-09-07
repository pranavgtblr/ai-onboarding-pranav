"""Corpus loader and manager for Phase 3 RAG.

Loads the 20 curated technical operations documents from data/corpus/.
"""

import re
from pathlib import Path

from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


class Document(BaseModel):
    """Represents a source document in the RAG corpus."""

    id: str = Field(description="Unique document identifier, e.g. doc_01")
    title: str = Field(description="Human-readable title of the document")
    filename: str = Field(
        description="Filename on disk, e.g. doc_01_oxygen_generation.md"
    )
    category: str = Field(description="Subsystem category, e.g. Life Support, Power")
    source_url: str = Field(
        default="",
        description="Canonical URL for referencing this document in citations",
    )
    text: str = Field(description="Full text content of the document")


CATEGORY_MAP: dict[str, str] = {
    "doc_01": "Life Support",
    "doc_02": "Life Support",
    "doc_03": "Environmental Control",
    "doc_04": "Power Systems",
    "doc_05": "Power Systems",
    "doc_06": "Power Systems",
    "doc_07": "Crew Equipment",
    "doc_08": "Habitat Operations",
    "doc_09": "Life Support",
    "doc_10": "Communications",
    "doc_11": "Mobility",
    "doc_12": "Infrastructure",
    "doc_13": "Medical & Health",
    "doc_14": "ISRU & Propulsion",
    "doc_15": "Life Support",
    "doc_16": "Safety & Survival",
    "doc_17": "Historical Records",
    "doc_18": "Emergency Procedures",
    "doc_19": "ISRU & Propulsion",
    "doc_20": "Governance & Administration",
}


def load_corpus(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load all markdown documents from data/corpus into typed Document models."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {data_dir}")

    files = sorted(data_dir.glob("doc_*.md"))
    documents: list[Document] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        # Extract title from first markdown H1 line (# Title)
        first_line = text.splitlines()[0] if text else path.stem
        title = re.sub(r"^#\s*", "", first_line).strip()
        parts = path.stem.split("_")
        doc_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else path.stem
        category = CATEGORY_MAP.get(doc_id, "General Operations")

        documents.append(
            Document(
                id=doc_id,
                title=title,
                filename=path.name,
                category=category,
                source_url=f"https://odyssey.base.internal/docs/{path.name}",
                text=text,
            )
        )

    return documents
