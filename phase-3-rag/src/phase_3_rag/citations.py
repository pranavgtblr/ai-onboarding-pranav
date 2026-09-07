"""Citation validation, extraction, and footnoting for RAG answers.

Ensures every factual claim in a generated answer maps back to verified
chunk IDs and canonical source URLs from the retrieved context.
"""

import re

from pydantic import BaseModel, Field

from phase_3_rag.chunking import Chunk


class Citation(BaseModel):
    """Represents a verified document chunk reference used in an answer."""

    chunk_id: str = Field(description="Referenced chunk identifier, e.g. doc_14_c0")
    doc_id: str = Field(description="Parent document identifier, e.g. doc_14")
    source_title: str = Field(description="Human-readable title of source document")
    source_url: str = Field(description="Canonical URL of source document")


class CitedAnswer(BaseModel):
    """Complete synthesized answer paired with verified and unverified citations."""

    query: str = Field(description="User query that prompted this answer")
    raw_answer: str = Field(description="Raw LLM output with inline citation tags")
    clean_answer: str = Field(description="Clean answer formatted for end users")
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of verified citations extracted from the answer",
    )
    unverified_citations: list[str] = Field(
        default_factory=list,
        description="Any cited chunk IDs that were NOT in the retrieved context",
    )

    @property
    def is_fully_verified(self) -> bool:
        """Return True if all citations map to actual retrieved context."""
        return len(self.citations) > 0 and len(self.unverified_citations) == 0

    def format_display(self) -> str:
        """Format the answer and footnotes into clean readable Markdown."""
        lines = [self.clean_answer.strip(), ""]

        if self.citations:
            lines.append("📚 [VERIFIED CITATIONS]")
            for i, cite in enumerate(self.citations, start=1):
                lines.append(f"  [{i}] Chunk: {cite.chunk_id} ({cite.doc_id})")
                lines.append(f"      Title: {cite.source_title}")
                lines.append(f"      URL:   {cite.source_url}")
            lines.append("")

        if self.unverified_citations:
            lines.append("⚠️ [UNVERIFIED / HALLUCINATED CITATIONS]")
            for bad_id in self.unverified_citations:
                lines.append(f"  - {bad_id} (Not present in retrieved context!)")
            lines.append("")

        return "\n".join(lines)


def format_cited_prompt(query: str, retrieved_chunks: list[Chunk]) -> str:
    """Format technical RAG prompt with strict inline citation rules."""
    context_blocks = []
    for c in retrieved_chunks:
        header = (
            f"--- [Doc: {c.source_title} | ID: {c.chunk_id} | URL: {c.source_url}] ---"
        )
        context_blocks.append(f"{header}\n{c.text.strip()}\n")

    context_str = "\n".join(context_blocks)

    return (
        "You are an AI mission operations assistant for Project Odyssey Mars Base.\n"
        "Answer the question using ONLY the provided technical context.\n\n"
        "MANDATORY CITATION RULES:\n"
        "1. Every factual claim or sentence MUST include an inline citation tag in "
        "square brackets referencing the exact Chunk ID, e.g. [doc_14_c0].\n"
        "2. ONLY cite Chunk IDs that appear in the provided context below.\n"
        "3. If the context does not contain the answer, state that the information is "
        "not available from the provided documents. Do not speculate.\n\n"
        f"=== RETRIEVED CONTEXT ({len(retrieved_chunks)} CHUNKS) ===\n"
        f"{context_str}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {query}\n\n"
        "Answer with inline citations:"
    )


def extract_and_validate_citations(
    raw_answer: str,
    retrieved_chunks: list[Chunk],
    *,
    query: str = "",
) -> CitedAnswer:
    """Extract inline [doc_XX_cYY] citations and validate against retrieved chunks.

    Args:
        raw_answer: Raw text response from LLM containing inline citation tags.
        retrieved_chunks: Context chunks that were provided to the LLM.
        query: Original user query string.

    Returns:
        CitedAnswer with verified citations mapped to source titles and URLs.
    """
    chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in retrieved_chunks}

    # Find all inline citation tags: e.g. [doc_14_c0], [doc_06_c1]
    cited_ids_found = re.findall(r"\[(doc_\d+_c\d+)\]", raw_answer)

    verified_citations: list[Citation] = []
    unverified_citations: list[str] = []
    seen_ids: set[str] = set()

    for cid in cited_ids_found:
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        if cid in chunk_map:
            chunk = chunk_map[cid]
            verified_citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_title=chunk.source_title,
                    source_url=chunk.source_url,
                )
            )
        else:
            unverified_citations.append(cid)

    # Convert inline [doc_14_c0] tags into numbered footnote superscripts [1], [2]
    # for clean readable presentation
    clean_text = raw_answer
    for idx, cite in enumerate(verified_citations, start=1):
        clean_text = clean_text.replace(f"[{cite.chunk_id}]", f"[{idx}]")

    return CitedAnswer(
        query=query,
        raw_answer=raw_answer,
        clean_answer=clean_text,
        citations=verified_citations,
        unverified_citations=unverified_citations,
    )
