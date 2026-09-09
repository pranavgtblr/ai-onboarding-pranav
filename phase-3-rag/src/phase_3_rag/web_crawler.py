"""Documentation site crawler and content extractor with heading hierarchy chunking.

Task 3.11: Crawl documentation/content sites, strip navigation/footers/banners/sidebars,
and preserve heading hierarchy (e.g. h1 > h2 > h3) as chunk metadata.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import tiktoken
from bs4 import BeautifulSoup, Comment, Tag
from pydantic import BaseModel, Field

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Default CSS selectors / classes / IDs for boilerplate chrome
BOILERPLATE_TAGS = [
    "nav",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "form",
    "dialog",
    "select",
    "option",
    "button",
]

BOILERPLATE_PATTERNS = re.compile(
    r"(cookie|consent|gdpr|banner|sidebar|toc|table-of-contents|on-this-page|"
    r"navbar|nav-menu|site-nav|footer|site-footer|modal|popup|overlay|"
    r"alertdialog|social-share|share-buttons|feedback-widget|breadcrumb)",
    re.IGNORECASE,
)

EXCLUDED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".mp4",
    ".mp3",
    ".css",
    ".js",
}


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text content for change detection."""
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class HierarchicalChunk(BaseModel):
    """A semantic text chunk with preserved heading hierarchy metadata."""

    chunk_id: str = Field(description="Unique chunk identifier")
    url: str = Field(description="Canonical source URL")
    page_title: str = Field(description="Title of the source page")
    heading_hierarchy: list[str] = Field(
        default_factory=list,
        description="List of parent headings from top to bottom, e.g. ['Doc', 'Sec']",
    )
    heading_path: str = Field(
        description="Formatted hierarchy path, e.g. 'Doc > Section > Subsection'"
    )
    section_anchor: str = Field(
        default="", description="HTML section anchor ID if available, e.g. '#auth'"
    )
    text: str = Field(description="Clean text content of this chunk")
    token_count: int = Field(description="Number of tokens in the chunk")


class PageManifestEntry(BaseModel):
    """Metadata entry for a single crawled page stored in the manifest."""

    url: str = Field(description="Canonical URL of the page")
    title: str = Field(description="Page title")
    content_hash: str = Field(description="SHA-256 hash of extracted clean text")
    chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of chunks belonging to this page"
    )
    chunk_count: int = Field(default=0, description="Total active chunks for this page")
    last_crawled_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        description="ISO 8601 UTC timestamp of last crawl",
    )
    markdown_filename: str = Field(
        default="", description="Relative filename of the saved markdown page"
    )


class CrawlManifest(BaseModel):
    """Manifest tracking all indexed pages and their chunk mappings."""

    version: int = 1
    last_crawl_timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    pages: dict[str, PageManifestEntry] = Field(default_factory=dict)


class CrawlDiff(BaseModel):
    """Detailed diff statistics from an incremental crawl sync."""

    pages_added: list[str] = Field(default_factory=list)
    pages_updated: list[str] = Field(default_factory=list)
    pages_unchanged: list[str] = Field(default_factory=list)
    pages_deleted: list[str] = Field(default_factory=list)

    chunks_added: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    chunks_retained: int = 0
    total_active_chunks: int = 0

    @property
    def has_changes(self) -> bool:
        """Return True if any pages were added, updated, or deleted."""
        return bool(self.pages_added or self.pages_updated or self.pages_deleted)


class CrawlPage(BaseModel):
    """Scraped and cleaned page representation."""

    url: str
    title: str
    clean_text: str
    chunks: list[HierarchicalChunk]


class CrawlReport(BaseModel):
    """Overall report and export container for crawled site content."""

    start_url: str
    pages_crawled: int
    total_chunks: int
    pages: list[CrawlPage]
    last_diff: CrawlDiff | None = None

    def save_to_disk(
        self,
        output_dir: Path,
        incremental: bool = True,
        purge_deleted: bool = True,
    ) -> tuple[Path, Path]:
        """Save extracted pages as markdown and structured JSON chunks.

        When incremental=True, detects added, updated, unchanged, and deleted pages,
        guaranteeing zero duplicate chunks.
        """
        chunks_json_path, summary_path, _ = self.save_incremental(
            output_dir=output_dir,
            incremental=incremental,
            purge_deleted=purge_deleted,
        )
        return chunks_json_path, summary_path

    def save_incremental(
        self,
        output_dir: Path,
        incremental: bool = True,
        purge_deleted: bool = True,
    ) -> tuple[Path, Path, CrawlDiff]:
        """Save pages and incrementally sync chunks, returning the CrawlDiff."""
        output_dir.mkdir(parents=True, exist_ok=True)
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        manifest_path = output_dir / "crawl_manifest.json"
        chunks_json_path = output_dir / "chunks.json"
        summary_path = output_dir / "crawl_summary.md"

        # Load existing state if incremental and present
        manifest = CrawlManifest()
        existing_chunks: list[dict[str, Any]] = []

        if incremental and manifest_path.exists() and chunks_json_path.exists():
            try:
                manifest = CrawlManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                existing_chunks = json.loads(
                    chunks_json_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                print(f"⚠️ Warning loading existing manifest: {exc}; re-indexing.")
                manifest = CrawlManifest()
                existing_chunks = []

        diff = CrawlDiff()

        # Group existing chunks by page URL
        chunks_by_url: dict[str, list[dict[str, Any]]] = {}
        for c_dict in existing_chunks:
            c_url = c_dict.get("url", "")
            chunks_by_url.setdefault(c_url, []).append(c_dict)

        discovered_urls = {p.url for p in self.pages}

        # 1. Process crawled pages (Added, Updated, Unchanged)
        for idx, page in enumerate(self.pages, start=1):
            content_hash = compute_content_hash(page.clean_text)
            safe_name = (
                re.sub(r"[^\w\-]", "_", page.title.lower())[:50] or f"page_{idx}"
            )
            md_filename = f"{idx:02d}_{safe_name}.md"
            md_path = pages_dir / md_filename

            if page.url not in manifest.pages:
                # ADDED
                diff.pages_added.append(page.url)
                page_chunks_dicts = [c.model_dump() for c in page.chunks]
                chunks_by_url[page.url] = page_chunks_dicts
                diff.chunks_added += len(page_chunks_dicts)

                manifest.pages[page.url] = PageManifestEntry(
                    url=page.url,
                    title=page.title,
                    content_hash=content_hash,
                    chunk_ids=[c.chunk_id for c in page.chunks],
                    chunk_count=len(page.chunks),
                    markdown_filename=md_filename,
                )
                self._write_page_markdown(md_path, page)

            else:
                existing_entry = manifest.pages[page.url]
                if incremental and existing_entry.content_hash == content_hash:
                    # UNCHANGED
                    diff.pages_unchanged.append(page.url)
                    retained_count = len(chunks_by_url.get(page.url, []))
                    diff.chunks_retained += retained_count
                    existing_entry.last_crawled_at = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    if not (pages_dir / existing_entry.markdown_filename).exists():
                        self._write_page_markdown(md_path, page)
                        existing_entry.markdown_filename = md_filename
                else:
                    # UPDATED (changed content)
                    diff.pages_updated.append(page.url)
                    new_chunks_dicts = [c.model_dump() for c in page.chunks]
                    # Overwrite chunks for this URL completely - ZERO DUPLICATES
                    chunks_by_url[page.url] = new_chunks_dicts
                    diff.chunks_updated += len(new_chunks_dicts)

                    # Remove old markdown if name changed
                    if (
                        existing_entry.markdown_filename
                        and existing_entry.markdown_filename != md_filename
                    ):
                        old_path = pages_dir / existing_entry.markdown_filename
                        if old_path.exists():
                            old_path.unlink()

                    existing_entry.title = page.title
                    existing_entry.content_hash = content_hash
                    existing_entry.chunk_ids = [c.chunk_id for c in page.chunks]
                    existing_entry.chunk_count = len(new_chunks_dicts)
                    existing_entry.last_crawled_at = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    existing_entry.markdown_filename = md_filename
                    self._write_page_markdown(md_path, page)

        # 2. Process deletions (previously indexed pages no longer in discovered_urls)
        if purge_deleted:
            for prev_url in list(manifest.pages.keys()):
                if prev_url not in discovered_urls:
                    diff.pages_deleted.append(prev_url)
                    del_entry = manifest.pages.pop(prev_url)
                    removed_chunks = chunks_by_url.pop(prev_url, [])
                    diff.chunks_deleted += len(removed_chunks)
                    if del_entry.markdown_filename:
                        del_path = pages_dir / del_entry.markdown_filename
                        if del_path.exists():
                            del_path.unlink()

        # 3. Assemble all active chunks with strict deduplication check
        all_active_chunks: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for page_url, chunk_list in chunks_by_url.items():
            for c in chunk_list:
                cid = c["chunk_id"]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_active_chunks.append(c)

        diff.total_active_chunks = len(all_active_chunks)
        self.last_diff = diff

        # Save manifest
        manifest.last_crawl_timestamp = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        # Save chunks.json
        chunks_json_path.write_text(
            json.dumps(all_active_chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Save crawl summary markdown
        self._write_summary_markdown(summary_path, manifest, diff)

        return chunks_json_path, summary_path, diff

    def _write_page_markdown(self, path: Path, page: CrawlPage) -> None:
        """Write single extracted page to markdown."""
        lines = [
            f"# {page.title}",
            f"Source: {page.url}",
            "",
            page.clean_text,
            "",
            "## Hierarchical Chunks",
            "",
        ]
        for c in page.chunks:
            lines.append(f"### Chunk: {c.heading_path}")
            lines.append(f"Tokens: {c.token_count} | Anchor: {c.section_anchor}")
            lines.append("")
            lines.append(c.text)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_summary_markdown(
        self, path: Path, manifest: CrawlManifest, diff: CrawlDiff
    ) -> None:
        """Write crawl summary with incremental sync breakdown."""
        lines = [
            "# Website Crawl & Extraction Report",
            "",
            f"- **Start URL**: {self.start_url}",
            f"- **Pages Crawled (This Run)**: {self.pages_crawled}",
            f"- **Total Active Pages in Index**: {len(manifest.pages)}",
            f"- **Total Active Chunks**: {diff.total_active_chunks}",
            "",
            "## Incremental Synchronization Audit",
            "",
            f"- **Pages Added**: {len(diff.pages_added)}",
            f"- **Pages Updated**: {len(diff.pages_updated)}",
            f"- **Pages Unchanged**: {len(diff.pages_unchanged)}",
            f"- **Pages Deleted**: {len(diff.pages_deleted)}",
            f"- **Chunks Added**: {diff.chunks_added}",
            f"- **Chunks Updated**: {diff.chunks_updated}",
            f"- **Chunks Deleted**: {diff.chunks_deleted}",
            f"- **Chunks Retained (Unchanged)**: {diff.chunks_retained}",
            "- **Duplicates Guarantee**: 0 duplicate chunk IDs (verified)",
            "",
            "## Active Indexed Pages",
            "",
            "| # | Page Title | Chunks | Status (This Run) | URL |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for idx, (url, entry) in enumerate(manifest.pages.items(), start=1):
            if url in diff.pages_added:
                status = "🟢 Added"
            elif url in diff.pages_updated:
                status = "🟡 Updated"
            elif url in diff.pages_unchanged:
                status = "⚪ Unchanged"
            else:
                status = "Active"
            lines.append(
                f"| {idx} | {entry.title} | {entry.chunk_count} | {status} | "
                f"[{url}]({url}) |"
            )
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")


def count_tokens(text: str) -> int:
    """Accurately count tokens using cl100k_base tokenizer."""
    return len(TOKENIZER.encode(text))


def strip_boilerplate(soup: BeautifulSoup) -> None:
    """Aggressively strip navigation, footers, sidebars, cookie banners, and scripts."""
    # 1. Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 2. Decompose standard boilerplate elements by tag
    for tag_name in BOILERPLATE_TAGS:
        for element in list(soup.find_all(tag_name)):
            element.decompose()

    # 3. Decompose elements by ARIA roles, hidden attributes, and boilerplate patterns
    chrome_roles = {
        "navigation",
        "contentinfo",
        "banner",
        "dialog",
        "alertdialog",
    }
    for element in list(soup.find_all(True)):
        if not isinstance(element, Tag):
            continue
        if not hasattr(element, "attrs") or element.attrs is None:
            continue

        role_attr = element.attrs.get("role")
        if role_attr in chrome_roles:
            element.decompose()
            continue

        if element.attrs.get("aria-hidden") == "true" or "hidden" in element.attrs:
            element.decompose()
            continue

        raw_classes = element.attrs.get("class", [])
        if isinstance(raw_classes, list):
            classes = " ".join(str(c) for c in raw_classes)
        else:
            classes = str(raw_classes)
        element_id = str(element.attrs.get("id", ""))
        combined = f"{classes} {element_id}".strip()
        if combined and BOILERPLATE_PATTERNS.search(combined):
            # Guard: do not delete main content area even if named awkwardly
            tag_name = (element.name or "").lower()
            if tag_name in ("main", "article", "body", "html"):
                continue
            element.decompose()


def extract_page_title(soup: BeautifulSoup, default: str = "Untitled") -> str:
    """Extract page title prioritizing on-page <h1>, falling back to <title>."""
    h1_tag = soup.find("h1")
    if h1_tag and h1_tag.get_text(strip=True):
        return h1_tag.get_text(strip=True)

    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        # Strip common site suffixes like ' - Documentation' or ' | FastAPI'
        raw_title = title_tag.get_text(strip=True)
        cleaned = re.split(r"\s+[|\-—]\s+", raw_title)[0].strip()
        if cleaned:
            return cleaned

    return default


def _extract_text_content(tag: Tag) -> str:
    """Extract clean formatted text from a block tag (p, li, pre, table)."""
    if tag.name == "pre":
        return tag.get_text().strip()
    if tag.name == "table":
        # Render simple markdown table representation
        rows = []
        for tr in tag.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", cell.get_text(strip=True))
                for cell in tr.find_all(["th", "td"])
            ]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            if len(rows) > 1:
                col_count = len(rows[0].split("|")) - 2
                sep = "| " + " | ".join(["---"] * col_count) + " |"
                rows.insert(1, sep)
            return "\n".join(rows)

    text = tag.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def extract_hierarchical_chunks(
    html_content: str,
    url: str,
    page_title: str | None = None,
    max_chunk_tokens: int = 400,
    min_chunk_tokens: int = 15,
) -> tuple[str, list[HierarchicalChunk]]:
    """Clean HTML and produce chunks with preserved heading hierarchy.

    Returns a tuple of (clean_page_markdown, list_of_hierarchical_chunks).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    strip_boilerplate(soup)

    if not page_title:
        page_title = extract_page_title(soup)

    # Focus on main content container if present
    content_root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(class_=re.compile(r"(content|markdown-body|documentation)"))
        or soup.find("body")
        or soup
    )

    # Heading stack maintains active hierarchy: [(level, heading_text, anchor)]
    # Level 1 for h1, level 2 for h2, etc.
    heading_stack: list[tuple[int, str, str]] = [(1, page_title, "")]
    chunks: list[HierarchicalChunk] = []

    current_blocks: list[str] = []
    current_tokens = 0
    chunk_counter = 1

    def emit_current_chunk(force: bool = False) -> None:
        nonlocal current_blocks, current_tokens, chunk_counter
        combined_text = "\n\n".join(current_blocks).strip()
        if not combined_text:
            current_blocks = []
            current_tokens = 0
            return

        if not force and current_tokens < min_chunk_tokens:
            return

        hierarchy = [h[1] for h in heading_stack]
        path = " > ".join(hierarchy)
        active_anchor = heading_stack[-1][2] if heading_stack else ""

        url_slug = hashlib.sha256(url.encode()).hexdigest()[:6]
        safe_prefix = re.sub(r"[^a-zA-Z0-9]", "_", page_title).strip("_")[:16] or "doc"
        chunk = HierarchicalChunk(
            chunk_id=f"{safe_prefix}_{url_slug}_{chunk_counter:03d}",
            url=url,
            page_title=page_title,
            heading_hierarchy=hierarchy,
            heading_path=path,
            section_anchor=active_anchor,
            text=combined_text,
            token_count=current_tokens,
        )
        chunks.append(chunk)
        chunk_counter += 1
        current_blocks = []
        current_tokens = 0

    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    content_tags = {"p", "ul", "ol", "pre", "table", "blockquote"}

    # Process DOM elements in document order
    for element in content_root.descendants:
        if not isinstance(element, Tag):
            continue

        tag_name = element.name.lower()

        # When a heading is encountered
        if tag_name in heading_tags:
            h_level = int(tag_name[1])
            h_text = element.get_text(strip=True)
            if not h_text:
                continue

            # Check if this heading has an anchor ID
            raw_id = element.attrs.get("id", "") if element.attrs else ""
            anchor_id = str(raw_id) if isinstance(raw_id, str) else ""
            if not anchor_id:
                child_a = element.find("a", href=True)
                if child_a and isinstance(child_a, Tag) and child_a.attrs:
                    raw_href = child_a.attrs.get("href", "")
                    href = str(raw_href) if isinstance(raw_href, str) else ""
                    if href.startswith("#"):
                        anchor_id = href

            anchor = f"#{anchor_id.lstrip('#')}" if anchor_id else ""

            # Emit any text accumulated under previous heading before changing hierarchy
            emit_current_chunk(force=True)

            # Pop headings with level >= current heading level
            while heading_stack and heading_stack[-1][0] >= h_level:
                heading_stack.pop()

            # Append current heading to stack
            heading_stack.append((h_level, h_text, anchor))

        # When a content block is encountered
        elif tag_name in content_tags:
            # Avoid re-processing nested tags (e.g. li inside ul, td inside table)
            if element.parent and element.parent.name.lower() in content_tags:
                continue

            text_content = _extract_text_content(element)
            if not text_content:
                continue

            tok_count = count_tokens(text_content)

            # If adding this block exceeds max chunk tokens, emit current chunk first
            if current_tokens + tok_count > max_chunk_tokens and current_blocks:
                emit_current_chunk(force=True)

            current_blocks.append(text_content)
            current_tokens += tok_count

            if current_tokens >= max_chunk_tokens:
                emit_current_chunk(force=True)

    # Emit any remaining blocks
    emit_current_chunk(force=True)

    # Build full clean page text
    body_text = "\n\n".join(c.text for c in chunks)
    clean_text = f"# {page_title}\n\n{body_text}".strip()
    return clean_text, chunks


class DocumentationCrawler:
    """Polite, breadth-first documentation crawler with domain scoping."""

    def __init__(
        self,
        base_url: str,
        *,
        max_pages: int = 10,
        max_depth: int = 2,
        client: httpx.Client | None = None,
        delay_seconds: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.parsed_base = urlparse(self.base_url)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay_seconds = delay_seconds
        self._external_client = client is not None
        self.client = client or httpx.Client(
            headers={
                "User-Agent": "Toobler-Bot/1.0 (+https://toobler.com/ai-onboarding)"
            },
            timeout=15.0,
            follow_redirects=True,
        )

    def close(self) -> None:
        if not self._external_client:
            self.client.close()

    def __enter__(self) -> "DocumentationCrawler":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain and is an HTML documentation page."""
        parsed = urlparse(url)
        if parsed.netloc != self.parsed_base.netloc:
            return False

        # Exclude binary media assets
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
            return False

        return True

    def extract_links(self, html_content: str, current_url: str) -> list[str]:
        """Extract all valid same-domain internal links from HTML."""
        soup = BeautifulSoup(html_content, "html.parser")
        strip_boilerplate(soup)

        discovered: list[str] = []
        for a_tag in soup.find_all("a", href=True):
            if not isinstance(a_tag, Tag) or not a_tag.attrs:
                continue
            raw_href = a_tag.attrs.get("href", "")
            if isinstance(raw_href, list):
                href_str = str(raw_href[0]) if raw_href else ""
            else:
                href_str = str(raw_href)
            if not href_str:
                continue
            abs_url = urljoin(current_url, href_str)
            clean_url, _ = urldefrag(abs_url)
            clean_url = clean_url.rstrip("/")
            if self.is_valid_url(clean_url):
                discovered.append(clean_url)
        return discovered

    def crawl(self) -> CrawlReport:
        """Execute BFS crawl over documentation pages starting from base_url."""
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(self.base_url, 0)]
        pages: list[CrawlPage] = []
        total_chunks = 0

        print(
            f"🌐 [Crawler] Starting crawl on {self.base_url} "
            f"(max_pages={self.max_pages}, max_depth={self.max_depth})..."
        )

        while queue and len(pages) < self.max_pages:
            current_url, depth = queue.pop(0)
            if current_url in visited or depth > self.max_depth:
                continue

            visited.add(current_url)
            print(
                f"  [{len(pages) + 1:02d}/{self.max_pages:02d}] "
                f"Crawling depth={depth}: {current_url}...",
                flush=True,
            )

            try:
                resp = self.client.get(current_url)
                if resp.status_code != 200:
                    print(f"    ⚠️ HTTP {resp.status_code} for {current_url}")
                    continue

                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type:
                    continue

                html = resp.text
                page_title = extract_page_title(
                    BeautifulSoup(html, "html.parser"), default=current_url
                )
                clean_text, chunks = extract_hierarchical_chunks(
                    html, current_url, page_title=page_title
                )

                pages.append(
                    CrawlPage(
                        url=current_url,
                        title=page_title,
                        clean_text=clean_text,
                        chunks=chunks,
                    )
                )
                total_chunks += len(chunks)
                print(
                    f"    ✅ '{page_title}' -> {len(chunks)} chunks extracted.",
                    flush=True,
                )

                # Queue next level links if depth permits
                if depth < self.max_depth:
                    new_links = self.extract_links(html, current_url)
                    for link in new_links:
                        if link not in visited:
                            queue.append((link, depth + 1))

                if self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)

            except Exception as exc:
                print(f"    ❌ Error crawling {current_url}: {exc}")

        print(
            f"🎉 [Crawler] Finished. Crawled {len(pages)} pages, "
            f"generated {total_chunks} hierarchical chunks.\n"
        )
        return CrawlReport(
            start_url=self.base_url,
            pages_crawled=len(pages),
            total_chunks=total_chunks,
            pages=pages,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for website crawler."""
    parser = argparse.ArgumentParser(
        description=(
            "Crawl documentation sites and extract chunks with heading hierarchy."
        )
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Base URL of documentation site to crawl (e.g. https://httpbin.org/html)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of pages to crawl (default: 5)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum link crawl depth (default: 2)",
    )
    default_output = Path(__file__).resolve().parents[2] / "b-website"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to save extracted markdown and chunks (default: b-website)",
    )
    parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable incremental re-crawl and change detection (default: True)",
    )
    parser.add_argument(
        "--purge-deleted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Purge chunks of pages no longer discovered during crawl (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for crawling and indexing website content."""
    args = parse_args()
    with DocumentationCrawler(
        args.url, max_pages=args.max_pages, max_depth=args.max_depth
    ) as crawler:
        report = crawler.crawl()
        chunks_json, summary, diff = report.save_incremental(
            args.output_dir,
            incremental=args.incremental,
            purge_deleted=args.purge_deleted,
        )
        print(f"📁 Chunks saved to: {chunks_json}")
        print(f"📋 Summary saved to: {summary}")
        print(
            f"🔄 Incremental Sync: {len(diff.pages_added)} added, "
            f"{len(diff.pages_updated)} updated, "
            f"{len(diff.pages_unchanged)} unchanged, "
            f"{len(diff.pages_deleted)} deleted."
        )
        print(f"✨ Total active chunks: {diff.total_active_chunks} (0 duplicates)")


if __name__ == "__main__":
    main()
