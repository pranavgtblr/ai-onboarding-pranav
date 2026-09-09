"""Documentation site crawler and content extractor with heading hierarchy chunking.

Task 3.11: Crawl documentation/content sites, strip navigation/footers/banners/sidebars,
and preserve heading hierarchy (e.g. h1 > h2 > h3) as chunk metadata.
"""

import argparse
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

    def save_to_disk(self, output_dir: Path) -> tuple[Path, Path]:
        """Save extracted pages as markdown and structured JSON chunks."""
        output_dir.mkdir(parents=True, exist_ok=True)
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        # Save individual clean markdown files
        for idx, p in enumerate(self.pages, start=1):
            safe_name = re.sub(r"[^\w\-]", "_", p.title.lower())[:50] or f"page_{idx}"
            md_path = pages_dir / f"{idx:02d}_{safe_name}.md"
            lines = [
                f"# {p.title}",
                f"Source: {p.url}",
                "",
                p.clean_text,
                "",
                "## Hierarchical Chunks",
                "",
            ]
            for c in p.chunks:
                lines.append(f"### Chunk: {c.heading_path}")
                lines.append(f"Tokens: {c.token_count} | Anchor: {c.section_anchor}")
                lines.append("")
                lines.append(c.text)
                lines.append("")
            md_path.write_text("\n".join(lines), encoding="utf-8")

        # Save structured JSON of all chunks
        all_chunks = [c.model_dump() for p in self.pages for c in p.chunks]
        chunks_json_path = output_dir / "chunks.json"
        chunks_json_path.write_text(
            json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Save crawl summary report
        summary_path = output_dir / "crawl_summary.md"
        summary_lines = [
            "# Website Crawl & Extraction Report",
            "",
            f"- **Start URL**: {self.start_url}",
            f"- **Pages Crawled**: {self.pages_crawled}",
            f"- **Total Chunks Extracted**: {self.total_chunks}",
            "",
            "## Crawled Pages",
            "",
            "| # | Page Title | Chunks | URL |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for idx, p in enumerate(self.pages, start=1):
            summary_lines.append(
                f"| {idx} | {p.title} | {len(p.chunks)} | [{p.url}]({p.url}) |"
            )
        summary_lines.append("")
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        return chunks_json_path, summary_path


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

        safe_prefix = re.sub(r"[^a-zA-Z0-9]", "_", page_title)[:20]
        chunk = HierarchicalChunk(
            chunk_id=f"{safe_prefix}_{chunk_counter:03d}",
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
    default_output = Path(__file__).resolve().parent
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to save extracted markdown and chunks (default: b-website)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for crawling and indexing website content."""
    args = parse_args()
    with DocumentationCrawler(
        args.url, max_pages=args.max_pages, max_depth=args.max_depth
    ) as crawler:
        report = crawler.crawl()
        chunks_json, summary = report.save_to_disk(args.output_dir)
        print(f"📁 Chunks saved to: {chunks_json}")
        print(f"📋 Summary saved to: {summary}")


if __name__ == "__main__":
    main()
