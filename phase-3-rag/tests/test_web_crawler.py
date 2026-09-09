"""Unit and integration tests for web crawler and hierarchical chunker (Task 3.11)."""

import json
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from phase_3_rag.web_crawler import (
    DocumentationCrawler,
    HierarchicalChunk,
    extract_hierarchical_chunks,
    extract_page_title,
    strip_boilerplate,
)

SAMPLE_DOC_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>Apollo Platform Documentation | Odyssey</title>
</head>
<body>
    <header class="navbar header-nav">
        <a href="/">Home</a>
        <a href="/docs">Docs</a>
        <button class="theme-toggle">Dark Mode</button>
    </header>

    <div class="cookie-banner" role="alertdialog">
        <p>We use cookies to improve your experience. <button>Accept</button></p>
    </div>

    <div class="sidebar doc-sidebar">
        <nav class="toc">
            <ul>
                <li><a href="#quickstart">Quickstart</a></li>
                <li><a href="#architecture">Architecture</a></li>
            </ul>
        </nav>
    </div>

    <main class="content markdown-body">
        <h1>Apollo Mission Architecture</h1>
        <p>Welcome to the Apollo Mission architecture documentation guide.</p>

        <h2 id="core-principles">Core Design Principles</h2>
        <p>The system is built on decentralized telemetry.</p>
        <p>Every habitat node operates autonomously during orbital blackout.</p>

        <h3 id="life-support-subsystem">Life Support Telemetry</h3>
        <p>Atmospheric monitoring runs at 10 Hz frequency.</p>
        <pre><code>def check_o2_levels():
    return current_o2_kpa >= 21.0</code></pre>

        <h2 id="power-distribution">Power Distribution Grid</h2>
        <p>Solar arrays and fission surface power feed a unified 120V DC bus.</p>
        <table>
            <tr><th>Bus Name</th><th>Voltage</th><th>Capacity</th></tr>
            <tr><td>Main Bus A</td><td>120V</td><td>45 kW</td></tr>
            <tr><td>Avionics Bus</td><td>28V</td><td>10 kW</td></tr>
        </table>
    </main>

    <aside class="sidebar-secondary">
        <p>Related documents and links.</p>
    </aside>

    <footer class="site-footer">
        <p>Copyright 2084 Project Odyssey Mars Operations. All rights reserved.</p>
    </footer>
</body>
</html>
"""


def test_strip_boilerplate_removes_chrome():
    """Verify navigation, footers, sidebars, cookie banners, and scripts are removed."""
    soup = BeautifulSoup(SAMPLE_DOC_HTML, "html.parser")
    strip_boilerplate(soup)

    # Check tags stripped
    assert soup.find("nav") is None
    assert soup.find("footer") is None
    assert soup.find("aside") is None
    assert soup.find("button") is None

    # Check class/id boilerplate stripped
    assert soup.find(class_="cookie-banner") is None
    assert soup.find(class_="navbar") is None
    assert soup.find(class_="sidebar") is None
    assert soup.find(class_="site-footer") is None

    # Check main content remains intact
    assert soup.find("main") is not None
    assert soup.find("h1") is not None
    assert "Apollo Mission Architecture" in soup.get_text()
    assert "Copyright 2084" not in soup.get_text()
    assert "We use cookies" not in soup.get_text()


def test_extract_page_title_strips_suffixes():
    """Verify title extractor grabs clean title and strips site branding suffixes."""
    soup = BeautifulSoup(SAMPLE_DOC_HTML, "html.parser")
    title = extract_page_title(soup)
    assert title == "Apollo Mission Architecture"

    # Fallback to <title> when H1 is absent (with branding suffix stripped)
    title_only_html = (
        "<html><head><title>Odyssey Fleet Manual | SpaceOps</title></head>"
        "<body><p>No header text here</p></body></html>"
    )
    title_soup = BeautifulSoup(title_only_html, "html.parser")
    assert extract_page_title(title_soup) == "Odyssey Fleet Manual"


def test_extract_hierarchical_chunks_preserves_heading_tree():
    """Verify hierarchical chunker accurately tracks nested heading stack."""
    clean_text, chunks = extract_hierarchical_chunks(
        SAMPLE_DOC_HTML,
        url="https://docs.odyssey.space/apollo",
        max_chunk_tokens=300,
    )

    assert len(chunks) >= 3
    assert "Apollo Mission Architecture" in clean_text

    # Chunk 1: Intro under H1
    c1 = chunks[0]
    assert c1.heading_hierarchy == ["Apollo Mission Architecture"]
    assert "Welcome to the Apollo Mission" in c1.text

    # Chunk 2: Core Design Principles (H1 > H2)
    c2 = chunks[1]
    assert c2.heading_hierarchy == [
        "Apollo Mission Architecture",
        "Core Design Principles",
    ]
    assert c2.heading_path == "Apollo Mission Architecture > Core Design Principles"
    assert c2.section_anchor == "#core-principles"
    assert "decentralized telemetry" in c2.text

    # Chunk 3: Life Support Telemetry (H1 > H2 > H3)
    c3 = chunks[2]
    assert c3.heading_hierarchy == [
        "Apollo Mission Architecture",
        "Core Design Principles",
        "Life Support Telemetry",
    ]
    expected_path = (
        "Apollo Mission Architecture > Core Design Principles > Life Support Telemetry"
    )
    assert c3.heading_path == expected_path
    assert c3.section_anchor == "#life-support-subsystem"
    assert "10 Hz frequency" in c3.text
    assert "def check_o2_levels():" in c3.text

    # Chunk 4: Power Distribution Grid (H1 > H2 - popped H3 and previous H2)
    c4 = chunks[3]
    assert c4.heading_hierarchy == [
        "Apollo Mission Architecture",
        "Power Distribution Grid",
    ]
    assert c4.heading_path == "Apollo Mission Architecture > Power Distribution Grid"
    assert c4.section_anchor == "#power-distribution"
    assert "Main Bus A" in c4.text
    assert "120V" in c4.text


def test_chunk_token_count_populated():
    """Verify each hierarchical chunk contains an accurate token count."""
    _, chunks = extract_hierarchical_chunks(SAMPLE_DOC_HTML, url="https://example.com")
    for c in chunks:
        assert isinstance(c, HierarchicalChunk)
        assert c.token_count > 0
        assert c.url == "https://example.com"
        assert c.page_title == "Apollo Mission Architecture"


def test_crawler_link_validation_and_exclusion():
    """Verify crawler validates same-domain links and excludes static media files."""
    crawler = DocumentationCrawler("https://docs.odyssey.space")

    # Valid internal links
    assert crawler.is_valid_url("https://docs.odyssey.space/quickstart")
    assert crawler.is_valid_url("https://docs.odyssey.space/api/v1/auth")

    # External links excluded
    assert not crawler.is_valid_url("https://external-api.com/status")
    assert not crawler.is_valid_url("https://github.com/odyssey/repo")

    # Static media assets excluded
    assert not crawler.is_valid_url("https://docs.odyssey.space/logo.png")
    assert not crawler.is_valid_url("https://docs.odyssey.space/manual.pdf")
    assert not crawler.is_valid_url("https://docs.odyssey.space/app.js")
    assert not crawler.is_valid_url("https://docs.odyssey.space/styles.css")


def test_mock_documentation_crawler_traversal(tmp_path: Path):
    """Test full BFS crawl, extraction, and report saving over mock multi-page site."""
    mock_site = {
        "https://docs.test.local": """
            <html><head><title>Test Docs Home</title></head><body>
            <nav><a href="/page1">P1</a></nav>
            <main>
                <h1>Welcome to Test Docs</h1>
                <p>Root page overview.</p>
                <a href="/guide">User Guide</a>
                <a href="/api">API Reference</a>
                <a href="https://external.com/blocked">Blocked External</a>
            </main>
            </body></html>
        """,
        "https://docs.test.local/guide": """
            <html><head><title>User Guide</title></head><body>
            <main>
                <h1>User Guide</h1>
                <h2>Setup</h2>
                <p>Installation guide details.</p>
                <a href="/guide/advanced">Advanced</a>
            </main>
            </body></html>
        """,
        "https://docs.test.local/api": """
            <html><head><title>API Reference</title></head><body>
            <main>
                <h1>API Reference</h1>
                <p>Public REST endpoints documentation.</p>
            </main>
            </body></html>
        """,
        "https://docs.test.local/guide/advanced": """
            <html><head><title>Advanced Guide</title></head><body>
            <main>
                <h1>Advanced Guide</h1>
                <p>Deep dive into architecture.</p>
            </main>
            </body></html>
        """,
    }

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url).rstrip("/")
        if url_str in mock_site:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text=mock_site[url_str],
            )
        return httpx.Response(404, text="Not Found")

    client = httpx.Client(
        transport=httpx.MockTransport(mock_transport_handler),
        base_url="https://docs.test.local",
    )

    crawler = DocumentationCrawler(
        "https://docs.test.local",
        max_pages=3,
        max_depth=2,
        client=client,
        delay_seconds=0.0,
    )

    report = crawler.crawl()
    assert report.pages_crawled == 3
    assert report.total_chunks >= 3

    # Verify disk export
    chunks_path, summary_path = report.save_to_disk(tmp_path)
    assert chunks_path.exists()
    assert summary_path.exists()

    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)
    assert len(chunks_data) == report.total_chunks
    assert "heading_hierarchy" in chunks_data[0]
    assert "heading_path" in chunks_data[0]
