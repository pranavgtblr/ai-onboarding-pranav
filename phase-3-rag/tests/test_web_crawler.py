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


def test_compute_content_hash_normalization():
    """Verify SHA-256 content hashing normalizes whitespace and detects edits."""
    from phase_3_rag.web_crawler import compute_content_hash

    text1 = "Heading\n\nParagraph text with spaces."
    text2 = "Heading   Paragraph  text  with  spaces.   "
    text3 = "Heading\n\nParagraph text with EDITED words."

    assert compute_content_hash(text1) == compute_content_hash(text2)
    assert compute_content_hash(text1) != compute_content_hash(text3)


def test_incremental_recrawl_detects_changes_and_purges_without_duplicates(
    tmp_path: Path,
):
    """Prove that running the crawl twice detects unchanged, updated,
    and deleted pages without duplicate chunks.
    """
    site_db = {
        "https://site.test": """
            <html><head><title>Home</title></head><body>
            <main>
                <h1>Home Page</h1>
                <p>Welcome to the platform.</p>
                <a href="/guide">Guide</a>
                <a href="/legacy">Legacy</a>
            </main></body></html>
        """,
        "https://site.test/guide": """
            <html><head><title>Guide</title></head><body>
            <main>
                <h1>User Guide</h1>
                <h2>Installation</h2>
                <p>Run pip install sample-app</p>
            </main></body></html>
        """,
        "https://site.test/legacy": """
            <html><head><title>Legacy</title></head><body>
            <main>
                <h1>Legacy Section</h1>
                <p>Old deprecated endpoints.</p>
            </main></body></html>
        """,
    }

    def transport(request: httpx.Request) -> httpx.Response:
        url = str(request.url).rstrip("/")
        if url in site_db:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text=site_db[url],
            )
        return httpx.Response(404, text="Not Found")

    # ==========================================
    # CRAWL 1: Initial Crawl (3 pages added)
    # ==========================================
    client_1 = httpx.Client(
        transport=httpx.MockTransport(transport),
        base_url="https://site.test",
    )
    crawler_1 = DocumentationCrawler(
        "https://site.test", max_pages=5, client=client_1, delay_seconds=0.0
    )
    report_1 = crawler_1.crawl()
    chunks_path, summary_path, diff_1 = report_1.save_incremental(
        tmp_path, incremental=True, purge_deleted=True
    )

    assert len(diff_1.pages_added) == 3
    assert len(diff_1.pages_updated) == 0
    assert len(diff_1.pages_deleted) == 0
    assert diff_1.total_active_chunks > 0

    with open(chunks_path, encoding="utf-8") as f:
        chunks_v1 = json.load(f)
    v1_chunk_ids = [c["chunk_id"] for c in chunks_v1]
    assert len(v1_chunk_ids) == len(set(v1_chunk_ids)), "Duplicates found in Crawl 1!"

    manifest_file = tmp_path / "crawl_manifest.json"
    assert manifest_file.exists()

    # ==========================================
    # EVOLUTION:
    # 1. / (Home) is UNCHANGED
    # 2. /guide is UPDATED with new section
    # 3. /legacy is DELETED (removed from site)
    # 4. /faq is ADDED
    # ==========================================
    site_db["https://site.test"] = """
        <html><head><title>Home</title></head><body>
        <main>
            <h1>Home Page</h1>
            <p>Welcome to the platform.</p>
            <a href="/guide">Guide</a>
            <a href="/faq">FAQ</a>
        </main></body></html>
    """
    site_db["https://site.test/guide"] = """
        <html><head><title>Guide</title></head><body>
        <main>
            <h1>User Guide</h1>
            <h2>Installation</h2>
            <p>Run pip install sample-app</p>
            <h2>Configuration</h2>
            <p>Export APP_ENV=production to configure settings.</p>
        </main></body></html>
    """
    del site_db["https://site.test/legacy"]
    site_db["https://site.test/faq"] = """
        <html><head><title>FAQ</title></head><body>
        <main>
            <h1>FAQ</h1>
            <h2>Questions</h2>
            <p>Frequently asked questions about billing and support.</p>
        </main></body></html>
    """

    # ==========================================
    # CRAWL 2: Incremental Re-Crawl
    # ==========================================
    client_2 = httpx.Client(
        transport=httpx.MockTransport(transport),
        base_url="https://site.test",
    )
    crawler_2 = DocumentationCrawler(
        "https://site.test", max_pages=5, client=client_2, delay_seconds=0.0
    )
    report_2 = crawler_2.crawl()
    chunks_path_2, summary_path_2, diff_2 = report_2.save_incremental(
        tmp_path, incremental=True, purge_deleted=True
    )

    # Verify classification of changes
    assert "https://site.test" in diff_2.pages_unchanged
    assert "https://site.test/guide" in diff_2.pages_updated
    assert "https://site.test/legacy" in diff_2.pages_deleted
    assert "https://site.test/faq" in diff_2.pages_added

    assert len(diff_2.pages_added) == 1
    assert len(diff_2.pages_updated) == 1
    assert len(diff_2.pages_unchanged) == 1
    assert len(diff_2.pages_deleted) == 1

    with open(chunks_path_2, encoding="utf-8") as f:
        chunks_v2 = json.load(f)

    # STRICT ASSERTIONS: Zero Duplicates & Clean Deletions
    v2_chunk_ids = [c["chunk_id"] for c in chunks_v2]
    assert len(v2_chunk_ids) == len(set(v2_chunk_ids)), (
        f"Duplicates found: {len(v2_chunk_ids)} vs {len(set(v2_chunk_ids))}"
    )

    # 1. Verify deleted page chunks were removed
    legacy_chunks = [c for c in chunks_v2 if "https://site.test/legacy" in c["url"]]
    assert len(legacy_chunks) == 0, (
        "Stale chunks from deleted page exist in chunks.json!"
    )

    # 2. Verify updated page has replaced chunks with new 'Configuration' section
    guide_chunks = [c for c in chunks_v2 if "https://site.test/guide" in c["url"]]
    assert any("Configuration" in c["heading_path"] for c in guide_chunks)

    # 3. Verify new FAQ page chunks exist
    faq_chunks = [c for c in chunks_v2 if "https://site.test/faq" in c["url"]]
    assert len(faq_chunks) > 0


def test_unauthenticated_crawl_blocks_protected_page():
    """Verify that unauthenticated crawl receives 401 and skips protected content."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).rstrip("/")
        if url == "https://secure.test":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text=(
                    "<html><body><main><h1>Public</h1>"
                    "<a href='/private'>Private</a></main></body></html>"
                ),
            )
        if url == "https://secure.test/private":
            return httpx.Response(401, text="Unauthorized")
        return httpx.Response(404, text="Not Found")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://secure.test",
    )
    crawler = DocumentationCrawler(
        "https://secure.test", max_pages=5, client=client, delay_seconds=0.0
    )
    report = crawler.crawl()

    crawled_urls = [p.url for p in report.pages]
    assert "https://secure.test" in crawled_urls
    assert "https://secure.test/private" not in crawled_urls
    assert all("Private" not in c.text for p in report.pages for c in p.chunks)


def test_authenticated_crawl_via_login_session_cookie():
    """Verify crawler logs in via POST, captures cookie, and crawls pages."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).rstrip("/")
        cookie = request.headers.get("cookie", "")

        if url == "https://secure.test/login" and request.method == "POST":
            body = request.content.decode("utf-8")
            if "username=admin" in body and "password=secret" in body:
                return httpx.Response(
                    302,
                    headers={
                        "Location": "https://secure.test/vault",
                        "Set-Cookie": "session_id=vault_auth_token_99; Path=/",
                    },
                    text="Redirecting",
                )
            return httpx.Response(401, text="Bad credentials")

        if url == "https://secure.test":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text=(
                    "<html><body><main><h1>Home</h1>"
                    "<a href='/vault'>Vault</a></main></body></html>"
                ),
            )

        if url == "https://secure.test/vault":
            if "session_id=vault_auth_token_99" in cookie:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    text=(
                        "<html><body><main><h1>Vault Secrets</h1>"
                        "<p>Top secret architecture.</p></main></body></html>"
                    ),
                )
            return httpx.Response(401, text="Unauthorized")

        return httpx.Response(404, text="Not Found")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://secure.test",
        follow_redirects=True,
    )
    crawler = DocumentationCrawler(
        "https://secure.test",
        max_pages=5,
        client=client,
        delay_seconds=0.0,
        login_url="https://secure.test/login",
        login_data={"username": "admin", "password": "secret"},
    )
    report = crawler.crawl()

    crawled_urls = [p.url for p in report.pages]
    assert "https://secure.test/vault" in crawled_urls
    assert any(
        "Top secret architecture" in c.text for p in report.pages for c in p.chunks
    )


def test_authenticated_crawl_via_cookie_and_header_injection():
    """Verify direct injection of cookies and auth headers for protected access."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).rstrip("/")
        cookie = request.headers.get("cookie", "")
        auth_hdr = request.headers.get("authorization", "")

        if url == "https://api.test/docs":
            if "auth_token=jwt_valid_123" in cookie or auth_hdr == "Bearer secret_jwt":
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    text=(
                        "<html><body><main><h1>VIP Docs</h1>"
                        "<p>Subscriber only content.</p></main></body></html>"
                    ),
                )
            return httpx.Response(403, text="Forbidden")

        return httpx.Response(404, text="Not Found")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.test",
    )
    crawler = DocumentationCrawler(
        "https://api.test/docs",
        max_pages=2,
        client=client,
        delay_seconds=0.0,
        cookies={"auth_token": "jwt_valid_123"},
        auth_headers={"Authorization": "Bearer secret_jwt"},
    )
    report = crawler.crawl()

    assert len(report.pages) == 1
    assert "Subscriber only content" in report.pages[0].clean_text
