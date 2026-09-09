"""Documentation site crawler with incremental re-crawl and change detection.

Task 3.12: Incremental re-crawl: detect changed pages and update or delete
their chunks without creating duplicates.
"""

import sys
from pathlib import Path

# Add phase-3-rag/src to path for standalone script execution
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from phase_3_rag.web_crawler import (  # noqa: F401, E402
    CrawlDiff,
    CrawlManifest,
    CrawlPage,
    CrawlReport,
    DocumentationCrawler,
    HierarchicalChunk,
    PageManifestEntry,
    compute_content_hash,
    count_tokens,
    extract_hierarchical_chunks,
    extract_page_title,
    main,
    parse_args,
    strip_boilerplate,
)

if __name__ == "__main__":
    main()
