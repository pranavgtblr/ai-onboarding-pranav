# Incremental Re-Crawl & Deduplication Proof (Task 3.12)

## Summary of Dual-Crawl Execution

| Metric | Crawl 1 (Initial) | Crawl 2 (Incremental Re-crawl) |
| :--- | :--- | :--- |
| **Pages Discovered** | 3 | 3 |
| **Pages Added** | 3 | 1 (https://docs.local/faq) |
| **Pages Updated** | 0 | 1 (https://docs.local/setup) |
| **Pages Unchanged** | 0 | 1 (https://docs.local) |
| **Pages Deleted** | 0 | 1 (https://docs.local/api) |
| **Chunks Retained** | 0 | 1 |
| **Chunks Updated** | 0 | 3 |
| **Chunks Deleted** | 0 | 1 |
| **Chunks Added** | 4 | 2 |
| **Total Active Chunks** | 4 | 6 |
| **Duplicate Chunks** | **0** | **0 (Verified)** |

## Audit Log & Assertions Verified

1. **Unchanged Pages (`/`)**: SHA-256 hash matched previous manifest entry. Chunks retained as-is with zero re-chunking overhead.
2. **Updated Pages (`/setup`)**: SHA-256 hash differed. Old chunks were completely excised and replaced with new chunks containing 'Docker Deployment'. No duplicate chunk IDs created.
3. **Deleted Pages (`/api`)**: URL absent from newly crawled pages. Old chunks were completely purged from `chunks.json`, and the corresponding markdown file in `pages/` was deleted.
4. **Added Pages (`/faq`)**: New URL detected. Chunks parsed with preserved heading hierarchies (`Billing`, `Support`) and appended to active index.
5. **Zero-Duplicate Invariant**: Checked `len(set(chunk_ids)) == len(chunks)`. Exactly 0 duplicate chunk IDs or overlapping text entries found.

## Manifest Snapshot After Crawl 2
```json
{
  "version": 1,
  "last_crawl_timestamp": "2026-09-09T06:07:49Z",
  "pages": {
    "https://docs.local": {
      "url": "https://docs.local",
      "title": "Acme Cloud Platform",
      "content_hash": "1ee1c38ec75e4f868a3b36944604dccaf70ba868a095d4b4635fcffd513858e0",
      "chunk_ids": [
        "Acme_Cloud_Platf_f2661a_001"
      ],
      "chunk_count": 1,
      "last_crawled_at": "2026-09-09T06:07:49Z",
      "markdown_filename": "01_acme_cloud_platform.md"
    },
    "https://docs.local/setup": {
      "url": "https://docs.local/setup",
      "title": "Setup Guide",
      "content_hash": "8cfd8be774133dddf6cf472caec1e24f113d9e850ccc070b64cbe17d3e6a9fb1",
      "chunk_ids": [
        "Setup_Guide_87cb6e_001",
        "Setup_Guide_87cb6e_002",
        "Setup_Guide_87cb6e_003"
      ],
      "chunk_count": 3,
      "last_crawled_at": "2026-09-09T06:07:49Z",
      "markdown_filename": "02_setup_guide.md"
    },
    "https://docs.local/faq": {
      "url": "https://docs.local/faq",
      "title": "Frequently Asked Questions",
      "content_hash": "8691cc36943d0c4deb1c4bf0635cdf0d603e08a7f47058b3fb600ee7cc7eb106",
      "chunk_ids": [
        "Frequently_Asked_64d311_001",
        "Frequently_Asked_64d311_002"
      ],
      "chunk_count": 2,
      "last_crawled_at": "2026-09-09T06:07:49Z",
      "markdown_filename": "03_frequently_asked_questions.md"
    }
  }
}
```