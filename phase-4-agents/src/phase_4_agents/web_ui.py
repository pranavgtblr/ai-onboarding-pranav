"""Web UI loader for Task 4.9: Progressive intermediate step streaming."""

from pathlib import Path

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"

INDEX_HTML = _INDEX_FILE.read_text(encoding="utf-8")
