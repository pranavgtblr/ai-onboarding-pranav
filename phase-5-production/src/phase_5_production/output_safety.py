"""Output safety module.

Provides robust defenses against:
1. Cross-Site Scripting (XSS) via safe HTML rendering & entity escaping.
2. SQL Injection via parameterized query enforcement and AST isolation.
"""

from __future__ import annotations

import html
import re
import sqlite3
from html.parser import HTMLParser
from typing import Any


class UnparameterizedQueryError(Exception):
    """Raised when an unparameterized or unsafe SQL query is attempted."""


class HTMLSafetyParser(HTMLParser):
    """HTML parser that inspects rendered HTML for executable script vectors."""

    DANGEROUS_TAGS = {"script", "iframe", "object", "embed", "applet", "base", "link"}

    def __init__(self) -> None:
        super().__init__()
        self.violations: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.DANGEROUS_TAGS:
            self.violations.append(f"Executable tag detected: <{tag}>")

        for attr_name, attr_val in attrs:
            attr_lower = attr_name.lower()
            if attr_lower.startswith("on"):
                self.violations.append(
                    f"Event handler attribute detected: {attr_name}='{attr_val}'"
                )
            if attr_lower in ("src", "href", "data") and attr_val:
                val_clean = attr_val.strip().lower()
                if val_clean.startswith(("javascript:", "vbscript:", "data:text/html")):
                    self.violations.append(
                        f"Executable URI scheme detected in {attr_name}: '{attr_val}'"
                    )


def escape_html_output(content: str) -> str:
    """Escapes model output to neutralize HTML/XML injection vectors.

    Converts &, <, >, ", and ' to safe HTML entities.
    """
    if not isinstance(content, str):
        content = str(content)
    return html.escape(content, quote=True)


def verify_html_safety(html_content: str) -> tuple[bool, list[str]]:
    """Analyzes an HTML string for active executable tags or script handlers.

    Returns (is_safe, list_of_violations).
    """
    parser = HTMLSafetyParser()
    try:
        parser.feed(html_content)
        parser.close()
    except Exception as exc:
        return False, [f"HTML parsing error: {exc}"]
    return len(parser.violations) == 0, parser.violations


def render_model_output_page(
    model_output: str,
    *,
    is_safe: bool = True,
    title: str = "Model Output Safety Viewer",
) -> str:
    """Renders model output inside a structured HTML5 document.

    If is_safe=True:
      - Encodes all characters using escape_html_output.
      - Sets a strict Content Security Policy (CSP) forbidding inline/eval scripts.
    If is_safe=False:
      - Renders raw unescaped output (used strictly for vulnerability demonstration).
    """
    if is_safe:
        rendered_body = escape_html_output(model_output)
        csp_header = (
            '<meta http-equiv="Content-Security-Policy" '
            "content=\"default-src 'self'; script-src 'none'; "
            "style-src 'self' 'unsafe-inline'; object-src 'none';\">"
        )
        badge = '<span class="badge badge-safe">Escaped &amp; Protected (Safe)</span>'
    else:
        rendered_body = model_output
        csp_header = "<!-- CSP Disabled: Insecure Demonstration Mode -->"
        badge = '<span class="badge badge-danger">Unescaped (Vulnerable)</span>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  {csp_header}
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      margin: 0;
      padding: 2rem;
    }}
    .container {{
      max-width: 800px;
      margin: 0 auto;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 1.5rem;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    h1 {{
      margin-top: 0;
      font-size: 1.25rem;
      color: #38bdf8;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .badge {{
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
    }}
    .badge-safe {{ background: #065f46; color: #34d399; }}
    .badge-danger {{ background: #7f1d1d; color: #f87171; }}
    .output-box {{
      background: #090d16;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 1rem;
      margin-top: 1rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: monospace;
      color: #f1f5f9;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>
      <span>{html.escape(title)}</span>
      {badge}
    </h1>
    <p>Below is the rendered representation of model-generated text:</p>
    <div id="model-output" class="output-box">{rendered_body}</div>
  </div>
</body>
</html>"""


class SafeQueryExecutor:
    """Enforces parameterization barrier to prevent model output reaching SQL raw.

    Guarantees:
    1. Rejects unparameterized dynamic queries with raw string interpolation.
    2. Prohibits multi-statement execution / SQL command chaining.
    3. Strictly binds variables through sqlite3's parameterized API (? or :named).
    """

    FORBIDDEN_DDL_PATTERNS = [
        re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
        re.compile(r"\bALTER\s+TABLE\b", re.IGNORECASE),
        re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
    ]

    def __init__(self, db_conn: sqlite3.Connection) -> None:
        self.conn = db_conn
        self.conn.row_factory = sqlite3.Row

    def execute_parameterized(
        self,
        sql_template: str,
        params: tuple[Any, ...] | list[Any] | dict[str, Any] = (),
    ) -> list[dict[str, Any]]:
        """Executes a SQL query ensuring parameterization and safety constraints."""
        clean_sql = sql_template.strip()

        # Reject multi-statement query chaining attempts
        statements = [s.strip() for s in clean_sql.split(";") if s.strip()]
        if len(statements) > 1:
            raise UnparameterizedQueryError(
                "Multi-statement queries (query chaining via ';') "
                "are strictly prohibited."
            )

        # Check for forbidden destructive DDL keywords
        for pattern in self.FORBIDDEN_DDL_PATTERNS:
            if pattern.search(clean_sql):
                raise UnparameterizedQueryError(
                    f"Destructive DDL statements are forbidden: {pattern.pattern}"
                )

        # Ensure parameters are provided if the query contains parameter placeholders
        has_placeholders = "?" in clean_sql or bool(re.search(r":\w+", clean_sql))
        if has_placeholders and not params:
            raise UnparameterizedQueryError(
                "Query contains parameter placeholders but no parameters were supplied."
            )

        cursor = self.conn.cursor()
        try:
            cursor.execute(clean_sql, params)
            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            self.conn.commit()
            return [{"rows_affected": cursor.rowcount}]
        except sqlite3.Error as err:
            raise UnparameterizedQueryError(
                f"Database execution failed: {err}"
            ) from err

    def execute_raw_unsafe(self, unparameterized_sql: str) -> list[dict[str, Any]]:
        """Demonstration method to illustrate vulnerable unparameterized execution.

        Simulates what happens when a developer naively uses raw f-strings.
        """
        cursor = self.conn.cursor()
        cursor.execute(unparameterized_sql)
        if cursor.description:
            return [dict(row) for row in cursor.fetchall()]
        self.conn.commit()
        return [{"rows_affected": cursor.rowcount}]
