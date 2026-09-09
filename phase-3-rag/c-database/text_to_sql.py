"""Secure Text-to-SQL Runner (Project C - Database RAG).

Enforces:
1. Pure SELECT statements (blocks INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA).
2. Allowlisted tables only (customers, orders, order_items, products, appointments).
3. Mandatory LIMIT enforcement (injects or clamps to MAX_LIMIT).
4. Strict read-only database credentials (PRAGMA query_only = ON, mode=ro).
5. AST validation via sqlglot prior to execution (never string interpolation).
"""

import sys
from pathlib import Path

# Add phase-3-rag/src to path for standalone execution
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from phase_3_rag.database import (  # noqa: E402, F401
    DEFAULT_DB_PATH,
    get_readonly_connection,
    init_database,
)
from phase_3_rag.text_to_sql import (  # noqa: E402
    ALLOWLISTED_TABLES,  # noqa: F401
    DEFAULT_MAX_LIMIT,  # noqa: F401
    SecurityValidationError,  # noqa: F401
    TextToSqlResult,  # noqa: F401
    execute_readonly_sql,  # noqa: F401
    main,
    run_text_to_sql_pipeline,  # noqa: F401
    validate_and_sanitize_sql,  # noqa: F401
)

if __name__ == "__main__":
    main()
