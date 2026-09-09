"""Database schema, initialization, and read-only connection manager for Project C.

Creates and seeds a realistic client relational database with customers, orders,
products, appointments, and sensitive administrative logs.
"""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "c-database" / "ecommerce.db"


def init_database(db_path: Path = DEFAULT_DB_PATH) -> Path:
    """Initialize SQLite database with schema and sample transactional records."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Allowlisted client tables
    cursor.executescript(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL,
            city TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_quantity INTEGER NOT NULL
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')
            ),
            total_amount REAL NOT NULL,
            shipping_address TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            service_type TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN ('scheduled', 'confirmed', 'completed', 'cancelled')
            ),
            notes TEXT DEFAULT '',
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        -- Non-allowlisted table for testing security boundaries
        CREATE TABLE sensitive_admin_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_user TEXT NOT NULL,
            action TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # 2. Seed customers
    customers = [
        ("Alice Johnson", "alice@example.com", "+1-555-0192", "San Francisco"),
        ("Bob Smith", "bob@example.com", "+1-555-0143", "New York"),
        ("Charlie Brown", "charlie@example.com", "+1-555-0188", "Chicago"),
        ("Diana Prince", "diana@example.com", "+1-555-0111", "Seattle"),
    ]
    cursor.executemany(
        "INSERT INTO customers (name, email, phone, city) VALUES (?, ?, ?, ?)",
        customers,
    )

    # 3. Seed products
    products = [
        ("MacBook Pro 16", "Laptops", 2499.00, 15),
        ("Dell XPS 15", "Laptops", 1899.00, 20),
        ("Keychron K2 Keyboard", "Accessories", 89.00, 50),
        ("Logitech MX Master 3S", "Accessories", 99.00, 45),
        ("LG 34-inch UltraWide Monitor", "Monitors", 799.00, 10),
        ("Sony WH-1000XM5 Headphones", "Audio", 399.00, 30),
    ]
    cursor.executemany(
        "INSERT INTO products (name, category, price, stock_quantity) "
        "VALUES (?, ?, ?, ?)",
        products,
    )

    # 4. Seed orders
    orders = [
        (1, "2026-09-01 10:30:00", "delivered", 2598.00, "123 Market St, SF"),
        (1, "2026-09-07 14:15:00", "shipped", 89.00, "123 Market St, SF"),
        (2, "2026-09-03 09:00:00", "delivered", 1899.00, "456 Broadway, NY"),
        (2, "2026-09-08 16:45:00", "pending", 399.00, "456 Broadway, NY"),
        (3, "2026-09-05 11:20:00", "processing", 898.00, "789 Michigan Ave, Chicago"),
    ]
    cursor.executemany(
        "INSERT INTO orders (customer_id, order_date, status, total_amount, "
        "shipping_address) VALUES (?, ?, ?, ?, ?)",
        orders,
    )

    # 5. Seed order items
    order_items = [
        (1, 1, 1, 2499.00),  # Order 1: MacBook Pro
        (1, 4, 1, 99.00),  # Order 1: Logitech Mouse
        (2, 3, 1, 89.00),  # Order 2: Keychron Keyboard
        (3, 2, 1, 1899.00),  # Order 3: Dell XPS 15
        (4, 6, 1, 399.00),  # Order 4: Sony Headphones
        (5, 5, 1, 799.00),  # Order 5: Monitor
        (5, 4, 1, 99.00),  # Order 5: Mouse
    ]
    cursor.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) "
        "VALUES (?, ?, ?, ?)",
        order_items,
    )

    # 6. Seed appointments
    appointments = [
        (
            1,
            "Hardware Diagnostics",
            "2026-09-12 11:00:00",
            "confirmed",
            "Battery check on MacBook",
        ),
        (
            1,
            "Cloud Architecture Review",
            "2026-09-20 15:30:00",
            "scheduled",
            "Discuss AWS migration",
        ),
        (
            2,
            "Device Pickup",
            "2026-09-10 14:00:00",
            "confirmed",
            "Pick up repaired Dell XPS",
        ),
        (
            3,
            "Consultation",
            "2026-09-15 10:00:00",
            "scheduled",
            "Office workstation setup",
        ),
        (
            4,
            "VIP Onboarding",
            "2026-09-02 09:30:00",
            "completed",
            "New executive setup",
        ),
    ]
    cursor.executemany(
        "INSERT INTO appointments "
        "(customer_id, service_type, scheduled_time, status, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        appointments,
    )

    # 7. Seed sensitive admin logs (to prove security boundary enforcement)
    admin_logs = [
        ("superadmin", "CREDENTIAL_ROTATION", "10.0.0.1"),
        ("root", "SSH_LOGIN_SUCCESS", "192.168.1.50"),
    ]
    cursor.executemany(
        "INSERT INTO sensitive_admin_logs (admin_user, action, ip_address) "
        "VALUES (?, ?, ?)",
        admin_logs,
    )

    conn.commit()
    conn.close()
    return db_path


def get_readonly_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a strictly read-only SQLite connection.

    Enforces read-only at both the URI and engine pragma level:
    - file:path?mode=ro
    - PRAGMA query_only = ON
    """
    if not db_path.exists():
        init_database(db_path)

    # Convert path to absolute POSIX path for SQLite URI mode
    resolved_path = db_path.resolve().as_posix()
    uri = f"file:{resolved_path}?mode=ro"

    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    return conn


def get_tenant_readonly_connection(
    customer_id: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> sqlite3.Connection:
    """Create a tenant-scoped read-only SQLite connection.

    Creates SQLite temporary views that shadow tenant-partitioned tables
    ('customers', 'orders', 'appointments', 'order_items') to physically
    restrict the connection to rows belonging to customer_id, then enforces
    PRAGMA query_only = ON.
    """
    if not db_path.exists():
        init_database(db_path)

    resolved_path = db_path.resolve().as_posix()
    uri = f"file:{resolved_path}?mode=ro"

    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    safe_cid = int(customer_id)

    # Create temporary tenant isolation views
    conn.execute(
        f"CREATE TEMP VIEW customers AS "
        f"SELECT * FROM main.customers WHERE customer_id = {safe_cid};"
    )
    conn.execute(
        f"CREATE TEMP VIEW orders AS "
        f"SELECT * FROM main.orders WHERE customer_id = {safe_cid};"
    )
    conn.execute(
        f"CREATE TEMP VIEW appointments AS "
        f"SELECT * FROM main.appointments WHERE customer_id = {safe_cid};"
    )
    conn.execute(
        f"CREATE TEMP VIEW order_items AS "
        f"SELECT oi.* FROM main.order_items oi "
        f"JOIN main.orders o ON oi.order_id = o.order_id "
        f"WHERE o.customer_id = {safe_cid};"
    )

    conn.execute("PRAGMA query_only = ON;")
    return conn
