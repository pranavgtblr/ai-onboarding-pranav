"""User Authentication & JWT Security Module.

Features salted PBKDF2 password hashing and type-safe HMAC-SHA256 JWT tokens
using Python standard library (no external C-extensions required).
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import timedelta
from typing import Any

from fastapi import Header
from sqlalchemy import select

from phase_6_capstone.config import settings
from phase_6_capstone.db import DatabaseManager, UserModel

JWT_SECRET_KEY = getattr(
    settings, "jwt_secret_key", "pg_recommends_production_jwt_secret_2026"
)
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24 * 7  # 7 days


def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a random cryptographic salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return f"{salt}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a plaintext password against a stored salt$hash string."""
    if not stored_hash or "$" not in stored_hash:
        return False
    try:
        salt, expected_hash = stored_hash.split("$", 1)
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100_000,
        )
        return hmac.compare_digest(key.hex(), expected_hash)
    except Exception:
        return False


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def create_access_token(
    user_id: str,
    email: str,
    tenant_id: str = "default_tenant",
    letterboxd_handle: str | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Creates a signed JSON Web Token (JWT) using HMAC-SHA256."""
    now = int(time.time())
    if expires_delta:
        expire = now + int(expires_delta.total_seconds())
    else:
        expire = now + (TOKEN_EXPIRY_HOURS * 3600)

    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": user_id,
        "email": email,
        "tenant_id": tenant_id,
        "letterboxd_handle": letterboxd_handle,
        "iat": now,
        "exp": expire,
    }

    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header_b64 = _b64url_encode(hdr_bytes)
    payload_b64 = _b64url_encode(payload_bytes)

    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(
        JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decodes and cryptographically verifies a JWT token. Returns payload or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(
            JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()

        # Constant time signature comparison
        if not hmac.compare_digest(_b64url_encode(expected_sig), sig_b64):
            return None

        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Verify expiration
        exp = payload.get("exp")
        if exp and int(time.time()) > int(exp):
            return None

        return payload
    except Exception:
        return None


async def get_optional_user(
    authorization: str | None = Header(default=None),
    db: DatabaseManager | None = None,
) -> UserModel | None:
    """FastAPI dependency: Returns UserModel if valid Bearer token, else None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.replace("Bearer ", "").strip()
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None

    user_id = payload["sub"]
    if not db:
        db = DatabaseManager(settings.database_url)

    async for session in db.get_session():
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    return None
