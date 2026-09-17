"""Tests for User Authentication, JWT Tokens, and Profile Match in PG Recommends."""

import pytest
from fastapi.testclient import TestClient

from phase_6_capstone.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from phase_6_capstone.db import DatabaseManager
from phase_6_capstone.server import create_app


@pytest.fixture
def auth_client():
    db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    app = create_app(db=db, initial_catalog=[])
    with TestClient(app) as client:
        yield client, db


def test_password_hashing_and_verification():
    """Test salted PBKDF2 password hashing and constant-time verification."""
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)
    assert hashed != password
    assert "$" in hashed

    # Valid password verification
    assert verify_password(password, hashed) is True

    # Invalid password verification
    assert verify_password("WrongPassword", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(password, "invalid_hash_string") is False


def test_jwt_token_lifecycle():
    """Test HMAC-SHA256 JWT creation, payload claims, and signature verification."""
    token = create_access_token(
        user_id="user_cinephile1",
        email="cinephile@example.com",
        tenant_id="tenant_cinema",
        letterboxd_handle="cinephile_99",
    )
    assert isinstance(token, str)
    assert len(token.split(".")) == 3

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "user_cinephile1"
    assert payload["email"] == "cinephile@example.com"
    assert payload["tenant_id"] == "tenant_cinema"
    assert payload["letterboxd_handle"] == "cinephile_99"

    # Tampered token verification fails
    tampered = token[:-4] + "abcd"
    assert decode_access_token(tampered) is None


def test_register_and_login_flow(auth_client):
    """Test user registration, duplicate email rejection, and successful login."""
    client, _ = auth_client

    # 1. Successful registration
    reg_resp = client.post(
        "/api/auth/register",
        json={
            "email": "david@fincherfans.com",
            "password": "MindhunterSeason3When?",
            "letterboxd_handle": "davidfincher",
        },
    )
    assert reg_resp.status_code == 200
    data = reg_resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == "david@fincherfans.com"
    assert data["user"]["letterboxd_handle"] == "davidfincher"
    assert data["user"]["taste_match_pct"] > 0
    token = data["access_token"]

    # 2. Duplicate registration fails
    dup_resp = client.post(
        "/api/auth/register",
        json={
            "email": "david@fincherfans.com",
            "password": "AnotherPassword123",
        },
    )
    assert dup_resp.status_code == 400

    # 3. Short password fails
    short_resp = client.post(
        "/api/auth/register",
        json={
            "email": "short@example.com",
            "password": "123",
        },
    )
    assert short_resp.status_code == 400

    # 4. Login with valid credentials
    login_resp = client.post(
        "/api/auth/login",
        json={
            "email": "david@fincherfans.com",
            "password": "MindhunterSeason3When?",
        },
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert "access_token" in login_data

    # 5. Login with wrong password fails
    bad_login = client.post(
        "/api/auth/login",
        json={
            "email": "david@fincherfans.com",
            "password": "WrongPassword!",
        },
    )
    assert bad_login.status_code == 401

    # 6. Verify /api/auth/me with Bearer token
    me_resp = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["authenticated"] is True
    assert me_data["user"]["email"] == "david@fincherfans.com"

    # 7. Verify /api/auth/me without token returns unauthenticated
    anon_resp = client.get("/api/auth/me")
    assert anon_resp.status_code == 200
    assert anon_resp.json()["authenticated"] is False
