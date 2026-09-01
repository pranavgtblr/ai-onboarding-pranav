import pytest
from fastapi.testclient import TestClient

from phase_0_baseline.app import app, generate_chunks

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "app_env" in data


def test_echo_endpoint_valid_body() -> None:
    # Standard valid payload
    payload = {"message": "Hello FastAPI", "metadata": {"source": "test", "id": 123}}
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Hello FastAPI"
    assert data["metadata"] == {"source": "test", "id": 123}
    assert "received_at" in data

    # Valid payload with default empty metadata
    response_no_meta = client.post("/echo", json={"message": "No metadata"})
    assert response_no_meta.status_code == 200
    assert response_no_meta.json()["metadata"] == {}


def test_echo_endpoint_invalid_body() -> None:
    # 1. Missing required 'message' field
    response_missing = client.post("/echo", json={"metadata": {}})
    assert response_missing.status_code == 422

    # 2. Empty string message (violates min_length=1)
    response_empty = client.post("/echo", json={"message": ""})
    assert response_empty.status_code == 422

    # 3. Invalid body type (e.g. dictionary passed as message string)
    response_invalid_obj = client.post("/echo", json={"message": {"nested": "obj"}})
    assert response_invalid_obj.status_code == 422

    # 4. Invalid JSON payload entirely
    response_not_json = client.post(
        "/echo",
        content="not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response_not_json.status_code == 422


def test_stream_endpoint() -> None:
    response = client.get("/stream")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    lines = [line for line in response.text.strip().split("\n") if line]
    assert len(lines) == 20
    assert lines[0] == "chunk 1"
    assert lines[-1] == "chunk 20"


@pytest.mark.asyncio
async def test_generate_chunks_generator() -> None:
    chunks: list[str] = []
    async for chunk in generate_chunks(total_chunks=3, delay_seconds=0.01):
        chunks.append(chunk)
    assert chunks == ["chunk 1\n", "chunk 2\n", "chunk 3\n"]
