import pytest
from fastapi.testclient import TestClient

from phase_0_baseline.app import app, generate_chunks

client = TestClient(app)


def test_echo_endpoint_success() -> None:
    payload = {"message": "Hello FastAPI", "metadata": {"source": "test"}}
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Hello FastAPI"
    assert data["metadata"] == {"source": "test"}
    assert "received_at" in data


def test_echo_endpoint_validation_error() -> None:
    # Missing required 'message' field
    response = client.post("/echo", json={"metadata": {}})
    assert response.status_code == 422

    # Empty message should also fail due to min_length=1
    response_empty = client.post("/echo", json={"message": ""})
    assert response_empty.status_code == 422


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
