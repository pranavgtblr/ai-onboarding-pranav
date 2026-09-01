import httpx
import pytest

from phase_0_baseline.app import app
from phase_0_baseline.concurrency_demo import (
    fetch_concurrent,
    fetch_sequential,
    run_benchmark,
    run_demo,
)


@pytest.mark.asyncio
async def test_fetch_concurrent_and_sequential() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        urls = ["http://testserver/delay/10"] * 5

        # Concurrent
        concurrent_results = await fetch_concurrent(urls, client=client)
        assert len(concurrent_results) == 5

        # Sequential
        sequential_results = await fetch_sequential(urls, client=client)
        assert len(sequential_results) == 5

        assert concurrent_results == sequential_results


@pytest.mark.asyncio
async def test_run_benchmark() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        urls = ["http://testserver/delay/10"] * 3
        result = await run_benchmark(urls, client=client)

        assert result.responses_count == 3
        assert result.concurrent_duration > 0
        assert result.sequential_duration > 0


@pytest.mark.asyncio
async def test_run_demo() -> None:
    result = await run_demo()
    assert result.responses_count == 10
    # Sequential should take roughly ~1.0s (10 * 100ms) whereas concurrent takes ~0.1s
    assert result.sequential_duration > result.concurrent_duration
