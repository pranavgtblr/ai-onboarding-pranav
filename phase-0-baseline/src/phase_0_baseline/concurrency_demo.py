import asyncio
import time
from typing import NamedTuple

import httpx


class BenchmarkResult(NamedTuple):
    concurrent_duration: float
    sequential_duration: float
    responses_count: int


async def fetch_url(client: httpx.AsyncClient, url: str) -> str:
    """Fetch a single URL asynchronously using httpx."""
    response = await client.get(url)
    response.raise_for_status()
    return response.text


async def fetch_concurrent(
    urls: list[str], client: httpx.AsyncClient | None = None
) -> list[str]:
    """Fetch all URLs concurrently using asyncio.gather."""
    should_close = False
    if client is None:
        client = httpx.AsyncClient()
        should_close = True

    try:
        tasks = [fetch_url(client, url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return list(responses)
    finally:
        if should_close:
            await client.aclose()


async def fetch_sequential(
    urls: list[str], client: httpx.AsyncClient | None = None
) -> list[str]:
    """Fetch all URLs sequentially one by one."""
    should_close = False
    if client is None:
        client = httpx.AsyncClient()
        should_close = True

    try:
        responses: list[str] = []
        for url in urls:
            res = await fetch_url(client, url)
            responses.append(res)
        return responses
    finally:
        if should_close:
            await client.aclose()


async def run_benchmark(
    urls: list[str], client: httpx.AsyncClient | None = None
) -> BenchmarkResult:
    """Run both concurrent and sequential requests and measure duration."""
    start_c = time.perf_counter()
    concurrent_results = await fetch_concurrent(urls, client=client)
    duration_c = time.perf_counter() - start_c

    start_s = time.perf_counter()
    sequential_results = await fetch_sequential(urls, client=client)
    duration_s = time.perf_counter() - start_s

    assert len(concurrent_results) == len(sequential_results) == len(urls)

    return BenchmarkResult(
        concurrent_duration=duration_c,
        sequential_duration=duration_s,
        responses_count=len(urls),
    )


async def run_demo() -> BenchmarkResult:
    """Run benchmark against 100ms delayed endpoint via in-memory ASGI."""
    from phase_0_baseline.app import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        urls = ["http://testserver/delay/100"] * 10
        return await run_benchmark(urls, client=client)


def main() -> None:
    """CLI runner demonstrating concurrency gap."""
    print("Running Concurrency Benchmark (10 requests @ 100ms latency)...")
    result = asyncio.run(run_demo())

    print("==================================================")
    print(f"Total Requests:        {result.responses_count}")
    print(f"Sequential Duration:   {result.sequential_duration:.4f} seconds")
    print(f"Concurrent Duration:   {result.concurrent_duration:.4f} seconds")
    speedup = (
        result.sequential_duration / result.concurrent_duration
        if result.concurrent_duration > 0
        else 0
    )
    print(f"Speedup Factor:        {speedup:.2f}x faster")
    print("==================================================")


if __name__ == "__main__":
    main()
