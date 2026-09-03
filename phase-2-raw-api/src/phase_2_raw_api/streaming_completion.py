"""Task 2.2: Streaming LLM completions using raw httpx and Server-Sent Events (SSE).

Demonstrates real-time token delivery via HTTP chunked streaming without frameworks.
Tokens are yielded and displayed immediately as they arrive over the wire.
"""

import argparse
import json
import sys
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.single_completion import build_generate_content_payload


class StreamChunk(BaseModel):
    """Structured representation of a single streaming chunk from the model."""

    text: str = Field(default="", description="Delta text received in this chunk")
    finish_reason: str | None = Field(
        default=None, description="Stop/finish reason if reported in this chunk"
    )
    input_tokens: int | None = Field(
        default=None, description="Input token count if reported"
    )
    output_tokens: int | None = Field(
        default=None, description="Output token count if reported"
    )
    total_tokens: int | None = Field(
        default=None, description="Total token count if reported"
    )
    is_final: bool = Field(
        default=False, description="Whether this is the final closing chunk"
    )


def parse_sse_line(line: str) -> StreamChunk | None:
    """Parse a single Server-Sent Events (SSE) line from the stream."""
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None

    json_payload = stripped[5:].strip()
    if not json_payload or json_payload == "[DONE]":
        return StreamChunk(is_final=True)

    try:
        data: dict[str, Any] = json.loads(json_payload)
    except json.JSONDecodeError:
        return None

    candidates = data.get("candidates", [])
    text_delta = ""
    finish_reason = None

    if candidates:
        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])
        text_delta = "".join(p.get("text", "") for p in parts if "text" in p)
        finish_reason = candidate.get("finishReason")

    usage = data.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount")
    output_tokens = usage.get("candidatesTokenCount")
    total_tokens = usage.get("totalTokenCount")

    is_final = finish_reason is not None

    return StreamChunk(
        text=text_delta,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        is_final=is_final,
    )


def stream_completion(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> Iterator[StreamChunk]:
    """Synchronous generator yielding StreamChunks as they arrive from the API."""
    settings = get_settings()
    effective_api_key = settings.gemini_api_key if api_key is None else api_key
    effective_model = settings.gemini_model if model is None else model

    if not effective_api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY in .env or pass api_key."
        )

    endpoint_url = (
        f"{settings.gemini_base_url}/models/"
        f"{effective_model}:streamGenerateContent?alt=sse"
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": effective_api_key,
    }
    payload = build_generate_content_payload(prompt)

    should_close_client = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close_client = True

    try:
        with client.stream(
            "POST", endpoint_url, headers=headers, json=payload
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"API stream failed with HTTP {response.status_code}: {body}"
                )

            for line in response.iter_lines():
                chunk = parse_sse_line(line)
                if chunk is not None:
                    yield chunk
    finally:
        if should_close_client:
            client.close()


async def stream_completion_async(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[StreamChunk]:
    """Asynchronous generator yielding StreamChunks in real time."""
    settings = get_settings()
    effective_api_key = settings.gemini_api_key if api_key is None else api_key
    effective_model = settings.gemini_model if model is None else model

    if not effective_api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY in .env or pass api_key."
        )

    endpoint_url = (
        f"{settings.gemini_base_url}/models/"
        f"{effective_model}:streamGenerateContent?alt=sse"
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": effective_api_key,
    }
    payload = build_generate_content_payload(prompt)

    should_close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=settings.timeout_seconds)
        should_close_client = True

    try:
        async with client.stream(
            "POST", endpoint_url, headers=headers, json=payload
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"API stream failed with HTTP {response.status_code}: {body}"
                )

            async for line in response.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is not None:
                    yield chunk
    finally:
        if should_close_client:
            await client.aclose()


def main() -> None:
    """CLI entrypoint demonstrating streaming token output."""
    parser = argparse.ArgumentParser(
        description="Phase 2.2 - Streaming LLM completion call"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain how a browser renders a web page from HTML parsing to GPU "
            "painting, step by step in three detailed paragraphs."
        ),
        help="Prompt text to send to the LLM",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Show explicit chunk boundaries and timing metadata",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PHASE 2.2: STREAMING COMPLETION (TOKEN-BY-TOKEN)")
    print("=" * 60)
    print(f"Prompt: {args.prompt}\n")
    print("Connecting to stream (measuring Time-To-First-Token)...")

    chunk_count = 0
    full_text = []
    final_stop_reason = None
    input_tokens = None
    output_tokens = None
    total_tokens = None
    first_token_received = False

    try:
        for chunk in stream_completion(args.prompt):
            if not first_token_received and chunk.text:
                first_token_received = True
                print("First token received! Live stream starting:")
                print("-" * 60)

            if chunk.text:
                if args.show_chunks:
                    sys.stdout.write(f"\n[Chunk {chunk_count + 1}]: {chunk.text}")
                else:
                    sys.stdout.write(chunk.text)
                sys.stdout.flush()
                full_text.append(chunk.text)
                chunk_count += 1

            if chunk.finish_reason:
                final_stop_reason = chunk.finish_reason
            if chunk.input_tokens is not None:
                input_tokens = chunk.input_tokens
            if chunk.output_tokens is not None:
                output_tokens = chunk.output_tokens
            if chunk.total_tokens is not None:
                total_tokens = chunk.total_tokens

        print("\n" + "-" * 60)
        print(f"CHUNKS RECEIVED: {chunk_count}")
        print(f"STOP REASON:     {final_stop_reason or 'UNKNOWN'}")
        if input_tokens is not None:
            print(f"INPUT TOKENS:    {input_tokens}")
        if output_tokens is not None:
            print(f"OUTPUT TOKENS:   {output_tokens}")
        if total_tokens is not None:
            print(f"TOTAL TOKENS:    {total_tokens}")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"\nError during streaming: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
