"""Task 2.1: Single completion call using raw httpx against the LLM REST API.

No LangChain, no high-level frameworks. Raw HTTP requests, headers,
payload construction, and explicit parsing of content, stop reason,
and token usage metadata.
"""

import argparse
import sys
from typing import Any

import httpx
from pydantic import BaseModel, Field

from phase_2_raw_api.config import get_settings


class SingleCompletionResponse(BaseModel):
    """Structured result of a single LLM completion call."""

    content: str = Field(description="Generated text content from the LLM")
    stop_reason: str = Field(description="Finish/stop reason returned by the model")
    input_tokens: int = Field(description="Count of input (prompt) tokens")
    output_tokens: int = Field(description="Count of output (completion) tokens")
    total_tokens: int = Field(description="Total tokens consumed (input + output)")
    model_version: str = Field(default="", description="Model version reported by API")
    raw_response: dict[str, Any] = Field(
        default_factory=dict, description="Full raw JSON response from API"
    )


def build_generate_content_payload(prompt: str) -> dict[str, Any]:
    """Construct the raw JSON payload expected by the Gemini REST API."""
    return {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }


def parse_gemini_response(data: dict[str, Any]) -> SingleCompletionResponse:
    """Parse raw JSON response from Gemini REST API into a typed response object."""
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates returned in API response: {data}")

    candidate = candidates[0]
    content_obj = candidate.get("content", {})
    parts = content_obj.get("parts", [])

    # Concatenate text from all text parts (handles multi-part or thought outputs)
    text_segments = [p.get("text", "") for p in parts if "text" in p]
    content_text = "".join(text_segments)

    stop_reason = candidate.get("finishReason", "UNKNOWN")

    usage = data.get("usageMetadata", {})
    input_tokens = int(usage.get("promptTokenCount", 0))
    output_tokens = int(usage.get("candidatesTokenCount", 0))
    total_tokens = int(usage.get("totalTokenCount", input_tokens + output_tokens))
    model_version = str(data.get("modelVersion", ""))

    return SingleCompletionResponse(
        content=content_text,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model_version=model_version,
        raw_response=data,
    )


def generate_single_completion(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> SingleCompletionResponse:
    """Send a single prompt to the LLM via raw HTTP POST and parse the result."""
    settings = get_settings()
    effective_api_key = settings.gemini_api_key if api_key is None else api_key
    effective_model = settings.gemini_model if model is None else model

    if not effective_api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY in .env or pass api_key."
        )

    endpoint_url = (
        f"{settings.gemini_base_url}/models/{effective_model}:generateContent"
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
        response = client.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        json_data = response.json()
        return parse_gemini_response(json_data)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        err_msg = exc.response.text
        raise RuntimeError(
            f"API request failed with HTTP {status}: {err_msg}"
        ) from exc
    finally:
        if should_close_client:
            client.close()


async def generate_single_completion_async(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> SingleCompletionResponse:
    """Async variant: Send a single prompt via raw async httpx POST."""
    settings = get_settings()
    effective_api_key = settings.gemini_api_key if api_key is None else api_key
    effective_model = settings.gemini_model if model is None else model

    if not effective_api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY in .env or pass api_key."
        )

    endpoint_url = (
        f"{settings.gemini_base_url}/models/{effective_model}:generateContent"
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
        response = await client.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        json_data = response.json()
        return parse_gemini_response(json_data)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        err_msg = exc.response.text
        raise RuntimeError(
            f"API request failed with HTTP {status}: {err_msg}"
        ) from exc
    finally:
        if should_close_client:
            await client.aclose()


def main() -> None:
    """CLI entrypoint to demonstrate Task 2.1."""
    parser = argparse.ArgumentParser(
        description="Phase 2.1 - Single raw httpx completion call"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain recursion in two sentences using Inception analogy.",
        help="Prompt text to send to the LLM",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PHASE 2.1: RAW HTTPX SINGLE COMPLETION CALL")
    print("=" * 60)
    print(f"Prompt: {args.prompt}\n")

    try:
        result = generate_single_completion(args.prompt)
        print("-" * 60)
        print("CONTENT:")
        print(result.content.strip())
        print("-" * 60)
        print(f"STOP REASON:   {result.stop_reason}")
        print(f"INPUT TOKENS:  {result.input_tokens}")
        print(f"OUTPUT TOKENS: {result.output_tokens}")
        print(f"TOTAL TOKENS:  {result.total_tokens}")
        print(f"MODEL VERSION: {result.model_version}")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"Error during raw completion: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
