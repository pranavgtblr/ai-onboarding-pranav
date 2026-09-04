"""Task 2.6: Hand-written Tool-Calling Loop with Calculator & Fake Weather.

Raw API interaction without frameworks.
Sends tool schemas, parses model function calls, executes them locally,
appends the function responses to the conversation, and repeats the loop
until the model stops requesting tools.
Includes a max-iteration guard and logs every iteration.
"""

import argparse
import logging
import sys
from typing import Any

import httpx
from pydantic import BaseModel, Field

from phase_2_raw_api.config import get_settings

logger = logging.getLogger("tool_calling")


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------


def execute_calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """Execute basic arithmetic operations."""
    op = operation.lower().strip()
    if op == "add":
        res = a + b
    elif op == "subtract":
        res = a - b
    elif op == "multiply":
        res = a * b
    elif op == "divide":
        if b == 0:
            return {"error": "Division by zero is undefined."}
        res = a / b
    else:
        return {"error": f"Unknown operation: '{operation}'."}

    # Format numbers cleanly (e.g. 714 instead of 714.0)
    if isinstance(res, float):
        formatted = int(res) if res.is_integer() else round(res, 4)
    else:
        formatted = res
    return {"result": formatted}


MOCK_WEATHER_DATABASE: dict[str, dict[str, Any]] = {
    "tokyo": {
        "city": "Tokyo",
        "temperature_c": 18,
        "condition": "Clear and Sunny",
        "humidity": "55%",
        "wind": "12 km/h",
    },
    "london": {
        "city": "London",
        "temperature_c": 11,
        "condition": "Overcast with light drizzle",
        "humidity": "82%",
        "wind": "20 km/h",
    },
    "new york": {
        "city": "New York",
        "temperature_c": 15,
        "condition": "Partly Cloudy",
        "humidity": "60%",
        "wind": "14 km/h",
    },
    "paris": {
        "city": "Paris",
        "temperature_c": 14,
        "condition": "Mild and Breezy",
        "humidity": "68%",
        "wind": "16 km/h",
    },
    "san francisco": {
        "city": "San Francisco",
        "temperature_c": 13,
        "condition": "Foggy morning clearing to sun",
        "humidity": "75%",
        "wind": "18 km/h",
    },
}


def execute_weather(city: str) -> dict[str, Any]:
    """Return mock weather data for a given city."""
    normalized = city.strip().lower()
    if normalized in MOCK_WEATHER_DATABASE:
        return MOCK_WEATHER_DATABASE[normalized]

    # Sensible fallback for unmocked cities
    return {
        "city": city.title(),
        "temperature_c": 20,
        "condition": "Partly Cloudy",
        "humidity": "65%",
        "wind": "10 km/h",
    }


# Map tool names to their execution functions
TOOL_REGISTRY: dict[str, Any] = {
    "calculator": execute_calculator,
    "get_weather": execute_weather,
}


# ---------------------------------------------------------------------------
# Tool Schemas for Gemini REST API
# ---------------------------------------------------------------------------


def get_tool_declarations() -> list[dict[str, Any]]:
    """Return the function declarations payload formatted for Gemini REST API."""
    return [
        {
            "function_declarations": [
                {
                    "name": "calculator",
                    "description": (
                        "Perform basic arithmetic calculations. "
                        "Supported operations: add, subtract, multiply, divide."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "operation": {
                                "type": "STRING",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "The arithmetic operation to perform",
                            },
                            "a": {
                                "type": "NUMBER",
                                "description": "First operand",
                            },
                            "b": {
                                "type": "NUMBER",
                                "description": "Second operand",
                            },
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
                {
                    "name": "get_weather",
                    "description": (
                        "Get current weather information and temperature for a city."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "city": {
                                "type": "STRING",
                                "description": "City name, e.g. Tokyo, London",
                            },
                        },
                        "required": ["city"],
                    },
                },
            ]
        }
    ]


# ---------------------------------------------------------------------------
# Data Models for Results & Execution Logs
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """Log record of a single tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


class IterationLog(BaseModel):
    """Log record of one iteration of the tool loop."""

    iteration: int
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    model_text: str = ""
    tokens_consumed: int = 0


class ToolLoopResult(BaseModel):
    """Final result of the tool calling execution loop."""

    final_text: str
    total_iterations: int
    iteration_logs: list[IterationLog]
    total_tokens: int
    hit_max_guard: bool = False


# ---------------------------------------------------------------------------
# Core Tool-Calling Loop
# ---------------------------------------------------------------------------


def run_tool_calling_loop(
    prompt: str,
    *,
    max_iterations: int = 5,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
    logger_override: logging.Logger | None = None,
) -> ToolLoopResult:
    """Execute multi-turn tool calling loop until model stops requesting tools.

    Args:
        prompt: Initial user question or instruction.
        max_iterations: Safety guard to prevent infinite loops (default 5).
        model: Optional model identifier override.
        api_key: Optional API key override.
        client: Optional shared httpx.Client instance.
        logger_override: Optional custom logger.

    Returns:
        ToolLoopResult: Final text answer, iteration logs, and token accounting.
    """
    active_logger = logger_override or logger
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
    tools_payload = get_tool_declarations()

    # Initial conversation state
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]

    iteration_logs: list[IterationLog] = []
    cumulative_tokens = 0
    final_text = ""
    hit_guard = False

    should_close_client = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close_client = True

    try:
        for iteration in range(1, max_iterations + 1):
            active_logger.info(
                "--- [Tool Loop Iteration %d/%d] Sending request to model ---",
                iteration,
                max_iterations,
            )

            payload = {
                "contents": contents,
                "tools": tools_payload,
            }

            response = client.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Accounting
            usage = data.get("usageMetadata", {})
            prompt_toks = int(usage.get("promptTokenCount", 0))
            cand_toks = int(usage.get("candidatesTokenCount", 0))
            iter_tokens = int(usage.get("totalTokenCount", prompt_toks + cand_toks))
            cumulative_tokens += iter_tokens

            candidates = data.get("candidates", [])
            if not candidates:
                active_logger.warning("No candidates returned by model.")
                break

            candidate = candidates[0]
            content_obj = candidate.get("content", {})
            parts = content_obj.get("parts", [])

            # Extract any tool calls in this response
            function_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for part in parts:
                if "functionCall" in part:
                    function_calls.append(part["functionCall"])
                if "text" in part:
                    text_parts.append(part["text"])

            current_text = "".join(text_parts).strip()

            # Case A: Model did NOT request any tools. Loop is complete!
            if not function_calls:
                final_text = current_text
                active_logger.info(
                    "[Iteration %d] Model finished thinking (no further tool calls).",
                    iteration,
                )
                iteration_logs.append(
                    IterationLog(
                        iteration=iteration,
                        tool_calls=[],
                        model_text=final_text,
                        tokens_consumed=iter_tokens,
                    )
                )
                break

            # Case B: Model requested one or more tool calls.
            active_logger.info(
                "[Iteration %d] Model requested %d tool call(s): %s",
                iteration,
                len(function_calls),
                [fc.get("name") for fc in function_calls],
            )

            # Important: Append the model's exact candidate content to conversation
            contents.append(content_obj)

            # Execute all requested functions and prepare responses
            tool_records: list[ToolCallRecord] = []
            function_response_parts: list[dict[str, Any]] = []

            for fc in function_calls:
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})

                active_logger.info(
                    "  ⚙️ Executing tool '%s' with args %s", fn_name, fn_args
                )

                fn_handler = TOOL_REGISTRY.get(fn_name)
                if fn_handler is None:
                    fn_result = {"error": f"Tool '{fn_name}' is not registered."}
                else:
                    try:
                        fn_result = fn_handler(**fn_args)
                    except Exception as e:
                        fn_result = {"error": f"Tool execution error: {e}"}

                active_logger.info("  ↳ Tool result: %s", fn_result)

                tool_records.append(
                    ToolCallRecord(
                        tool_name=fn_name,
                        arguments=fn_args,
                        output=fn_result,
                    )
                )

                function_response_parts.append(
                    {
                        "functionResponse": {
                            "name": fn_name,
                            "response": fn_result,
                        }
                    }
                )

            # Append the tool responses turn to history
            contents.append({"role": "user", "parts": function_response_parts})

            iteration_logs.append(
                IterationLog(
                    iteration=iteration,
                    tool_calls=tool_records,
                    model_text=current_text,
                    tokens_consumed=iter_tokens,
                )
            )

        else:
            # Reached loop exhaustion without normal break
            hit_guard = True
            active_logger.warning(
                "🚨 Max-iteration guard reached (%d iterations). Terminating loop.",
                max_iterations,
            )
            final_text = (
                final_text
                or "[Max iterations reached before model completed its response]"
            )

    finally:
        if should_close_client:
            client.close()

    return ToolLoopResult(
        final_text=final_text,
        total_iterations=len(iteration_logs),
        iteration_logs=iteration_logs,
        total_tokens=cumulative_tokens,
        hit_max_guard=hit_guard,
    )


# ---------------------------------------------------------------------------
# CLI Demo Interface
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for demonstrating the hand-written tool calling loop."""
    parser = argparse.ArgumentParser(
        description="Phase 2.6: Hand-written Tool-Calling Loop with Tools."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "What is 42 multiplied by 17, and what is the current weather in Tokyo? "
            "Please combine both answers."
        ),
        help="Prompt that triggers one or more tool calls.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum allowed loop iterations before aborting (default 5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model identifier override.",
    )

    args = parser.parse_args()

    # Configure stdout logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 78)
    print(" 🛠️  TASK 2.6: HAND-WRITTEN TOOL-CALLING LOOP")
    print("=" * 78)
    print(f"User Prompt    : '{args.prompt}'")
    print(f"Max Iterations : {args.max_iterations}")
    print("Registered Tools: calculator, get_weather")
    print("-" * 78)

    try:
        result = run_tool_calling_loop(
            prompt=args.prompt,
            max_iterations=args.max_iterations,
            model=args.model,
        )

        print("\n" + "=" * 78)
        print(" 📋 EXECUTION AUDIT TRAIL")
        print("=" * 78)
        for log in result.iteration_logs:
            print(f"\n[Iteration {log.iteration}] Tokens: {log.tokens_consumed}")
            if log.tool_calls:
                for tc in log.tool_calls:
                    print(f"  • Tool Invocation : {tc.tool_name}({tc.arguments})")
                    print(f"    Tool Output     : {tc.output}")
            if log.model_text:
                print(f"  • Intermediate Thought / Text: {log.model_text}")

        print("\n" + "=" * 78)
        print(" 🎯 FINAL SYNTHESIZED RESPONSE")
        print("=" * 78)
        print(result.final_text)
        print("-" * 78)
        print(f"Total Iterations : {result.total_iterations}")
        print(f"Total Tokens     : {result.total_tokens}")
        print(f"Hit Max Guard    : {result.hit_max_guard}")
        print("=" * 78 + "\n")

    except Exception as e:
        print(f"\n❌ Error during tool loop execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
