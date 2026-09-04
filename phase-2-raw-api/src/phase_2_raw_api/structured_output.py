"""Task 2.5: Structured Output with Pydantic Validation and Self-Healing Retry.

Direct LLM interaction without frameworks, generating validated Pydantic models.
If the model returns invalid JSON or fails schema validation, the error diagnostics
are appended to the conversation history and retried once.
Includes a '--force-bad' flag to deliberately trigger validation errors and prove
the self-healing retry mechanism.
"""

import argparse
import json
import re
import sys
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phase_2_raw_api.config import get_settings

T = TypeVar("T", bound=BaseModel)

# Intentionally malformed payload used when force_bad_first_attempt=True
FORCED_BAD_JSON = json.dumps(
    {
        "name": "Iron Man",
        "real_name": "Tony Stark",
        "role": "Tech Specialist",
        "power_level": 9999,  # Violates ge=1, le=100
        "abilities": [],  # Violates min_length=1
        "is_active": "not-a-boolean",  # Violates bool
        "summary": "T" * 280,  # Violates max_length=200
    },
    indent=2,
)


class HeroProfile(BaseModel):
    """Superhero profile with strict field-level constraints for validation."""

    name: str = Field(description="Superhero codename, e.g. Iron Man")
    real_name: str = Field(description="Civilian identity, e.g. Tony Stark")
    role: str = Field(
        description=(
            "Primary role, e.g. Leader, Tactician, Heavy Hitter, Tech Specialist"
        )
    )
    power_level: int = Field(
        ge=1,
        le=100,
        description="Power level rating strictly between 1 and 100",
    )
    abilities: list[str] = Field(
        min_length=1,
        max_length=5,
        description="List of 1 to 5 primary abilities or powers",
    )
    is_active: bool = Field(
        description="Whether the hero is currently active in the team"
    )
    summary: str = Field(
        max_length=200,
        description="Concise backstory or character summary under 200 characters",
    )


class StructuredOutputResult(BaseModel, Generic[T]):
    """Result of structured output generation containing the validated object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: T = Field(description="Validated Pydantic model instance")
    attempts: int = Field(description="Total number of attempts made (1 or 2)")
    retried: bool = Field(
        default=False, description="Whether at least one retry was performed"
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="List of validation error messages encountered during retries",
    )
    raw_responses: list[str] = Field(
        default_factory=list,
        description="Raw response text received on each attempt",
    )
    total_tokens: int = Field(
        default=0,
        description="Cumulative total tokens consumed across all attempts",
    )


class StructuredOutputError(Exception):
    """Raised when structured output generation fails validation after all retries."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        errors: list[str],
        raw_responses: list[str],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors
        self.raw_responses = raw_responses


def clean_json_text(raw_text: str) -> str:
    """Extract and clean raw JSON string from potential markdown code fences."""
    text = raw_text.strip()

    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()

    return text


def format_validation_error(error: Exception) -> str:
    """Format a Pydantic ValidationError into actionable feedback for LLMs."""
    if isinstance(error, ValidationError):
        lines: list[str] = []
        for err in error.errors():
            loc = " -> ".join(str(elem) for elem in err.get("loc", []))
            msg = err.get("msg", "")
            inp = err.get("input", None)
            lines.append(f"- Field '{loc}': {msg} (received invalid value: {inp!r})")
        return "\n".join(lines)
    return f"- JSON Syntax Error: {error}"


def build_schema_prompt(user_prompt: str, model_cls: type[BaseModel]) -> str:
    """Construct prompt with explicit JSON Schema and strict output instructions."""
    schema_json = json.dumps(model_cls.model_json_schema(), indent=2)
    return (
        f"{user_prompt}\n\n"
        "You MUST respond ONLY with a single valid JSON object that strictly adheres "
        "to the following JSON Schema. Do NOT include markdown code fences, comments, "
        "or explanations before or after the JSON.\n\n"
        f"JSON Schema:\n{schema_json}"
    )


def execute_content_generation(
    contents: list[dict[str, Any]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> tuple[str, int]:
    """Execute raw HTTP request to Gemini REST API with JSON response mime-type.

    Returns:
        tuple[str, int]: The generated raw response text and the total token count.
    """
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

    payload = {
        "contents": contents,
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }

    should_close_client = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close_client = True

    try:
        response = client.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    finally:
        if should_close_client:
            client.close()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates returned in API response: {data}")

    candidate = candidates[0]
    content_obj = candidate.get("content", {})
    parts = content_obj.get("parts", [])
    text_segments = [p.get("text", "") for p in parts if "text" in p]
    raw_text = "".join(text_segments)

    usage = data.get("usageMetadata", {})
    input_tokens = int(usage.get("promptTokenCount", 0))
    output_tokens = int(usage.get("candidatesTokenCount", 0))
    total_tokens = int(usage.get("totalTokenCount", input_tokens + output_tokens))

    return raw_text, total_tokens


def generate_structured_output(
    prompt: str,
    model_cls: type[T],
    *,
    max_retries: int = 1,
    force_bad_first_attempt: bool = False,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> StructuredOutputResult[T]:
    """Generate and validate a structured Pydantic object from an LLM prompt.

    If validation fails (or if force_bad_first_attempt is True on attempt 1),
    the error details are appended as a diagnostic retry turn and sent back to the
    model for self-correction.

    Args:
        prompt: User prompt describing the entity to generate.
        model_cls: Target Pydantic model class for validation.
        max_retries: Maximum number of retry attempts on validation error (default 1).
        force_bad_first_attempt: If True, deliberately forces attempt 1 to produce
            invalid data to demonstrate and prove the self-healing retry.
        model: Optional model identifier override.
        api_key: Optional API key override.
        client: Optional shared httpx.Client.

    Returns:
        StructuredOutputResult[T]: Validated Pydantic object and run diagnostics.

    Raises:
        StructuredOutputError: If validation fails after all retry attempts.
    """
    initial_prompt = build_schema_prompt(prompt, model_cls)
    contents: list[dict[str, Any]] = [
        {"role": "user", "parts": [{"text": initial_prompt}]}
    ]

    total_tokens_accumulated = 0
    validation_errors: list[str] = []
    raw_responses: list[str] = []
    total_attempts = 0

    max_attempts = 1 + max(0, max_retries)

    for attempt in range(1, max_attempts + 1):
        total_attempts = attempt

        # Execute API call
        raw_text, tokens = execute_content_generation(
            contents=contents,
            model=model,
            api_key=api_key,
            client=client,
        )
        total_tokens_accumulated += tokens

        # If forced bad output is requested for attempt 1, override raw_text
        if attempt == 1 and force_bad_first_attempt:
            raw_text = FORCED_BAD_JSON

        raw_responses.append(raw_text)

        # Attempt to clean and validate
        cleaned_json = clean_json_text(raw_text)
        try:
            validated_object = model_cls.model_validate_json(cleaned_json)
            return StructuredOutputResult[T](
                data=validated_object,
                attempts=total_attempts,
                retried=(total_attempts > 1),
                validation_errors=validation_errors,
                raw_responses=raw_responses,
                total_tokens=total_tokens_accumulated,
            )
        except (ValidationError, json.JSONDecodeError) as err:
            formatted_err = format_validation_error(err)
            validation_errors.append(formatted_err)

            # If retries remain, append error context and retry
            if attempt < max_attempts:
                # Turn 2: Assistant's faulty output
                contents.append({"role": "model", "parts": [{"text": raw_text}]})

                # Turn 3: User feedback with error message and request to fix
                retry_feedback = (
                    "Your previous response was INVALID and failed schema validation "
                    f"with the following errors:\n{formatted_err}\n\n"
                    "Please fix all validation errors above and return ONLY a valid "
                    "JSON object matching the required schema. Do not change "
                    "unaffected fields."
                )
                contents.append({"role": "user", "parts": [{"text": retry_feedback}]})

    # If we exited the loop without returning, all attempts failed
    raise StructuredOutputError(
        f"Failed to generate valid {model_cls.__name__} after "
        f"{total_attempts} attempts.",
        attempts=total_attempts,
        errors=validation_errors,
        raw_responses=raw_responses,
    )


def main() -> None:
    """CLI entrypoint for demonstrating structured output and self-healing retries."""
    parser = argparse.ArgumentParser(
        description="Phase 2.5: Structured Output with Pydantic Validation & Retry."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Generate a profile for Iron Man (Tony Stark)",
        help="Prompt describing the character profile to generate.",
    )
    parser.add_argument(
        "--force-bad",
        action="store_true",
        help=(
            "Deliberately inject an invalid output on attempt 1 to prove "
            "self-healing retry works."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model identifier (e.g. gemini-3.5-flash-lite).",
    )

    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(" 🛡️  TASK 2.5: STRUCTURED OUTPUT & SELF-HEALING RETRY")
    print("=" * 75)
    print(f"Target Schema : {HeroProfile.__name__}")
    print(f"User Prompt   : '{args.prompt}'")
    print(f"Force Bad Flag: {args.force_bad}")
    print("-" * 75)

    try:
        result = generate_structured_output(
            prompt=args.prompt,
            model_cls=HeroProfile,
            max_retries=1,
            force_bad_first_attempt=args.force_bad,
            model=args.model,
        )

        if result.retried:
            print("\n🚨 [ATTEMPT 1] Validation Failure Detected (Forced / Invalid):")
            print("   Raw Output Received:")
            print("   " + "\n   ".join(result.raw_responses[0].splitlines()))
            print("\n❌ Validation Errors Caught by Pydantic:")
            for err in result.validation_errors:
                for line in err.splitlines():
                    print(f"   {line}")
            print("\n🔄 [ATTEMPT 2] Appended Error Diagnostic & Retrying with LLM...")
            print("   Self-Correction Received from LLM:")
            print("   " + "\n   ".join(result.raw_responses[1].splitlines()))
        else:
            print("\n✅ [ATTEMPT 1] Output validated on first attempt without errors!")

        print("\n" + "=" * 75)
        print(" 🎉 VALIDATED PYDANTIC OBJECT (SUCCESS)")
        print("=" * 75)
        print(f"Attempts Made : {result.attempts}")
        healed_label = (
            "YES (Retry Successful)" if result.retried else "NO (Clean on 1st try)"
        )
        print(f"Self-Healed   : {healed_label}")
        print(f"Total Tokens  : {result.total_tokens}")
        print("-" * 75)
        print("Parsed Attributes:")
        print(f"  • Name         : {result.data.name}")
        print(f"  • Real Name    : {result.data.real_name}")
        print(f"  • Role         : {result.data.role}")
        print(f"  • Power Level  : {result.data.power_level}/100")
        print(f"  • Abilities    : {', '.join(result.data.abilities)}")
        print(f"  • Is Active    : {result.data.is_active}")
        print(f"  • Summary      : {result.data.summary}")
        print("-" * 75)
        print("JSON Dump:")
        print(result.data.model_dump_json(indent=2))
        print("=" * 75 + "\n")

    except StructuredOutputError as e:
        print("\n❌ Structured output generation failed permanently:")
        print(f"   Attempts: {e.attempts}")
        for idx, err in enumerate(e.errors, 1):
            print(f"   Error {idx}: {err}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
