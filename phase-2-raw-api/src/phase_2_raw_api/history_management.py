"""Task 2.4: History management with token thresholds (drop vs summarize).

When conversation context exceeds N tokens, manages history using either:
1. 'drop': Evicts the oldest turns (sliding window), dropping memory of early facts.
2. 'summarize': Uses the LLM to compress the oldest turns into a concise summary,
   preserving key facts and entity state while drastically reducing token count.
"""

import argparse
import sys
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from phase_2_raw_api.config import get_settings


class ChatMessage(BaseModel):
    """Represents a single message in the conversation history."""

    role: Literal["user", "model", "system"]
    content: str


def estimate_tokens(text: str) -> int:
    """Fast, reliable token estimation (~4 characters per token + word bonus)."""
    if not text:
        return 0
    # Words + char length heuristic closely approximates English BPE tokenizers
    words = len(text.split())
    chars = len(text)
    return max(1, int(chars / 4.0 * 0.5 + words * 0.75))


def summarize_conversation_chunk(
    messages_to_summarize: list[ChatMessage],
    *,
    existing_summary: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Call the raw LLM API to compress conversation history into a concise summary."""
    settings = get_settings()
    effective_api_key = settings.gemini_api_key if api_key is None else api_key
    effective_model = settings.gemini_model if model is None else model

    transcript_lines = []
    if existing_summary:
        transcript_lines.append(f"Previous Context: {existing_summary}")

    for msg in messages_to_summarize:
        sender = "User" if msg.role == "user" else "Assistant"
        transcript_lines.append(f"{sender}: {msg.content}")

    transcript = "\n".join(transcript_lines)

    prompt = (
        "You are an expert summarizer. Summarize the key facts, user preferences, "
        "names, and topics from the conversation below into 1-2 concise sentences. "
        "Do not include conversational filler.\n\n"
        f"Conversation:\n{transcript}\n\n"
        "Concise summary:"
    )

    endpoint_url = (
        f"{settings.gemini_base_url}/models/{effective_model}:generateContent"
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": effective_api_key,
    }
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ]
    }

    should_close_client = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close_client = True

    try:
        response = client.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return existing_summary or "Conversation in progress."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if "text" in p).strip()
    finally:
        if should_close_client:
            client.close()


class ManagedHistory:
    """Manages chat messages with token-threshold pruning (drop or summarize)."""

    def __init__(
        self,
        max_tokens: int = 150,
        strategy: Literal["drop", "summarize"] = "drop",
        system_instruction: str | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.strategy: Literal["drop", "summarize"] = strategy
        self.system_instruction = system_instruction
        self.messages: list[ChatMessage] = []
        self.summary: str | None = None
        self.eviction_events: list[str] = []

    def get_estimated_tokens(self) -> int:
        """Calculate total estimated tokens in current history and active summary."""
        total = 0
        if self.system_instruction:
            total += estimate_tokens(self.system_instruction)
        if self.summary:
            total += estimate_tokens(f"[Previous summary: {self.summary}]")
        for msg in self.messages:
            total += estimate_tokens(msg.content)
        return total

    def add_user_message(
        self,
        text: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Add user message and enforce token threshold constraint."""
        self.messages.append(ChatMessage(role="user", content=text))
        self.enforce_token_limit(api_key=api_key, model=model, client=client)

    def add_model_message(
        self,
        text: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Add model reply and enforce token threshold constraint."""
        self.messages.append(ChatMessage(role="model", content=text))
        self.enforce_token_limit(api_key=api_key, model=model, client=client)

    def enforce_token_limit(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Prune or summarize oldest turns if current tokens exceed max_tokens."""
        # Need at least 2 messages (1 turn) left in active context
        while self.get_estimated_tokens() > self.max_tokens and len(self.messages) > 2:
            if self.strategy == "drop":
                # Drop oldest pair (user + model or oldest message)
                evicted = self.messages.pop(0)
                event_desc = (
                    f"Dropped message from {evicted.role} "
                    f"({estimate_tokens(evicted.content)} tokens)"
                )
                self.eviction_events.append(event_desc)

            elif self.strategy == "summarize":
                # Take oldest 2 messages, summarize them, replace with summary
                to_compress = [self.messages.pop(0)]
                if self.messages and self.messages[0].role == "model":
                    to_compress.append(self.messages.pop(0))

                new_summary = summarize_conversation_chunk(
                    to_compress,
                    existing_summary=self.summary,
                    api_key=api_key,
                    model=model,
                    client=client,
                )
                self.summary = new_summary
                event_desc = (
                    f"Summarized {len(to_compress)} messages into: '{new_summary}'"
                )
                self.eviction_events.append(event_desc)

    def build_payload(self) -> dict[str, Any]:
        """Build Gemini contents array, injecting summary into context if present."""
        contents: list[dict[str, Any]] = []

        # If we have a summary from previous turns, inject it as initial context
        if self.summary:
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"[System Context Briefing: Here is a summary of the "
                                f"earlier conversation before pruning: {self.summary}]"
                            )
                        }
                    ],
                }
            )
            contents.append(
                {
                    "role": "model",
                    "parts": [
                        {"text": ("Understood. I will remember this context briefing.")}
                    ],
                }
            )

        for msg in self.messages:
            contents.append(
                {
                    "role": msg.role,
                    "parts": [{"text": msg.content}],
                }
            )

        payload: dict[str, Any] = {"contents": contents}

        if self.system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": self.system_instruction}]
            }

        return payload

    def reset(self) -> None:
        """Clear all conversation history and active summaries."""
        self.messages.clear()
        self.summary = None
        self.eviction_events.clear()


def execute_managed_turn(
    history: ManagedHistory,
    user_input: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    client: httpx.Client | None = None,
) -> tuple[str, int, int]:
    """Send turn with managed history.

    Returns (reply, prompt_tokens, candidate_tokens).
    """
    settings = get_settings()
    effective_api_key = settings.gemini_api_key if api_key is None else api_key
    effective_model = settings.gemini_model if model is None else model

    should_close_client = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close_client = True

    try:
        # Add user message with token threshold enforcement
        history.add_user_message(
            user_input, api_key=effective_api_key, model=effective_model, client=client
        )

        endpoint_url = (
            f"{settings.gemini_base_url}/models/{effective_model}:generateContent"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": effective_api_key,
        }
        payload = history.build_payload()

        response = client.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates in API response: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        reply_text = "".join(p.get("text", "") for p in parts if "text" in p)

        # Add model message with token threshold enforcement
        history.add_model_message(
            reply_text, api_key=effective_api_key, model=effective_model, client=client
        )

        usage = data.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        candidate_tokens = int(usage.get("candidatesTokenCount", 0))

        return reply_text, prompt_tokens, candidate_tokens
    finally:
        if should_close_client:
            client.close()


def main() -> None:
    """CLI to compare 'drop' vs 'summarize' history management strategies."""
    parser = argparse.ArgumentParser(
        description="Phase 2.4 - History Management: Drop vs Summarize"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["drop", "summarize"],
        default="summarize",
        help="History management strategy when exceeding token threshold",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Token threshold before pruning or summarizing occurs (e.g. 120)",
    )
    args = parser.parse_args()

    history = ManagedHistory(
        max_tokens=args.max_tokens,
        strategy=args.strategy,
        system_instruction="You are a helpful assistant.",
    )

    print("\n" + "=" * 65)
    print("PHASE 2.4: CONTEXT MANAGEMENT (DROP VS SUMMARIZE)")
    print("=" * 65)
    print(f"Strategy:       {args.strategy.upper()}")
    print(f"Max Tokens:     {args.max_tokens}")
    print("Type your message and press Enter. Type 'exit' to quit, '/reset' to clear.")
    print("=" * 65 + "\n")

    turn_count = 0

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "/exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            history.reset()
            turn_count = 0
            print("\n[History and summaries reset]\n")
            continue

        turn_count += 1
        history.eviction_events.clear()

        try:
            reply, prompt_tokens, candidate_tokens = execute_managed_turn(
                history, user_input
            )

            if history.eviction_events:
                print("\n" + "!" * 65)
                for event in history.eviction_events:
                    print(f"  [MANAGEMENT EVENT - {args.strategy.upper()}]: {event}")
                print("!" * 65)

            print(f"\nAI  > {reply.strip()}\n")
            print("-" * 65)
            print(
                f"Turn #{turn_count} Stats | "
                f"Active Messages: {len(history.messages)} | "
                f"Context Tokens: ~{history.get_estimated_tokens()} / {args.max_tokens}"
            )
            print(f"  • Prompt Tokens (Actual sent):    {prompt_tokens:>5}")
            print(f"  • Output Tokens:                   {candidate_tokens:>5}")
            if history.summary:
                print(f"  • Active Summary:                  '{history.summary}'")
            print("-" * 65 + "\n")
        except Exception as exc:
            print(f"\n[Error during turn: {exc}]\n", file=sys.stderr)


if __name__ == "__main__":
    main()
