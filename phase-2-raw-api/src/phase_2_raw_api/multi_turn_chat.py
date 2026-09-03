"""Task 2.3: Multi-turn CLI chat with manual message list and context tracking.

Manages conversation state explicitly in client code. Demonstrates how the context
window grows quadratically with each turn because the full message history must be
resent to the stateless LLM REST API.
"""

import argparse
import sys
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from phase_2_raw_api.config import get_settings


class ChatMessage(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "model", "system"] = Field(
        description="Sender role: 'user', 'model', or 'system'"
    )
    content: str = Field(description="Textual content of the message")


class TurnStats(BaseModel):
    """Token metrics and timing for a single conversation turn."""

    turn_index: int
    prompt_tokens: int
    candidate_tokens: int
    turn_total_tokens: int
    session_total_tokens: int
    history_message_count: int


class ConversationManager:
    """Manages the message list, payload formatting, and running token accounting."""

    def __init__(
        self,
        system_instruction: str | None = None,
        max_history_turns: int | None = None,
    ) -> None:
        self.system_instruction = system_instruction
        self.max_history_turns = max_history_turns
        self.messages: list[ChatMessage] = []
        self.turn_history: list[TurnStats] = []
        self.cumulative_tokens: int = 0

    def add_user_message(self, text: str) -> None:
        """Append a user message to the history."""
        self.messages.append(ChatMessage(role="user", content=text))
        self._apply_pruning()

    def add_model_message(self, text: str) -> None:
        """Append a model response to the history."""
        self.messages.append(ChatMessage(role="model", content=text))
        self._apply_pruning()

    def _apply_pruning(self) -> None:
        """Apply sliding window pruning if maximum turn limit is exceeded."""
        if self.max_history_turns is None:
            return

        # Each turn consists of 1 user message + 1 model message (2 ChatMessages)
        max_messages = self.max_history_turns * 2
        if len(self.messages) > max_messages:
            # Retain the most recent messages
            self.messages = self.messages[-max_messages:]

    def build_payload(self) -> dict[str, Any]:
        """Convert message history into Gemini API contents payload format."""
        contents: list[dict[str, Any]] = []
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

    def record_turn(self, prompt_tokens: int, candidate_tokens: int) -> TurnStats:
        """Update token ledger and return turn statistics."""
        turn_total = prompt_tokens + candidate_tokens
        self.cumulative_tokens += turn_total

        stats = TurnStats(
            turn_index=len(self.turn_history) + 1,
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            turn_total_tokens=turn_total,
            session_total_tokens=self.cumulative_tokens,
            history_message_count=len(self.messages),
        )
        self.turn_history.append(stats)
        return stats

    def reset(self) -> None:
        """Clear all conversation history and reset token counters."""
        self.messages.clear()
        self.turn_history.clear()
        self.cumulative_tokens = 0


def send_chat_turn(
    conversation: ConversationManager,
    user_input: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> tuple[str, TurnStats]:
    """Execute a single conversation turn against the raw LLM REST API."""
    settings = get_settings()
    effective_api_key = settings.gemini_api_key if api_key is None else api_key
    effective_model = settings.gemini_model if model is None else model

    if not effective_api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY in .env or pass api_key."
        )

    # 1. Append user's new message to the managed state list
    conversation.add_user_message(user_input)

    # 2. Build the full multi-turn wire payload
    endpoint_url = (
        f"{settings.gemini_base_url}/models/{effective_model}:generateContent"
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": effective_api_key,
    }
    payload = conversation.build_payload()

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
            raise ValueError(f"No candidates returned in API response: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        reply_text = "".join(p.get("text", "") for p in parts if "text" in p)

        # 3. Append model's reply to the managed state list
        conversation.add_model_message(reply_text)

        # 4. Extract token counts and update running ledger
        usage = data.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        candidate_tokens = int(usage.get("candidatesTokenCount", 0))
        stats = conversation.record_turn(prompt_tokens, candidate_tokens)

        return reply_text, stats
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        err_msg = exc.response.text
        raise RuntimeError(
            f"Chat API request failed with HTTP {status}: {err_msg}"
        ) from exc
    finally:
        if should_close_client:
            client.close()


def main() -> None:
    """Interactive multi-turn CLI chat loop demonstrating context window growth."""
    parser = argparse.ArgumentParser(
        description="Phase 2.3 - Multi-turn CLI Chat with Context & Token Tracking"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful, concise AI technical mentor.",
        help="Optional system prompt instruction",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Sliding window turn limit (prunes oldest turns beyond limit)",
    )
    args = parser.parse_args()

    conversation = ConversationManager(
        system_instruction=args.system, max_history_turns=args.max_turns
    )

    print("\n" + "=" * 65)
    print("PHASE 2.3: MULTI-TURN CLI CHAT (MANUAL STATE MANAGEMENT)")
    print("=" * 65)
    print(f"System: {args.system}")
    if args.max_turns:
        print(f"Sliding Window: Retaining last {args.max_turns} turns")
    print("Commands: '/reset' to clear history | '/exit' or 'exit' to quit\n")
    print("Watch the 'Prompt Tokens' increase on every turn as context grows!")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "exit", "quit", "/quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            conversation.reset()
            print("\n[Conversation history and token counters reset to 0]\n")
            continue

        try:
            reply, stats = send_chat_turn(conversation, user_input)
            print(f"\nAI  > {reply.strip()}\n")
            print("-" * 65)
            print(
                f"Turn #{stats.turn_index} Token Metrics | "
                f"History Messages: {stats.history_message_count}"
            )
            print(
                f"  • Prompt Tokens (Context sent):    {stats.prompt_tokens:>5} "
                f"(Sent all previous turns!)"
            )
            print(f"  • Output Tokens (Model reply):     {stats.candidate_tokens:>5}")
            print(f"  • Turn Total Tokens:               {stats.turn_total_tokens:>5}")
            print(
                f"  • Cumulative Session Tokens:       {stats.session_total_tokens:>5}"
            )
            print("-" * 65 + "\n")
        except Exception as exc:
            print(f"\n[Error during chat turn: {exc}]\n", file=sys.stderr)


if __name__ == "__main__":
    main()
