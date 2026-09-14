"""Task 5.8: Reliability Engineering for Autonomous Agents.

Provides:
1. `ReliableFallbackModel`: Transparently falls back to a secondary LLM provider
   when the primary provider encounters rate limits (HTTP 429) or outages (500/503).
2. `GracefulDegradationHandler`: Gracefully degrades when retrieval tools return
   nothing, preventing hallucinations and infinite retry loops.
3. `ReliableAgent`: Integrated agent pipeline combining provider fallback and
   retrieval degradation handling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger("phase_5_production.reliability")

# Recognizable error substrings and types for rate limits & outages
TRANSIENT_OUTAGE_INDICATORS = (
    "503",
    "500",
    "502",
    "504",
    "service unavailable",
    "overloaded",
    "internal server error",
    "connection error",
    "timeout",
    "timed out",
    "bad gateway",
)

RATE_LIMIT_INDICATORS = (
    "429",
    "rate limit",
    "resource exhausted",
    "quota exceeded",
    "too many requests",
)


@dataclass
class ProviderFailoverEvent:
    """Telemetry record captured whenever a provider failover occurs."""

    timestamp: str
    primary_provider: str
    fallback_provider: str
    error_type: str
    error_message: str
    reason: str  # "rate_limit" or "outage"


@dataclass
class DegradationNotice:
    """Structured notice when retrieval yields no matches."""

    searched_entity_or_topic: str
    source: str
    message: str
    suggested_alternatives: list[str]


class ReliableFallbackModel:
    """Wraps primary and secondary chat models with automatic failover.

    Guarantees:
    - If primary model raises 429/RateLimit, failover to secondary.
    - If primary model raises 500/503/Outage, failover to secondary.
    - Emits structured ProviderFailoverEvent for observability.
    """

    def __init__(
        self,
        primary_model: BaseChatModel | Callable[..., Any],
        fallback_model: BaseChatModel | Callable[..., Any],
        primary_name: str = "primary-provider",
        fallback_name: str = "secondary-provider",
        failover_callback: Callable[[ProviderFailoverEvent], None] | None = None,
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.failover_callback = failover_callback

        self.failover_events: list[ProviderFailoverEvent] = []
        self.total_invocations: int = 0
        self.primary_successes: int = 0
        self.fallback_successes: int = 0

    @property
    def failover_count(self) -> int:
        return len(self.failover_events)

    def _is_transient_or_rate_limit(self, exc: Exception) -> tuple[bool, str]:
        """Classifies if exception is a rate limit or transient service outage."""
        err_msg = str(exc).lower()
        err_type = type(exc).__name__.lower()

        for ind in RATE_LIMIT_INDICATORS:
            if ind in err_msg or ind in err_type:
                return True, "rate_limit"

        for ind in TRANSIENT_OUTAGE_INDICATORS:
            if ind in err_msg or ind in err_type:
                return True, "outage"

        return False, "unknown"

    def _invoke_model(
        self,
        model: BaseChatModel | Callable[..., Any],
        messages: list[BaseMessage] | list[Any],
        **kwargs: Any,
    ) -> Any:
        """Helper to invoke either a LangChain BaseChatModel or a callable mock."""
        if hasattr(model, "invoke"):
            invoker = getattr(model, "invoke")
            return invoker(messages, **kwargs)
        if callable(model):
            return model(messages, **kwargs)
        raise TypeError("Model must be a BaseChatModel or Callable.")

    def invoke(self, messages: list[BaseMessage] | list[Any], **kwargs: Any) -> Any:
        """Executes primary model with automatic failover to fallback model."""
        self.total_invocations += 1

        try:
            res = self._invoke_model(self.primary_model, messages, **kwargs)
            self.primary_successes += 1
            return res
        except Exception as primary_exc:
            is_recoverable, reason = self._is_transient_or_rate_limit(primary_exc)
            if not is_recoverable:
                # Non-transient error (e.g. fatal syntax error or bad parameter)
                logger.error(
                    f"Non-recoverable error in primary {self.primary_name}: "
                    f"{primary_exc}"
                )
                raise primary_exc

            event = ProviderFailoverEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                primary_provider=self.primary_name,
                fallback_provider=self.fallback_name,
                error_type=type(primary_exc).__name__,
                error_message=str(primary_exc),
                reason=reason,
            )
            self.failover_events.append(event)
            logger.warning(
                f"🚨 PROVIDER FAILOVER TRIGGERED: Primary '{self.primary_name}' "
                f"failed with {reason.upper()} ({type(primary_exc).__name__}: "
                f"{primary_exc}). Switching immediately to '{self.fallback_name}'."
            )
            if self.failover_callback:
                self.failover_callback(event)

            # Execute on fallback model
            try:
                fallback_res = self._invoke_model(
                    self.fallback_model, messages, **kwargs
                )
                self.fallback_successes += 1
                return fallback_res
            except Exception as fallback_exc:
                logger.critical(
                    f"Both primary ({self.primary_name}) and fallback "
                    f"({self.fallback_name}) failed! Primary: {primary_exc} | "
                    f"Fallback: {fallback_exc}"
                )
                raise fallback_exc


class GracefulDegradationHandler:
    """Detects and gracefully degrades when retrieval tools return nothing.

    Prevents model hallucinations by enforcing truthful degradation messages
    specifying what was searched and suggesting adjacent alternatives.
    """

    EMPTY_RETRIEVAL_MARKERS = (
        "rows returned: 0",
        "no matching records found",
        "no relevant chunks found",
        "0 results",
        "no matches",
        "empty result",
        "[]",
    )

    @classmethod
    def is_empty_retrieval(cls, tool_output: str) -> bool:
        """Determines if tool output indicates zero matches or empty results."""
        if not tool_output or not tool_output.strip():
            return True
        output_lower = tool_output.lower()
        return any(marker in output_lower for marker in cls.EMPTY_RETRIEVAL_MARKERS)

    @classmethod
    def create_degradation_response(
        cls,
        query: str,
        tool_name: str,
        tool_output: str,
    ) -> DegradationNotice:
        """Builds a grounded, helpful degradation notice for empty retrieval."""
        q_clean = query.strip().rstrip("?").strip()

        alternatives = [
            "Verify the spelling or search terms (e.g. check city name or ID).",
            "Try broadening your query keywords.",
            "Contact support or administrator to verify data indexing status.",
        ]

        if tool_name == "db_query":
            message = (
                f"No database records matched your search query '{q_clean}'. "
                f"The database returned 0 results."
            )
            alternatives = [
                "Search with broader city names or customer name fragments.",
                "List available customer records using 'show all customers'.",
                "Verify that the customer or order ID exists in the database.",
            ]
        elif tool_name == "pdf_search":
            message = (
                f"No documentation chunks matched '{q_clean}' in our index. "
                "The search returned no relevant passages."
            )
            alternatives = [
                "Search for broader topic terms (e.g. 'airlock specs').",
                "Review the table of contents or available chapter headings.",
            ]
        elif tool_name == "site_search":
            message = (
                f"No scraped website content matched '{q_clean}'. "
                "The site index contains no relevant sections."
            )
        else:
            message = (
                f"Search for '{q_clean}' returned no results across available sources."
            )

        return DegradationNotice(
            searched_entity_or_topic=q_clean,
            source=tool_name,
            message=message,
            suggested_alternatives=alternatives,
        )

    @classmethod
    def format_grounded_answer(cls, notice: DegradationNotice) -> str:
        """Formats the degradation notice into a safe, non-hallucinated response."""
        alt_bullets = "\n".join(f"  • {alt}" for alt in notice.suggested_alternatives)
        return (
            f"ℹ️ **Notice: Information Not Found**\n\n"
            f"{notice.message}\n\n"
            f"To help locate what you need, you might try:\n"
            f"{alt_bullets}\n\n"
            f"*(Note: To maintain data integrity, the assistant does not generate "
            f"speculative or unverified answers when retrieval yields zero matches.)*"
        )


@dataclass
class ReliableAgentExecutionResult:
    """Telemetry and outcome of a reliable agent query execution."""

    query: str
    response: str
    active_provider: str
    failover_occurred: bool
    failover_reason: str | None
    is_degraded: bool
    degradation_notice: DegradationNotice | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)


class ReliableAgent:
    """Agent that integrates provider fallback and graceful retrieval degradation."""

    def __init__(
        self,
        fallback_model: ReliableFallbackModel,
        tools_map: dict[str, Callable[[str], str]] | None = None,
    ) -> None:
        self.fallback_model = fallback_model
        self.tools_map = tools_map or {}
        self.degradation_handler = GracefulDegradationHandler()

    def run_query(
        self,
        query: str,
        tool_name: str | None = None,
        tool_arg: str | None = None,
    ) -> ReliableAgentExecutionResult:
        """Executes query with provider failover and graceful retrieval degradation."""
        failover_before = self.fallback_model.failover_count

        # Step 1: Execute tool if applicable
        tool_output = ""
        is_degraded = False
        degradation_notice: DegradationNotice | None = None

        if tool_name and tool_name in self.tools_map:
            arg = tool_arg if tool_arg is not None else query
            tool_output = self.tools_map[tool_name](arg)

            # Check for empty retrieval
            if self.degradation_handler.is_empty_retrieval(tool_output):
                is_degraded = True
                degradation_notice = (
                    self.degradation_handler.create_degradation_response(
                        query=query,
                        tool_name=tool_name,
                        tool_output=tool_output,
                    )
                )

        # Step 2: Synthesis
        if is_degraded and degradation_notice is not None:
            # Graceful degradation response: 100% grounded, no hallucination
            final_response = self.degradation_handler.format_grounded_answer(
                degradation_notice
            )
            provider_used = self.fallback_model.primary_name
            failover_happened = False
            failover_reason = None
        else:
            # Prepare synthesis messages
            prompt = (
                f"User Question: {query}\n"
                f"Retrieved Context: {tool_output or 'General conversation.'}\n"
                f"Provide a helpful, accurate answer based strictly on the context."
            )
            raw_response = self.fallback_model.invoke([AIMessage(content=prompt)])
            final_response = (
                raw_response.content
                if hasattr(raw_response, "content")
                else str(raw_response)
            )

            failover_after = self.fallback_model.failover_count
            failover_happened = failover_after > failover_before
            if failover_happened:
                provider_used = self.fallback_model.fallback_name
                failover_reason = self.fallback_model.failover_events[-1].reason
            else:
                provider_used = self.fallback_model.primary_name
                failover_reason = None

        return ReliableAgentExecutionResult(
            query=query,
            response=final_response,
            active_provider=provider_used,
            failover_occurred=failover_happened,
            failover_reason=failover_reason,
            is_degraded=is_degraded,
            degradation_notice=degradation_notice,
            telemetry={
                "total_failovers": self.fallback_model.failover_count,
                "primary_successes": self.fallback_model.primary_successes,
                "fallback_successes": self.fallback_model.fallback_successes,
            },
        )
