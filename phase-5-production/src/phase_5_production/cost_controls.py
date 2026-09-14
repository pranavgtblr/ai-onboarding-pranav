"""Task 5.7: Cost Controls for Autonomous Agents.

Provides:
1. `PromptCacheManager`: Static prompt prefix caching with token discounts.
2. `TokenBudgetManager`: Per-conversation token budget tracking & alerts.
3. `TieredModelRouter`: Asymmetric routing (small for routing, large for synthesis).
4. `CostControlledAgent`: End-to-end multi-turn agent enforcing cost controls.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("phase_5_production.cost_controls")

# Standard production pricing ($ / 1 Million Tokens)
SMALL_MODEL_INPUT_PRICE = 0.15
SMALL_MODEL_OUTPUT_PRICE = 0.60
LARGE_MODEL_INPUT_PRICE = 1.25
LARGE_MODEL_OUTPUT_PRICE = 5.00
PROMPT_CACHE_DISCOUNT = 0.75  # 75% discount on cached input tokens


class BudgetAlertLevel(str, Enum):
    """Alert severity tiers for token budget consumption."""

    INFO = "INFO"
    WARNING = "WARNING"
    EXCEEDED = "EXCEEDED"


class BudgetExceededError(Exception):
    """Raised when conversation token consumption breaches the hard budget."""


@dataclass
class BudgetAlert:
    """Alert event emitted when conversation token usage crosses threshold."""

    level: BudgetAlertLevel
    conversation_id: str
    tokens_used: int
    budget_limit: int
    percent_used: float
    cost_usd: float
    message: str


@dataclass
class TokenUsage:
    """Detailed token consumption metrics for a single operation."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cached_tokens + self.output_tokens


class PromptCacheManager:
    """Manages static prompt prefix caching and applies cached token discounts.

    Separates invariant system instructions and tool definitions from dynamic
    turn content, recording cache hits and computing provider pricing discounts.
    """

    def __init__(self, discount_rate: float = PROMPT_CACHE_DISCOUNT) -> None:
        self.discount_rate = discount_rate
        self.cache_store: dict[str, int] = {}  # prefix_hash -> token_count
        self.total_cache_reads: int = 0
        self.total_cache_hits: int = 0
        self.total_cached_tokens_served: int = 0
        self.total_uncached_tokens_served: int = 0

    def get_or_set_prefix(
        self,
        static_prefix: str,
        token_count: int | None = None,
    ) -> tuple[bool, int]:
        """Checks if static prefix is cached.

        Returns (is_cache_hit, token_count).
        """
        prefix_hash = hashlib.sha256(static_prefix.encode("utf-8")).hexdigest()
        self.total_cache_reads += 1

        if token_count is None:
            token_count = max(1, len(static_prefix) // 4)

        if prefix_hash in self.cache_store:
            self.total_cache_hits += 1
            self.total_cached_tokens_served += token_count
            return True, token_count

        # Cache Miss: Write to cache
        self.cache_store[prefix_hash] = token_count
        self.total_uncached_tokens_served += token_count
        return False, token_count

    def calculate_cost(
        self,
        base_rate_per_1m: float,
        uncached_tokens: int,
        cached_tokens: int,
    ) -> float:
        """Computes input token cost applying prompt cache discount."""
        full_rate = base_rate_per_1m / 1_000_000.0
        cached_rate = full_rate * (1.0 - self.discount_rate)
        return (uncached_tokens * full_rate) + (cached_tokens * cached_rate)

    @property
    def hit_rate(self) -> float:
        if self.total_cache_reads == 0:
            return 0.0
        return self.total_cache_hits / self.total_cache_reads


class TokenBudgetManager:
    """Tracks per-conversation token usage and emits proactive threshold alerts."""

    def __init__(
        self,
        conversation_id: str,
        budget_limit: int = 8000,
        warning_threshold: float = 0.75,
        alert_callback: Callable[[BudgetAlert], None] | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.budget_limit = budget_limit
        self.warning_threshold = warning_threshold
        self.alert_callback = alert_callback

        self.input_tokens: int = 0
        self.cached_tokens: int = 0
        self.output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.warning_emitted: bool = False
        self.exceeded_emitted: bool = False
        self.alerts: list[BudgetAlert] = []

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cached_tokens + self.output_tokens

    @property
    def percent_used(self) -> float:
        if self.budget_limit <= 0:
            return 0.0
        return (self.total_tokens / self.budget_limit) * 100.0

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> list[BudgetAlert]:
        """Records token consumption and evaluates budget threshold alerts."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens
        self.total_cost_usd += cost_usd

        new_alerts: list[BudgetAlert] = []
        fraction = self.total_tokens / self.budget_limit

        # Warning threshold (75%)
        if fraction >= self.warning_threshold and not self.warning_emitted:
            self.warning_emitted = True
            alert = BudgetAlert(
                level=BudgetAlertLevel.WARNING,
                conversation_id=self.conversation_id,
                tokens_used=self.total_tokens,
                budget_limit=self.budget_limit,
                percent_used=self.percent_used,
                cost_usd=self.total_cost_usd,
                message=(
                    f"Conversation {self.conversation_id} has consumed "
                    f"{self.total_tokens}/{self.budget_limit} tokens "
                    f"({self.percent_used:.1f}% of budget)."
                ),
            )
            self.alerts.append(alert)
            new_alerts.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        # Exceeded threshold (100%)
        if fraction >= 1.0 and not self.exceeded_emitted:
            self.exceeded_emitted = True
            alert = BudgetAlert(
                level=BudgetAlertLevel.EXCEEDED,
                conversation_id=self.conversation_id,
                tokens_used=self.total_tokens,
                budget_limit=self.budget_limit,
                percent_used=self.percent_used,
                cost_usd=self.total_cost_usd,
                message=(
                    f"Conversation {self.conversation_id} token budget EXCEEDED: "
                    f"{self.total_tokens} tokens used (Limit: {self.budget_limit})."
                ),
            )
            self.alerts.append(alert)
            new_alerts.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        return new_alerts


class TieredModelRouter:
    """Dispatches classification to small model and synthesis to large model."""

    def __init__(
        self,
        small_model_name: str = "gemini-1.5-flash",
        large_model_name: str = "gemini-1.5-pro",
    ) -> None:
        self.small_model_name = small_model_name
        self.large_model_name = large_model_name
        self.small_model_calls: int = 0
        self.large_model_calls: int = 0

    def classify_and_route(
        self,
        query: str,
        prompt_cache: PromptCacheManager,
    ) -> tuple[dict[str, Any], TokenUsage]:
        """Uses small model for low-cost intent classification & routing."""
        self.small_model_calls += 1
        q_lower = query.lower()

        # Check static routing prompt in cache
        static_router_prompt = (
            "You are an intent router. Classify user queries into DIRECT_ANSWER "
            "or TOOL_CALL with target tool."
        )
        is_hit, prefix_tokens = prompt_cache.get_or_set_prefix(static_router_prompt)

        # Token estimation for classification turn
        dynamic_tokens = max(1, len(query) // 4)
        out_tokens = 25  # short routing decision

        if is_hit:
            in_cost = prompt_cache.calculate_cost(
                SMALL_MODEL_INPUT_PRICE,
                uncached_tokens=dynamic_tokens,
                cached_tokens=prefix_tokens,
            )
            cached_count = prefix_tokens
            uncached_count = dynamic_tokens
        else:
            in_cost = prompt_cache.calculate_cost(
                SMALL_MODEL_INPUT_PRICE,
                uncached_tokens=prefix_tokens + dynamic_tokens,
                cached_tokens=0,
            )
            cached_count = 0
            uncached_count = prefix_tokens + dynamic_tokens

        out_cost = (out_tokens * SMALL_MODEL_OUTPUT_PRICE) / 1_000_000.0
        usage = TokenUsage(
            input_tokens=uncached_count,
            cached_tokens=cached_count,
            output_tokens=out_tokens,
            cost_usd=in_cost + out_cost,
        )

        # Classification heuristics
        if any(g in q_lower for g in ("hello", "hi", "hey", "help", "thanks")):
            return {"action": "direct_answer", "tool": None}, usage
        if "customer" in q_lower or "order" in q_lower or "sql" in q_lower:
            return {"action": "tool_call", "tool": "db_query"}, usage
        if "pdf" in q_lower or "airlock" in q_lower or "spec" in q_lower:
            return {"action": "tool_call", "tool": "pdf_search"}, usage
        if "site" in q_lower or "artemis" in q_lower or "launch" in q_lower:
            return {"action": "tool_call", "tool": "site_search"}, usage
        if "weather" in q_lower:
            return {"action": "tool_call", "tool": "get_weather"}, usage

        return {"action": "tool_call", "tool": "pdf_search"}, usage

    def synthesize(
        self,
        query: str,
        context: str,
        prompt_cache: PromptCacheManager,
        force_large_model: bool = True,
    ) -> tuple[str, TokenUsage]:
        """Uses the large model exclusively for synthesis across retrieved context."""
        static_synth_prompt = (
            "You are a high-reasoning synthesis engine. Synthesize retrieved facts "
            "into a comprehensive, grounded answer citing sources."
        )
        is_hit, prefix_tokens = prompt_cache.get_or_set_prefix(static_synth_prompt)

        dynamic_tokens = max(1, (len(query) + len(context)) // 4)
        out_tokens = 150  # comprehensive synthesis output

        if force_large_model:
            self.large_model_calls += 1
            input_rate = LARGE_MODEL_INPUT_PRICE
            output_rate = LARGE_MODEL_OUTPUT_PRICE
        else:
            self.small_model_calls += 1
            input_rate = SMALL_MODEL_INPUT_PRICE
            output_rate = SMALL_MODEL_OUTPUT_PRICE

        if is_hit:
            in_cost = prompt_cache.calculate_cost(
                input_rate,
                uncached_tokens=dynamic_tokens,
                cached_tokens=prefix_tokens,
            )
            cached_count = prefix_tokens
            uncached_count = dynamic_tokens
        else:
            in_cost = prompt_cache.calculate_cost(
                input_rate,
                uncached_tokens=prefix_tokens + dynamic_tokens,
                cached_tokens=0,
            )
            cached_count = 0
            uncached_count = prefix_tokens + dynamic_tokens

        out_cost = (out_tokens * output_rate) / 1_000_000.0
        usage = TokenUsage(
            input_tokens=uncached_count,
            cached_tokens=cached_count,
            output_tokens=out_tokens,
            cost_usd=in_cost + out_cost,
        )

        chosen_model = (
            self.large_model_name if force_large_model else self.small_model_name
        )
        answer = (
            f"[Synthesized via {chosen_model}]: Based on retrieved context "
            f"({len(context)} chars), the answer to '{query}' is verified."
        )
        return answer, usage


@dataclass
class AgentTurnResult:
    """Result of a single agent conversation turn."""

    turn_number: int
    query: str
    response: str
    model_used: str
    is_cache_hit: bool
    turn_usage: TokenUsage
    cumulative_tokens: int
    cumulative_cost_usd: float
    alerts: list[BudgetAlert] = field(default_factory=list)


class CostControlledAgent:
    """Orchestrates agent pipeline with prompt caching, tiered routing, and budget."""

    def __init__(
        self,
        conversation_id: str,
        budget_limit: int = 8000,
        enable_caching: bool = True,
        enable_tiered_routing: bool = True,
    ) -> None:
        self.conversation_id = conversation_id
        self.enable_caching = enable_caching
        self.enable_tiered_routing = enable_tiered_routing

        self.prompt_cache = PromptCacheManager()
        self.budget_manager = TokenBudgetManager(
            conversation_id=conversation_id,
            budget_limit=budget_limit,
        )
        self.router = TieredModelRouter()
        self.turn_history: list[AgentTurnResult] = []

    def execute_turn(self, query: str, retrieved_context: str = "") -> AgentTurnResult:
        """Executes a single user conversation turn with cost controls."""
        turn_num = len(self.turn_history) + 1

        # 1. Check if budget already exceeded
        if self.budget_manager.percent_used >= 100.0:
            raise BudgetExceededError(
                f"Conversation token budget of {self.budget_manager.budget_limit} "
                "tokens exceeded. Execution halted."
            )

        # 2. Classification & Tool Routing
        decision, route_usage = self.router.classify_and_route(
            query,
            self.prompt_cache,
        )

        # 3. Synthesis vs Direct Answer
        if decision["action"] == "direct_answer":
            answer = "Hello! How can I assist you with your queries today?"
            synth_usage = TokenUsage()
            active_model = self.router.small_model_name
        else:
            # Context synthesis: use large model if tiered routing enabled
            use_large = self.enable_tiered_routing and bool(retrieved_context)
            synth_answer, synth_usage = self.router.synthesize(
                query,
                retrieved_context or "Mock retrieved knowledge context.",
                self.prompt_cache,
                force_large_model=use_large,
            )
            answer = synth_answer
            active_model = (
                self.router.large_model_name
                if use_large
                else self.router.small_model_name
            )

        # Combine step usage
        total_in = route_usage.input_tokens + synth_usage.input_tokens
        total_cached = route_usage.cached_tokens + synth_usage.cached_tokens
        total_out = route_usage.output_tokens + synth_usage.output_tokens
        total_cost = route_usage.cost_usd + synth_usage.cost_usd

        turn_usage = TokenUsage(
            input_tokens=total_in,
            cached_tokens=total_cached,
            output_tokens=total_out,
            cost_usd=total_cost,
        )

        # 4. Record to Budget Manager & check for alerts
        alerts = self.budget_manager.record_usage(
            input_tokens=total_in,
            output_tokens=total_out,
            cached_tokens=total_cached,
            cost_usd=total_cost,
        )

        result = AgentTurnResult(
            turn_number=turn_num,
            query=query,
            response=answer,
            model_used=active_model,
            is_cache_hit=total_cached > 0,
            turn_usage=turn_usage,
            cumulative_tokens=self.budget_manager.total_tokens,
            cumulative_cost_usd=self.budget_manager.total_cost_usd,
            alerts=alerts,
        )
        self.turn_history.append(result)
        return result
