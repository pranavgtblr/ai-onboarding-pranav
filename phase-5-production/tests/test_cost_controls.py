"""Tests for Task 5.7: Cost Controls (Prompt Caching, Tiered Routing, Token Budget)."""

from __future__ import annotations

import pytest

from phase_5_production.cost_controls import (
    BudgetAlert,
    BudgetAlertLevel,
    BudgetExceededError,
    CostControlledAgent,
    PromptCacheManager,
    TieredModelRouter,
    TokenBudgetManager,
)

# =====================================================================
# PART 1: Prompt Caching Tests
# =====================================================================


def test_prompt_cache_records_write_then_hit() -> None:
    """Turn 1 must write to cache (miss); subsequent identical calls must hit."""
    cache = PromptCacheManager(discount_rate=0.75)
    static_prefix = (
        "You are an enterprise assistant. Answer truthfully based on context."
    )

    is_hit_1, tokens_1 = cache.get_or_set_prefix(static_prefix, token_count=100)
    assert is_hit_1 is False
    assert tokens_1 == 100
    assert cache.total_cache_hits == 0
    assert cache.total_cache_reads == 1

    is_hit_2, tokens_2 = cache.get_or_set_prefix(static_prefix, token_count=100)
    assert is_hit_2 is True
    assert tokens_2 == 100
    assert cache.total_cache_hits == 1
    assert cache.total_cache_reads == 2
    assert cache.hit_rate == 0.5


def test_prompt_cache_cost_discount() -> None:
    """Cached tokens must receive the configured discount (e.g. 75%)."""
    cache = PromptCacheManager(discount_rate=0.75)
    base_rate = 1.0  # $1.00 per 1M tokens

    # Uncached 1M tokens = $1.00
    uncached_cost = cache.calculate_cost(
        base_rate,
        uncached_tokens=1_000_000,
        cached_tokens=0,
    )
    assert uncached_cost == pytest.approx(1.00, rel=1e-5)

    # Cached 1M tokens with 75% discount = $0.25
    cached_cost = cache.calculate_cost(
        base_rate,
        uncached_tokens=0,
        cached_tokens=1_000_000,
    )
    assert cached_cost == pytest.approx(0.25, rel=1e-5)


# =====================================================================
# PART 2: Token Budget & Threshold Alerting Tests
# =====================================================================


def test_token_budget_normal_usage_no_alerts() -> None:
    """Usage below 75% threshold produces no alerts."""
    captured_alerts: list[BudgetAlert] = []
    manager = TokenBudgetManager(
        conversation_id="conv-1",
        budget_limit=1000,
        warning_threshold=0.75,
        alert_callback=captured_alerts.append,
    )

    alerts = manager.record_usage(input_tokens=300, output_tokens=200)
    assert len(alerts) == 0
    assert len(captured_alerts) == 0
    assert manager.total_tokens == 500
    assert manager.percent_used == 50.0


def test_token_budget_emits_warning_alert_at_75_percent() -> None:
    """Usage reaching or crossing 75% must emit a single WARNING alert."""
    captured_alerts: list[BudgetAlert] = []
    manager = TokenBudgetManager(
        conversation_id="conv-2",
        budget_limit=1000,
        warning_threshold=0.75,
        alert_callback=captured_alerts.append,
    )

    # Advance to 760 tokens (76%)
    alerts = manager.record_usage(input_tokens=500, output_tokens=260)
    assert len(alerts) == 1
    assert alerts[0].level == BudgetAlertLevel.WARNING
    assert alerts[0].tokens_used == 760
    assert alerts[0].percent_used == 76.0

    # Next call remains above 75% but does not re-emit duplicate warning
    alerts_2 = manager.record_usage(input_tokens=50, output_tokens=50)
    assert len(alerts_2) == 0
    assert len(captured_alerts) == 1


def test_token_budget_emits_exceeded_alert_at_100_percent() -> None:
    """Usage reaching 100% must emit an EXCEEDED alert."""
    captured_alerts: list[BudgetAlert] = []
    manager = TokenBudgetManager(
        conversation_id="conv-3",
        budget_limit=1000,
        alert_callback=captured_alerts.append,
    )

    # Step 1: cross warning threshold
    manager.record_usage(input_tokens=600, output_tokens=200)  # 800 tokens (80%)
    assert len(captured_alerts) == 1
    assert captured_alerts[0].level == BudgetAlertLevel.WARNING

    # Step 2: cross budget limit
    alerts = manager.record_usage(input_tokens=150, output_tokens=100)  # 1050 tokens
    assert len(alerts) == 1
    assert alerts[0].level == BudgetAlertLevel.EXCEEDED
    assert alerts[0].tokens_used == 1050
    assert len(captured_alerts) == 2


# =====================================================================
# PART 3: Tiered Model Routing Tests
# =====================================================================


def test_tiered_router_greetings_only_use_small_model() -> None:
    """Greeting queries are handled without invoking the large model."""
    router = TieredModelRouter()
    cache = PromptCacheManager()

    decision, usage = router.classify_and_route("Hello there!", cache)
    assert decision["action"] == "direct_answer"
    assert router.small_model_calls == 1
    assert router.large_model_calls == 0
    assert usage.output_tokens > 0


def test_tiered_router_synthesis_invokes_large_model() -> None:
    """Complex context synthesis uses the large reasoning model."""
    router = TieredModelRouter()
    cache = PromptCacheManager()

    query = "What is the depressurization cycle duration?"
    context = "Nominal depressurization cycle requires 180 seconds under staging."
    answer, usage = router.synthesize(
        query,
        context,
        cache,
        force_large_model=True,
    )

    assert "[Synthesized via gemini-1.5-pro]" in answer
    assert router.large_model_calls == 1
    assert usage.cost_usd > 0.0


# =====================================================================
# PART 4: End-to-End Cost Controlled Agent Tests
# =====================================================================


def test_cost_controlled_agent_multi_turn_caching() -> None:
    """Multi-turn conversation must leverage cached prompt prefixes on turns 2+."""
    agent = CostControlledAgent(
        conversation_id="test-conv-e2e",
        budget_limit=5000,
    )

    # Turn 1: Cache Miss (Populates static prefix)
    turn_1 = agent.execute_turn(
        "Find nominal airlock cycle specs",
        retrieved_context="180 seconds nominal cycle time.",
    )
    assert turn_1.turn_number == 1
    assert turn_1.is_cache_hit is False

    # Turn 2: Cache Hit on shared system/tool prefixes
    turn_2 = agent.execute_turn(
        "What about emergency decompression?",
        retrieved_context="Emergency dump takes 45 seconds.",
    )
    assert turn_2.turn_number == 2
    assert turn_2.is_cache_hit is True
    assert turn_2.turn_usage.cached_tokens > 0
    assert turn_2.cumulative_tokens > turn_1.cumulative_tokens


def test_cost_controlled_agent_enforces_budget_hard_cap() -> None:
    """Agent must halt and raise BudgetExceededError when budget is exhausted."""
    agent = CostControlledAgent(
        conversation_id="tight-budget-conv",
        budget_limit=300,  # small budget to test enforcement
    )

    # Turn 1 consumes ~180-210 tokens
    turn_1 = agent.execute_turn("Query 1", retrieved_context="Some initial data.")
    assert turn_1.turn_number == 1

    # Turn 2 pushes total tokens over 300 (exceeding budget limit)
    turn_2 = agent.execute_turn("Query 2", retrieved_context="Some more data.")
    assert turn_2.turn_number == 2
    assert agent.budget_manager.percent_used >= 100.0

    # Turn 3 must be blocked with BudgetExceededError
    with pytest.raises(BudgetExceededError, match="token budget of 300 tokens"):
        agent.execute_turn("Query 3", retrieved_context="Blocked.")
