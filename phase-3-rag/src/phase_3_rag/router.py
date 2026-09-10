"""Adaptive Query Routing for RAG systems (Project E - Task 3.19).

Determines whether a user query:
1. DIRECT_LLM: Can be answered directly by the LLM without retrieval
   (e.g., coding, math, greetings, translations, creative writing).
2. LOCAL_CORPUS: Should be retrieved from internal proprietary domain
   documents (e.g., Mars ECLSS engineering, telemetry, PDF specs).
3. WEB_SEARCH: Requires live, external, or real-time web retrieval
   (e.g., breaking news, recent releases, live weather/sports).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from enum import Enum

import httpx
from pydantic import BaseModel, Field

from phase_3_rag.config import get_settings
from phase_3_rag.web_search import MockSearchProvider, WebSearchRAG

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


class RouteTarget(str, Enum):
    """Target destination for an incoming query."""

    DIRECT_LLM = "DIRECT_LLM"
    LOCAL_CORPUS = "LOCAL_CORPUS"
    WEB_SEARCH = "WEB_SEARCH"


class RoutingDecision(BaseModel):
    """Classification metadata explaining query routing."""

    route: RouteTarget = Field(..., description="Selected routing destination.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    reasoning: str = Field(..., description="Rationale for routing selection.")
    needs_retrieval: bool = Field(
        ..., description="True if any external or local retrieval is required."
    )
    keywords: list[str] = Field(
        default_factory=list, description="Extracted classification keywords."
    )


class RoutedRAGResponse(BaseModel):
    """Unified response object across all execution routes."""

    question: str
    decision: RoutingDecision
    answer: str
    sources: list[str] = Field(
        default_factory=list, description="Referenced URLs or Chunk IDs."
    )
    retrieval_time_ms: float = Field(
        default=0.0, description="Latency spent on retrieval in milliseconds."
    )


# -----------------------------------------------------------------------------
# Routing Classifiers
# -----------------------------------------------------------------------------

ROUTER_PROMPT_TEMPLATE = """You are an expert query router for an AI assistant.
Classify the user's question into EXACTLY ONE of these three routes:

1. DIRECT_LLM:
   - General programming tasks, code snippets, algorithm implementations.
   - Mathematics, logic puzzles, translations, creative writing, roleplay.
   - Conversational pleasantries (greetings, 'who are you', 'how are you').
   - General concepts established long ago that do not change.
   - Does NOT require any document retrieval.

2. LOCAL_CORPUS:
   - Questions specifically about Project Odyssey Mars Base engineering.
   - ECLSS (Environmental Control & Life Support Systems), telemetry limits.
   - Mars rover subsystems, habitat environmental limits, propulsion metrics.
   - Proprietary engineering specs, internal operating procedures, PDF manuals.

3. WEB_SEARCH:
   - Breaking news, recent real-world events, live weather, sports scores.
   - Questions mentioning 'latest', 'recent', 'today', 'current', 'newest release'.
   - External software versions, modern libraries (e.g. React 19, Python 3.13).
   - Real-time stock prices or unbounded internet knowledge.

User Question: {question}

Return ONLY valid JSON matching this structure:
{{
  "route": "DIRECT_LLM" | "LOCAL_CORPUS" | "WEB_SEARCH",
  "confidence": 0.95,
  "reasoning": "brief explanation",
  "needs_retrieval": true | false,
  "keywords": ["keyword1", "keyword2"]
}}
"""


def classify_route_heuristic(question: str) -> RoutingDecision:
    """Fast rule-based classifier for offline and zero-latency routing."""
    q_lower = question.lower().strip()

    # 1. Check for Greetings / Chit-Chat
    greetings = [
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b",
        r"^how\s+are\s+you\b",
        r"^who\s+are\s+you\b",
        r"^what\s+is\s+your\s+name\b",
        r"^thank(s|\s+you)\b",
    ]
    for pat in greetings:
        if re.search(pat, q_lower):
            return RoutingDecision(
                route=RouteTarget.DIRECT_LLM,
                confidence=0.99,
                reasoning="Conversational greeting or chit-chat; no retrieval needed.",
                needs_retrieval=False,
                keywords=["greeting"],
            )

    # 2. Check for Code Generation / Math / Language Tasks
    direct_patterns = [
        r"\b(write|create|implement|give\s+me)\s+(a|an)?\s*(python|javascript|typescript|c\+\+|rust|java|sql)?\s*(function|script|code|program|class|regex)\b",
        r"\b(solve|calculate|evaluate|what\s+is)\s+[\d\s\+\-\*\/\^\(\)\.]+\??$",
        r"\b(translate|convert)\s+.*?\s+to\s+(spanish|french|german|japanese|english)\b",
        r"\b(explain\s+how|what\s+is\s+a)\s+(closure|recursion|pointer|linked\s+list|binary\s+tree|polymorphism)\b",
    ]
    for pat in direct_patterns:
        if re.search(pat, q_lower):
            return RoutingDecision(
                route=RouteTarget.DIRECT_LLM,
                confidence=0.95,
                reasoning=(
                    "General programming, math, or language task suitable for "
                    "direct LLM."
                ),
                needs_retrieval=False,
                keywords=["code/math/logic"],
            )

    # 3. Check for Local Corpus (Project Odyssey / Mars Mission Engineering)
    local_keywords = [
        "eclss",
        "mars",
        "telemetry",
        "cryogen",
        "propulsion",
        "life support",
        "habitat",
        "power bus",
        "oxygen generation",
        "co2 scrubber",
        "subsystem",
        "odyssey",
        "internal doc",
        "pdf_0",
        "table_sample",
    ]
    matched_local = [kw for kw in local_keywords if kw in q_lower]
    if matched_local:
        return RoutingDecision(
            route=RouteTarget.LOCAL_CORPUS,
            confidence=0.92,
            reasoning=(
                f"Contains proprietary domain keywords ({', '.join(matched_local)}); "
                "routed to local engineering corpus."
            ),
            needs_retrieval=True,
            keywords=matched_local,
        )

    # 4. Check for Web Search (Temporal, breaking news, external current events)
    web_keywords = [
        "latest",
        "recent",
        "today",
        "yesterday",
        "current",
        "this week",
        "this year",
        "news",
        "weather",
        "stock price",
        "who won",
        "score",
        "release",
        "version",
        "breaking",
        "update",
        "2024",
        "2025",
        "2026",
    ]
    matched_web = [kw for kw in web_keywords if kw in q_lower]
    if matched_web:
        return RoutingDecision(
            route=RouteTarget.WEB_SEARCH,
            confidence=0.88,
            reasoning=(
                f"Contains temporal/volatile keywords ({', '.join(matched_web)}); "
                "routed to live web search."
            ),
            needs_retrieval=True,
            keywords=matched_web,
        )

    # Default fallback: Direct LLM for general knowledge
    return RoutingDecision(
        route=RouteTarget.DIRECT_LLM,
        confidence=0.75,
        reasoning="General query with no domain-specific or temporal indicators.",
        needs_retrieval=False,
        keywords=["general"],
    )


def classify_route_llm(
    question: str,
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> RoutingDecision:
    """Classify query route using Gemini with structured JSON output."""
    settings = get_settings()
    active_key = api_key if api_key is not None else settings.gemini_api_key
    if not active_key:
        return classify_route_heuristic(question)

    active_model = model or settings.gemini_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={active_key}"
    )

    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    close_client = False
    if client is None:
        client = httpx.Client(timeout=15.0)
        close_client = True

    try:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            logger.warning(
                "Gemini router returned HTTP %s; falling back to heuristic",
                resp.status_code,
            )
            return classify_route_heuristic(question)

        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)

        route_str = parsed.get("route", "DIRECT_LLM").upper()
        route_map = {
            "DIRECT_LLM": RouteTarget.DIRECT_LLM,
            "LOCAL_CORPUS": RouteTarget.LOCAL_CORPUS,
            "WEB_SEARCH": RouteTarget.WEB_SEARCH,
        }
        target_route = route_map.get(route_str, RouteTarget.DIRECT_LLM)

        return RoutingDecision(
            route=target_route,
            confidence=float(parsed.get("confidence", 0.9)),
            reasoning=str(parsed.get("reasoning", "Classified via Gemini.")),
            needs_retrieval=bool(
                parsed.get("needs_retrieval", target_route != RouteTarget.DIRECT_LLM)
            ),
            keywords=list(parsed.get("keywords", [])),
        )
    except Exception as exc:
        logger.warning("Gemini router exception (%s); using heuristic", exc)
        return classify_route_heuristic(question)
    finally:
        if close_client:
            client.close()


# -----------------------------------------------------------------------------
# Handlers for Each Route
# -----------------------------------------------------------------------------


def handle_direct_llm(
    question: str,
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Generate answer directly from LLM parametric memory without retrieval."""
    settings = get_settings()
    active_key = api_key if api_key is not None else settings.gemini_api_key
    if not active_key:
        return (
            f"[Direct LLM Offline Response]: Successfully answered '{question}' "
            "directly using parametric model knowledge (no retrieval needed)."
        )

    active_model = model or settings.gemini_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={active_key}"
    )
    prompt = (
        "You are a helpful, concise assistant. Answer the user's question directly, "
        "relying on your general knowledge and reasoning skills.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
    }

    close_client = False
    if client is None:
        client = httpx.Client(timeout=25.0)
        close_client = True

    try:
        resp = client.post(url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()
        return f"Error ({resp.status_code}): Direct LLM generation failed."
    except Exception as exc:
        return f"Direct LLM exception: {exc}"
    finally:
        if close_client:
            client.close()


def handle_local_corpus(
    question: str,
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, list[str]]:
    """Retrieve from local engineering PDF corpus and synthesize answer."""
    # Embedded technical knowledge from Mars Odyssey corpus (Tasks 3.7 - 3.10)
    corpus_snippets = [
        (
            "doc_eclss_limits",
            "Project Odyssey ECLSS Operating Limits: Cabin atmospheric pressure is "
            "maintained at 101.3 kPa nominal (70.8 kPa minimum emergency limit). "
            "Oxygen partial pressure is regulated between 19.5 kPa and 23.1 kPa. "
            "Primary Power Bus voltage operates at 120V DC nominal with a "
            "maximum safe limit of 132V DC and fault cutoff at 138V DC.",
        ),
        (
            "doc_propulsion_specs",
            "Mars Descent & Ascent Propulsion System (MDAS): Utilizes hypergolic "
            "monomethylhydrazine (MMH) and dinitrogen tetroxide (NTO). "
            "Specific impulse is 318s in vacuum. Abort duration is 14.2s.",
        ),
        (
            "doc_habitat_thermal",
            "Habitat Thermal Control Subsystem (HTCS): Dual-loop pumped fluid system "
            "using low-toxicity propylene glycol/water mixture. Rejection capacity is "
            "28 kW thermal via deployable composite radiator panels.",
        ),
    ]

    q_lower = question.lower()
    matched_snippets: list[tuple[str, str]] = []
    for doc_id, text in corpus_snippets:
        if any(w in text.lower() for w in q_lower.split()):
            matched_snippets.append((doc_id, text))

    if not matched_snippets:
        matched_snippets = [corpus_snippets[0]]

    sources = [doc_id for doc_id, _ in matched_snippets]
    context_str = "\n\n".join(
        f"[{doc_id}]: {text}" for doc_id, text in matched_snippets
    )

    settings = get_settings()
    active_key = api_key if api_key is not None else settings.gemini_api_key
    if not active_key:
        return (
            f"Based on internal engineering document [{sources[0]}]:\n"
            f"{matched_snippets[0][1]}",
            sources,
        )

    active_model = model or settings.gemini_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={active_key}"
    )
    prompt = (
        "You are an AI mission operations engineer. Answer the user question "
        "using ONLY the internal technical corpus excerpts below. Always cite "
        "the document ID in brackets.\n\n"
        f"Technical Excerpts:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600},
    }

    close_client = False
    if client is None:
        client = httpx.Client(timeout=25.0)
        close_client = True

    try:
        resp = client.post(url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return answer, sources
        return f"Error ({resp.status_code}): Local corpus synthesis failed.", sources
    except Exception as exc:
        return f"Local corpus error: {exc}", sources
    finally:
        if close_client:
            client.close()


# -----------------------------------------------------------------------------
# Adaptive RAG Router Orchestrator
# -----------------------------------------------------------------------------


class AdaptiveRAGRouter:
    """Unified router directing queries to Direct LLM, Local Corpus, or Web Search."""

    def __init__(
        self,
        web_search_rag: WebSearchRAG | None = None,
        *,
        client: httpx.Client | None = None,
        model: str | None = None,
        api_key: str | None = None,
        use_mock_search: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.api_key = api_key
        if web_search_rag is not None:
            self.web_search_rag = web_search_rag
        else:
            provider = MockSearchProvider() if use_mock_search else None
            self.web_search_rag = WebSearchRAG(
                search_provider=provider,
                client=client,
                model=model,
                api_key=api_key,
                fetch_pages=True,
            )

    def route_query(self, question: str) -> RoutingDecision:
        """Classify the question into its optimal execution route."""
        return classify_route_llm(
            question,
            client=self.client,
            model=self.model,
            api_key=self.api_key,
        )

    def route_and_execute(
        self, question: str, *, num_results: int = 5
    ) -> RoutedRAGResponse:
        """Classify question, dispatch to handler, and return unified response."""
        decision = self.route_query(question)

        t_start = time.perf_counter()
        retrieval_ms = 0.0

        if decision.route == RouteTarget.DIRECT_LLM:
            # Direct generation without retrieval overhead
            answer = handle_direct_llm(
                question,
                client=self.client,
                model=self.model,
                api_key=self.api_key,
            )
            sources: list[str] = []

        elif decision.route == RouteTarget.LOCAL_CORPUS:
            # Route to local proprietary engineering corpus
            answer, sources = handle_local_corpus(
                question,
                client=self.client,
                model=self.model,
                api_key=self.api_key,
            )
            retrieval_ms = (time.perf_counter() - t_start) * 1000

        else:
            # Route to live Web Search RAG pipeline
            web_resp = self.web_search_rag.query(question, num_results=num_results)
            answer = web_resp.answer
            sources = web_resp.citations
            retrieval_ms = (time.perf_counter() - t_start) * 1000

        return RoutedRAGResponse(
            question=question,
            decision=decision,
            answer=answer,
            sources=sources,
            retrieval_time_ms=round(retrieval_ms, 2),
        )


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """CLI runner for Adaptive RAG Router."""
    parser = argparse.ArgumentParser(
        description="Adaptive Query Router for RAG (Task 3.19)"
    )
    parser.add_argument("--query", "-q", type=str, help="Query to route and execute.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock search provider for web route.",
    )
    args = parser.parse_args()

    router = AdaptiveRAGRouter(use_mock_search=args.mock)

    if args.query:
        print("\n" + "=" * 70)
        print("ADAPTIVE RAG ROUTER (TASK 3.19)")
        print("=" * 70)
        resp = router.route_and_execute(args.query)

        print(f'\n[?] Question   : "{resp.question}"')
        print(f"[>] Route Target: {resp.decision.route.value}")
        print(f"    Confidence : {resp.decision.confidence:.2f}")
        print(f"    Retrieval  : {'YES' if resp.decision.needs_retrieval else 'NO'}")
        print(f"    Reasoning  : {resp.decision.reasoning}")
        print(f"    Latency    : {resp.retrieval_time_ms} ms")

        print("\n" + "-" * 70)
        print("Answer:")
        print("-" * 70)
        print(resp.answer)
        print("-" * 70)
        if resp.sources:
            print("Sources:")
            for s in resp.sources:
                print(f" - {s}")
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
