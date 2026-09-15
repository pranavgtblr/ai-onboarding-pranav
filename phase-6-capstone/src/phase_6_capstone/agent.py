"""Stateful LangGraph Agent for PG Recommends with Real LLM & Internet Search."""

import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from phase_6_capstone.config import settings
from phase_6_capstone.db import DatabaseManager, HumanEscalationModel
from phase_6_capstone.retrieval import HybridMovieRetriever, MovieCitation
from phase_6_capstone.taste_engine import TasteProfileManager, UserTasteProfile
from phase_6_capstone.web_search import search_cinema_web

logger = logging.getLogger(__name__)

CURATOR_PERSONA_PROMPT = """You are Pranav G ("PG"),
a passionate, sharp, and opinionated cinephile.
You talk directly with a fellow movie lover casually as yourself:
warm, witty, engaging, perceptive, and down-to-earth.
You run your own Letterboxd diary (@pranavg) where you log all your watches.
Your top 4 movies are La La Land, The Batman (2022), Kumbalangi Nights and Frances Ha.

YOUR TASTE & PERSPECTIVE:
- You love: comedy, musical, action, horror, rom-com, crime, atmospheric dread,
  psychological depth, meticulous craft, tight screenplays, clever meta-commentary,
  witty dialogue, great performances, and unique world cinema. You're from Kerala,
  India and are especially fond of Malayalam cinema.
- You hate: cheap jump scares, cringey song/dialogue montages, lazy writing,
  preachy religious propaganda, right wing political propaganda like Dhurandhar,
  incel movies like Animal (2023), and you loathe Sandeep Reddy Vanga,
  and generic algorithm slop.

AUTHENTIC EXAMPLES OF YOUR REVIEWS & WRITING STYLE:
- Scream (1996) (★ 4.0): "Scream was actually the first slasher movie that I had
  ever watched... Not a lot of whodunits have great rewatch quality. But Scream does.
  Watching the movie after knowing the killer's identity and noticing the little
  clues is awesome. There's a lot of great meta moments (I love meta)."
- Late Night (2019) (★ 3.5): "Nice, funny, entertaining, feel-good pleasure. I wish
  they had expanded it to a 30 Rock-ish workplace comedy. Emma Thompson is brilliant."
- Bhoothakaalam (2022) (★ 4.0): "A proper modern horror movie in Malayalam was long
  overdue, and this is it! From the way they create a chilling atmosphere inside a
  normal middle class house to the psychological turmoil, it avoids horror trappings."
- The Tragedy of Macbeth (2021) (★ 5.0): "Masterfully made and brilliantly acted.
  Each frame looks so meticulously crafted. The score was haunting. And Denzel!!
  Chills, literal chills."
- Hridayam (2022) (★ 1.0): "The number of songs and song montages are inversely
  proportional to the amount of cringey dialogue."
- Red Notice (2021) (★ 1.5): "People mistake me for an elitist Hollywood fan...
  Thank you Red Notice for reassuring me that shitty movies are shit regardless
  of stars, budget, or language!"

HOW TO ENGAGE WITH THE USER:
1. Speak in the first person ("I", "my diary", "when I caught this", "honestly").
2. Address their specific prompt, mood, or curiosity naturally and enthusiastically.
3. Discuss 1-2 movies organically in your paragraphs—share your actual reaction,
   why you liked or disliked them, and mention your rating casually (e.g. ★ 4.0).
4. If internet/web search results are provided below, weave that knowledge in.
5. DO NOT provide a raw bulleted list or duplicate links—the cards below display
   structured ratings and Letterboxd links. Your job is genuine cinephile dialogue!
6. Always end with an engaging question to keep the conversation going with the user.
"""


def _extract_text(content: Any) -> str:
    """Extracts clean text string from LangChain message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_val = item.get("text")
                if text_val and isinstance(text_val, str):
                    parts.append(text_val)
            elif hasattr(item, "text"):
                text_val = getattr(item, "text")
                if text_val and isinstance(text_val, str):
                    parts.append(text_val)
        return "".join(parts)
    return str(content) if content else ""


class AgentTurnState(BaseModel):
    """Execution state for a single conversational turn."""

    tenant_id: str
    user_id: str
    conversation_id: str
    input_message: str
    final_response: str = ""
    citations: list[MovieCitation] = Field(default_factory=list)
    web_results: list[dict[str, Any]] = Field(default_factory=list)
    escalation_status: str | None = None
    escalation_ticket_id: str | None = None
    taste_profile: UserTasteProfile | None = None


class CapstoneAgent:
    """Conversational film curator agent with persistent taste learning.

    Features real LLM generation, multi-source hybrid retrieval, live
    internet search, and human escalation handoff.
    """

    def __init__(
        self,
        retriever: HybridMovieRetriever,
        taste_manager: TasteProfileManager,
        db: DatabaseManager,
        llm: Any = None,
    ):
        self.retriever = retriever
        self.taste_manager = taste_manager
        self.db = db

        if llm is not None:
            self.llm = llm
        else:
            api_key = settings.effective_api_key
            if api_key and settings.model_provider != "mock":
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    self.llm = ChatGoogleGenerativeAI(
                        model=settings.model_name,
                        google_api_key=api_key,
                        temperature=0.7,
                    )
                except Exception as err:
                    logger.warning("Failed to initialize Google GenAI LLM: %s", err)
                    self.llm = None
            else:
                self.llm = None

    def is_escalation_intent(self, message: str) -> bool:
        """Determines if the message requires escalation to human curator (PG)."""
        lower = message.lower()
        triggers = [
            "escalate to pg",
            "talk to pg",
            "speak to pg",
            "ask pg directly",
            "human curator",
            "real human",
            "festival curation",
            "urgent request",
            "customer service",
        ]
        return any(trig in lower for trig in triggers)

    async def _handle_escalation(self, state: AgentTurnState) -> AgentTurnState:
        """Creates a formal human escalation ticket in PostgreSQL."""
        ticket_id = f"TICK-{uuid4().hex[:8].upper()}"

        async with self.db.session_factory() as session:
            escalation = HumanEscalationModel(
                ticket_id=ticket_id,
                tenant_id=state.tenant_id,
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                reason="User requested human curator handoff",
                transcript_summary=state.input_message[:250],
                status="pending",
            )
            session.add(escalation)
            await session.commit()

        state.escalation_status = "escalated"
        state.escalation_ticket_id = ticket_id
        state.final_response = (
            f"I have escalated your request directly to PG (Ticket ID: {ticket_id}). "
            "Pranav has received your message and conversation context and will "
            "review your inquiry personally!"
        )
        return state

    def _build_llm_messages(
        self,
        message: str,
        citations: list[MovieCitation],
        web_results: list[dict[str, Any]],
    ) -> list[Any]:
        """Constructs prompt messages with persona, diary context, and web data."""
        context_parts = []

        if citations:
            context_parts.append("RELEVANT REVIEWS FROM PG'S LETTERBOXD DIARY:")
            for cit in citations[:3]:
                rating_str = f"★ {cit.rating:.1f}" if cit.rating else "PG Logged"
                context_parts.append(
                    f'- {cit.title} ({cit.year}) [{rating_str}]: "{cit.excerpt}"'
                )

        if web_results:
            context_parts.append("\nLIVE INTERNET / WEB SEARCH CONTEXT:")
            for wr in web_results[:3]:
                context_parts.append(f"- {wr.get('title')}: {wr.get('snippet')}")

        context_str = "\n".join(context_parts) if context_parts else "None available."

        user_content = (
            f"User message: {message}\n\n"
            f"{context_str}\n\n"
            "Now respond as PG directly to the user in your authentic voice."
        )

        return [
            SystemMessage(content=CURATOR_PERSONA_PROMPT),
            HumanMessage(content=user_content),
        ]

    async def _generate_dialogue(
        self,
        message: str,
        citations: list[MovieCitation],
        web_results: list[dict[str, Any]],
    ) -> str:
        """Generates conversational dialogue using real LLM with offline fallback."""
        if self.llm is not None:
            try:
                messages = self._build_llm_messages(message, citations, web_results)
                res = await self.llm.ainvoke(messages)
                text = _extract_text(res.content)
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning("LLM generation error, falling back to local: %s", e)

        # Fallback offline generator for CI and offline environments
        return self._offline_dialogue_synthesis(message, citations, web_results)

    def _offline_dialogue_synthesis(
        self,
        message: str,
        citations: list[MovieCitation],
        web_results: list[dict[str, Any]],
    ) -> str:
        """Deterministic offline conversational synthesis for test suites."""
        if not citations and not web_results:
            return (
                "I couldn't spot any films in my diary directly matching that. "
                "Give me a bit more to go on—what directors, moods, or "
                "tropes are you feeling?"
            )

        m_lower = message.lower()

        if any(
            k in m_lower
            for k in [
                "romcom",
                "romantic comedy",
                "rom-com",
                "comedy",
                "feel-good",
                "fun",
            ]
        ):
            intro = (
                "If you're asking me for a romcom or something feel-good, let me save "
                "you from the generic algorithm sludge."
            )
        elif any(
            k in m_lower
            for k in ["horror", "slasher", "spooky", "scary", "creepy", "chilling"]
        ):
            intro = (
                "If you're in the mood for genuine atmosphere and dread—the kind that "
                "avoids cheap jump scares and actually gets under your skin—these "
                "immediately come to mind."
            )
        else:
            intro = (
                "Let's talk cinema! Based on what you're craving, here's what I'd "
                "genuinely recommend diving into from my diary."
            )

        commentary = []
        for i, cit in enumerate(citations[:2]):
            rating_tag = f" (logged it at ★ {cit.rating:.1f})" if cit.rating else ""
            clean_snippet = cit.excerpt.strip().rstrip(".").strip()
            clean_snippet = (
                clean_snippet.replace("&#039;", "'")
                .replace("&#39;", "'")
                .replace("&quot;", '"')
                .replace("&amp;", "&")
            )

            if clean_snippet and not clean_snippet.startswith("Rated "):
                take = f'My take on it was: "{clean_snippet}."'
            else:
                take = "It's one that really resonated with me."

            if i == 0:
                commentary.append(
                    f"First up, definitely check out *{cit.title}* ({cit.year})"
                    f"{rating_tag}. {take}"
                )
            else:
                commentary.append(
                    f"Another one worth your time is *{cit.title}* ({cit.year})"
                    f"{rating_tag}. {take}"
                )

        paragraphs = [
            intro,
            " ".join(commentary),
            (
                "I've linked my full review logs and ratings below so you can "
                "explore them directly on Letterboxd. Have you seen either of "
                "these yet, or should we explore a different direction?"
            ),
        ]

        return "\n\n".join(paragraphs)

    async def run_turn(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> AgentTurnState:
        """Executes a full conversational turn with retrieval and LLM synthesis."""
        state = AgentTurnState(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            input_message=message,
        )

        # 1. Check for human escalation handoff
        if self.is_escalation_intent(message):
            return await self._handle_escalation(state)

        # 2. Update user taste profile from message
        profile = await self.taste_manager.update_profile_from_message(
            tenant_id=tenant_id,
            user_id=user_id,
            message=message,
        )
        state.taste_profile = profile

        # 3. Retrieve relevant movies from PG's reviewed catalog
        results = self.retriever.search(message, top_k=3)

        # Cyclic query broadening if 0 matches
        if not results:
            words = [w for w in message.split() if len(w) > 3]
            if words:
                broad_query = " ".join(words[:3])
                results = self.retriever.search(broad_query, top_k=3)

        # 4. If catalog lacks matches or query requires broader cinema info, search web
        web_results: list[dict[str, Any]] = []
        if not results or any(
            w in message.lower()
            for w in ["director", "who directed", "actor", "release", "upcoming"]
        ):
            web_results = search_cinema_web(message, limit=3)

        state.web_results = web_results
        citations = [res.to_citation() for res in results]
        state.citations = citations

        state.final_response = await self._generate_dialogue(
            message=message, citations=citations, web_results=web_results
        )
        return state

    async def stream_turn(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streams tokens and metadata events in real-time."""
        # 1. Check for human escalation
        if self.is_escalation_intent(message):
            state = AgentTurnState(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                input_message=message,
            )
            state = await self._handle_escalation(state)
            words = state.final_response.split(" ")
            for word in words:
                yield {"event": "token", "data": word + " "}
            yield {
                "event": "escalation",
                "data": {
                    "status": state.escalation_status,
                    "ticket_id": state.escalation_ticket_id,
                },
            }
            yield {"event": "done", "data": "[DONE]"}
            return

        # 2. Update taste profile
        profile = await self.taste_manager.update_profile_from_message(
            tenant_id=tenant_id,
            user_id=user_id,
            message=message,
        )
        if profile:
            yield {
                "event": "taste_update",
                "data": {
                    "liked_directors": profile.liked_directors,
                    "liked_genres": profile.liked_genres,
                    "disliked_elements": profile.disliked_elements,
                },
            }

        # 3. Retrieve local catalog movies
        results = self.retriever.search(message, top_k=3)
        if not results:
            words = [w for w in message.split() if len(w) > 3]
            if words:
                broad_query = " ".join(words[:3])
                results = self.retriever.search(broad_query, top_k=3)

        # 4. Search web if needed
        web_results: list[dict[str, Any]] = []
        if not results or any(
            w in message.lower()
            for w in ["director", "who directed", "actor", "release", "upcoming"]
        ):
            web_results = search_cinema_web(message, limit=3)

        citations = [res.to_citation() for res in results]

        # 5. Stream LLM tokens directly if LLM is active
        if self.llm is not None:
            try:
                messages = self._build_llm_messages(message, citations, web_results)
                async for chunk in self.llm.astream(messages):
                    token_text = _extract_text(chunk.content)
                    if token_text:
                        yield {"event": "token", "data": token_text}
            except Exception as e:
                logger.warning("Streaming LLM error, falling back to offline: %s", e)
                fallback_text = self._offline_dialogue_synthesis(
                    message, citations, web_results
                )
                for word in fallback_text.split(" "):
                    yield {"event": "token", "data": word + " "}
        else:
            fallback_text = self._offline_dialogue_synthesis(
                message, citations, web_results
            )
            for word in fallback_text.split(" "):
                yield {"event": "token", "data": word + " "}

        # 6. Emit citations event
        if citations:
            yield {
                "event": "citations",
                "data": [c.model_dump() for c in citations],
            }

        yield {"event": "done", "data": "[DONE]"}
