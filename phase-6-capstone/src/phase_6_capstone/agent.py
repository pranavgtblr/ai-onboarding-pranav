"""Stateful LangGraph Agent for PG Recommends with Human Escalation."""

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from phase_6_capstone.db import DatabaseManager, HumanEscalationModel
from phase_6_capstone.retrieval import HybridMovieRetriever, MovieCitation
from phase_6_capstone.taste_engine import TasteProfileManager, UserTasteProfile


class AgentTurnState(BaseModel):
    """Execution state for a single conversational turn."""

    tenant_id: str
    user_id: str
    conversation_id: str
    input_message: str
    final_response: str = ""
    citations: list[MovieCitation] = Field(default_factory=list)
    escalation_status: str | None = None
    escalation_ticket_id: str | None = None
    taste_profile: UserTasteProfile | None = None


class CapstoneAgent:
    """Conversational film curator agent with persistent taste learning.

    Features multi-source hybrid retrieval and human escalation handoff.
    """

    def __init__(
        self,
        retriever: HybridMovieRetriever,
        taste_manager: TasteProfileManager,
        db: DatabaseManager,
    ):
        self.retriever = retriever
        self.taste_manager = taste_manager
        self.db = db

    def is_escalation_intent(self, message: str) -> bool:
        """Determines if the message requires escalation to the human curator (PG)."""
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

    async def run_turn(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> AgentTurnState:
        """Executes a full conversational turn."""
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

        # 3. Retrieve relevant movies from PG's catalog
        results = self.retriever.search(message, top_k=3)

        # Cyclic query rewriting if 0 matches
        if not results:
            # Broaden query by stripping specific conjunctions
            words = [w for w in message.split() if len(w) > 3]
            if words:
                broad_query = " ".join(words[:3])
                results = self.retriever.search(broad_query, top_k=3)

        if not results:
            state.final_response = (
                "I couldn't find any films in PG's reviewed catalog directly matching "
                "that query. Could you tell me more about the genres, directors, or "
                "moods you're looking for?"
            )
            return state

        # 4. Generate recommendations with PG's ratings and citations
        citations = [res.to_citation() for res in results]
        state.citations = citations
        state.final_response = self._synthesize_curator_dialogue(
            message=message, citations=citations
        )
        return state

    def _synthesize_curator_dialogue(
        self, message: str, citations: list[MovieCitation]
    ) -> str:
        """Synthesizes an authentic, conversational curator response in PG's voice."""
        if not citations:
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
        elif any(
            k in m_lower for k in ["sci-fi", "space", "alien", "dune", "cyberpunk"]
        ):
            intro = (
                "Big ideas, meticulous craft, and visuals that demand your full "
                "attention. Here are the ones from my logs that really stuck with me."
            )
        elif any(
            k in m_lower for k in ["malayalam", "mollywood", "shahi kabir", "sadasivan"]
        ):
            intro = (
                "Malayalam cinema has been delivering some of the absolute tightest, "
                "most atmospheric filmmaking lately. A couple of these left a huge "
                "mark on me."
            )
        else:
            intro = (
                "Let's talk cinema! Based on what you're craving, here's what I'd "
                "genuinely recommend diving into from my diary."
            )

        # Talk about top picks conversationally without plain-text bullet listing
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

    async def stream_turn(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streams tokens and metadata events for real-time UI rendering."""
        turn_state = await self.run_turn(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
        )

        if turn_state.taste_profile:
            yield {
                "event": "taste_update",
                "data": {
                    "liked_directors": turn_state.taste_profile.liked_directors,
                    "liked_genres": turn_state.taste_profile.liked_genres,
                    "disliked_elements": turn_state.taste_profile.disliked_elements,
                },
            }

        # Stream words as token chunks
        words = turn_state.final_response.split(" ")
        for word in words:
            yield {"event": "token", "data": word + " "}

        # Emit citations event
        if turn_state.citations:
            yield {
                "event": "citations",
                "data": [c.model_dump() for c in turn_state.citations],
            }

        # Emit escalation event if triggered
        if turn_state.escalation_status:
            yield {
                "event": "escalation",
                "data": {
                    "status": turn_state.escalation_status,
                    "ticket_id": turn_state.escalation_ticket_id,
                },
            }

        yield {"event": "done", "data": "[DONE]"}
