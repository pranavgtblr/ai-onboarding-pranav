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

        lines = ["Here are PG's top recommendations based on your taste:\n"]
        for idx, cit in enumerate(citations, 1):
            rating_str = f"({cit.rating}★)" if cit.rating is not None else ""
            lines.append(
                f"{idx}. **{cit.title}** ({cit.year}) {rating_str}\n"
                f'   *PG\'s Take:* "{cit.excerpt}"\n'
                f"   [View on Letterboxd]({cit.letterboxd_url})"
            )

        state.final_response = "\n\n".join(lines)
        return state

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
