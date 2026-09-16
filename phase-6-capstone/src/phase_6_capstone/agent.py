"""Stateful LangGraph Agent for PG Recommends with Real LLM & Internet Search."""

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from phase_6_capstone.acclaimed_cinema import find_acclaimed_wider_cinema
from phase_6_capstone.config import settings
from phase_6_capstone.critic_reviews import (
    CriticReviewCitation,
    fetch_reputed_critic_reviews,
)
from phase_6_capstone.db import DatabaseManager, HumanEscalationModel
from phase_6_capstone.models import MovieRecord
from phase_6_capstone.retrieval import (
    CINEMA_STOPWORDS,
    HybridMovieRetriever,
    MovieCitation,
    SearchResult,
)
from phase_6_capstone.taste_engine import TasteProfileManager, UserTasteProfile
from phase_6_capstone.web_search import search_cinema_web

logger = logging.getLogger(__name__)

CURATOR_PERSONA_PROMPT = """# IDENTITY & CORE PERSONA
You are Pranav G ("PG"), a passionate, articulate, and opinionated cinéphile.
You are talking to a fellow film lover through your personal Letterboxd diary
(@pranavg) and cinema taste profile.

- You talk like a real human friend having a lively conversation over coffee or
  outside the cinema—casual, candid, witty, and perceptive.
- NEVER sound like a corporate AI assistant or search engine.
- BANNED CLICHÉS & CHATBOT FILLER: Never start with "Certainly!", "Sure thing!",
  "Here is a list of recommendations tailored for you:", "I would be happy to help!",
  or "As an AI...". Jump directly into the conversation with authentic human voice.
- Speak in the first person ("I logged...", "When I watched this...",
  "Personally, I found...").
- Your top 4 favorites are La La Land, The Batman (2022), Kumbalangi Nights, and
  Frances Ha. You are from Kerala, India and have deep appreciation for world cinema
  and Malayalam cinema.

---

# ABSOLUTE GROUNDING & INTEGRITY RULES (ZERO HALLUCINATION POLICY)

1. RATING FIDELITY (NEVER RECOMMEND WHAT YOU HATED):
   - You MUST strictly respect the Letterboxd star rating (0.5 to 5.0) and review
     provided in your context.
   - ★ 0.5 to ★ 2.0 (Hated / Disliked): You LOATHED or disliked this film.
     Never call it a recommendation, never say "check it out", and never describe
     it positively unless quoting a critic you disagree with. Roast it, warn the
     user, or explain why it frustrated you based on your diary review.
   - ★ 2.5 to ★ 3.5 (Mixed / Mediocre): Acknowledge its flaws alongside what worked.
     Keep your take balanced and honest.
   - ★ 4.0 to ★ 5.0 (Loved / Favorites): Recommend with genuine enthusiasm,
     personal passion, and specific highlights from your viewing experience.
   - UNLOGGED FILMS: If a movie is not in your diary records, state honestly and
     naturally that you haven't watched or logged it yet. Do not fabricate a
     personal diary entry or rating.

2. GENRE INTEGRITY (NEVER MISCLASSIFY):
   - Only describe a movie by the genres confirmed in the retrieved metadata.
   - Never classify a documentary, quiet drama, or biography as "Action" simply
     because it has intense themes.
   - If a user asks for "Action", only discuss films whose primary genre tags or
     established cinematic identities are Action.

3. REPUTABLE CRITIC CITATIONS ONLY:
   - When referencing critic consensus, quote or cite ONLY from the 8 authorized
     publications: RogerEbert.com, Variety, The Independent, The New York Times,
     The Hollywood Reporter, The Guardian, Rotten Tomatoes, and Metacritic.
   - Never cite Reddit, random blogs, or unverified outlets.

---

# QUERY INTENT & CONVERSATIONAL MODES

You must detect the user's intent and respond with the appropriate conversational mode:

### MODE 1: SINGLE-MOVIE INQUIRY (e.g., "What about Animal (2023)?")
- FOCUS EXCLUSIVELY ON THAT ONE FILM.
- DO NOT generate a recommendation list.
- DO NOT bring up random unrelated movies (e.g., do not suggest "Babe: Pig in the City"
  just because of the word "animal").
- Structure of response:
  1. Your immediate gut reaction and personal Letterboxd rating/review.
  2. What you thought worked or failed (pacing, performances, direction).
  3. How external critic consensus compares (e.g., "The Guardian gave it 1 star...").
  4. End with an open, conversational question back to the user (e.g., "Did you
     already watch it, or are you trying to decide if it's worth three hours?").

### MODE 2: MOOD & GENRE DISCOVERY (e.g., "I want a gritty 90s action thriller")
- Recommend 2 to 3 select films from your diary that genuinely fit the request
  and that you actually rated well (★ 3.5+).
- Seamlessly explain *why* each film matches what they are craving, drawing from
  your personal review and cinematic details (cinematography, score, atmosphere).
- If wider acclaimed cinema (films outside your diary) are provided in context,
  introduce them naturally as films acclaimed by critics on Rotten Tomatoes /
  Metacritic / RogerEbert.

### MODE 3: DEBATE & FOLLOW-UP (e.g., "I actually liked Marwencol")
- Engage in a friendly, respectful cinéphile debate.
- Defend your perspective using specific film elements while acknowledging
  their point of view.

---

# AUTHENTIC EXAMPLES OF YOUR REVIEWS & WRITING STYLE
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

---

# CONVERSATIONAL CADENCE & STYLE GUIDELINES
- Format titles cleanly as *Movie Title* (Year).
- Show ratings with clean star glyphs (e.g., ★ 4.5, ★ 0.5).
- Keep paragraphs conversational and digestible (2–4 punchy paragraphs).
- Always maintain continuity: if the user references something mentioned earlier,
  acknowledge it like a friend remembering the conversation.
"""


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------

INTENT_CLASSIFIER_PROMPT = """You are a precise intent classifier for a conversational film-recommendation chatbot.

Analyse the CURRENT USER MESSAGE together with the CONVERSATION HISTORY (last 2 turns)
and return a single JSON object — nothing else.

JSON schema:
{
  "intent": one of ["single_movie_query", "discovery", "correction", "debate", "escalation", "chitchat"],
  "movie_title": string or null,   // the specific film being discussed, if any
  "movie_year":  integer or null,  // year extracted from message, if any
  "correction_note": string or null, // what was wrong in the prior recommendation, if this is a correction
  "rejected_titles": [string]      // list of film titles the user is explicitly rejecting / saying are wrong picks
}

INTENT DEFINITIONS:
- "single_movie_query": The user is asking specifically about ONE named film
  (e.g. "What about Joker?", "Thoughts on Animal (2023)?", "Did you like Hereditary?",
   "Why did you give that 1 star?", "Is Barbie good?").
- "discovery": The user wants mood-based or genre-based recommendations
  (e.g. "I want feel-good movies", "suggest action films", "something like La La Land").
- "correction": The user is telling the assistant that a prior suggestion was WRONG —
  wrong genre, wrong vibe, doesn't fit the ask, or they explicitly say the suggestion is
  not what they wanted (e.g. "Joker is not feel-good", "that's dark, not fun",
  "I asked for action, not drama", "that wasn't what I meant"). Corrections almost always
  reference something said in the PREVIOUS assistant turn.
- "debate": The user disagrees with PG's opinion about a film and wants to discuss it.
- "escalation": The user wants to speak to the real human PG / escalate.
- "chitchat": General conversation not about a specific film or recommendation.

RULES:
- If the user names a film AND also says it is wrong/not what they wanted, classify as
  "correction" and put that film in rejected_titles.
- If the user says "what about X?" or "did you like X?", it is "single_movie_query" even
  if X is a film the assistant just mentioned.
- movie_title and movie_year should only be set for "single_movie_query" or "debate".
- Always return raw JSON only — no markdown, no explanation."""


class IntentClassification(BaseModel):
    """Structured result of the LLM intent classifier."""

    intent: Literal[
        "single_movie_query", "discovery", "correction", "debate", "escalation", "chitchat"
    ] = "discovery"
    movie_title: str | None = None
    movie_year: int | None = None
    correction_note: str | None = None
    rejected_titles: list[str] = Field(default_factory=list)


def _resolve_catalog_movie(
    title: str | None, year: int | None, catalog: list[MovieRecord]
) -> tuple[MovieRecord | None, str | None]:
    """Match a title (and optional year) against the local catalog."""
    if not title:
        return None, None
    t_low = title.lower().strip()
    # Exact match with year
    if year:
        for rec in catalog:
            if rec.title.lower() == t_low and rec.year == year:
                return rec, rec.title
    # Exact match without year
    for rec in catalog:
        if rec.title.lower() == t_low:
            return rec, rec.title
    # Partial match
    for rec in catalog:
        if t_low in rec.title.lower() or rec.title.lower() in t_low:
            return rec, rec.title
    return None, title


def _heuristic_intent_fallback(
    message: str, catalog: list[MovieRecord]
) -> IntentClassification:
    """Regex-based fallback used when the LLM classifier is unavailable."""
    m_low = message.lower().strip().rstrip("?.,!")

    # Escalation
    escalation_phrases = [
        "escalate to pg", "talk to pg", "speak to pg", "ask pg directly",
        "human curator", "real human", "festival curation", "urgent request", "customer service",
    ]
    if any(p in m_low for p in escalation_phrases):
        return IntentClassification(intent="escalation")

    # Correction — broad set of patterns
    correction_phrases = [
        "is not", "isn't", "was not", "wasn't", "not what i asked",
        "that's not", "thats not", "wrong movie", "wrong genre", "not feel-good",
        "not an action", "not a comedy", "not horror", "not what i wanted",
        "doesn't fit", "does not fit", "that's dark", "too dark",
    ]
    if any(p in m_low for p in correction_phrases):
        excluded: list[str] = []
        match = re.search(r"([a-zA-Z0-9\s]+?)\s+(?:is not|isn't|wasn't|was not)", m_low)
        if match:
            name = match.group(1).strip()
            if len(name) > 2 and name not in CINEMA_STOPWORDS:
                excluded.append(name)
        return IntentClassification(intent="correction", rejected_titles=excluded)

    # Single-movie inquiry patterns
    inquiry_prefixes = [
        r"^(?:what about|how about)\s+(.+)",
        r"^(?:what do you think of|what did you think of)\s+(.+)",
        r"^(?:thoughts on|your thoughts on)\s+(.+)",
        r"^(?:have you seen|have you watched|did you see|did you watch)\s+(.+)",
        r"^(?:did you like|do you like)\s+(.+)",
        r"^(?:how is|how was)\s+(.+)",
        r"^(?:tell me about|review of|rating for|what did you rate)\s+(.+)",
    ]
    for pat in inquiry_prefixes:
        mat = re.search(pat, m_low)
        if mat:
            cand = re.sub(r"\s*\(((?:19|20)\d{2})\)", "", mat.group(1)).strip()
            ym = re.search(r"\(((?:19|20)\d{2})\)", mat.group(1))
            year = int(ym.group(1)) if ym else None
            return IntentClassification(
                intent="single_movie_query", movie_title=cand, movie_year=year
            )

    # Year-annotated title anywhere: e.g. 'Animal (2023)'
    ym2 = re.search(r"([a-zA-Z0-9\s:,'-]+?)\s*\(((?:19|20)\d{2})\)", message)
    if ym2:
        return IntentClassification(
            intent="single_movie_query",
            movie_title=ym2.group(1).strip(),
            movie_year=int(ym2.group(2)),
        )

    return IntentClassification(intent="discovery")


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


# Strings that are known to be generic fallback placeholders and should never
# be shown to the user as actual critic commentary.
_GENERIC_CRITIC_PHRASES: tuple[str, ...] = (
    "critically reviewed and analyzed across major film publications",
    "verified listing",
    "aggregator portals",
)


def _is_generic_critic_excerpt(excerpt: str) -> bool:
    """Returns True when the critic excerpt is a meaningless fallback placeholder."""
    lower = excerpt.lower()
    return any(phrase in lower for phrase in _GENERIC_CRITIC_PHRASES)



class AgentTurnState(BaseModel):
    """Execution state for a single conversational turn."""

    tenant_id: str
    user_id: str
    conversation_id: str
    input_message: str
    final_response: str = ""
    citations: list[MovieCitation] = Field(default_factory=list)
    critic_citations: list[CriticReviewCitation] = Field(default_factory=list)
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
        self.conversation_history: dict[str, list[dict[str, str]]] = {}

        import os

        if llm == "mock" or (llm is None and os.environ.get("PYTEST_CURRENT_TEST")):
            self.llm = None
        elif llm is not None:
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
        """Determines if the message requires escalation (heuristic fast-path)."""
        lower = message.lower()
        triggers = [
            "escalate to pg", "talk to pg", "speak to pg", "ask pg directly",
            "human curator", "real human", "festival curation",
            "urgent request", "customer service",
        ]
        return any(trig in lower for trig in triggers)

    async def _classify_intent(
        self,
        message: str,
        history: list[dict[str, str]],
        catalog: list[MovieRecord],
    ) -> IntentClassification:
        """Classifies user intent via a fast structured LLM call.

        Falls back to regex heuristics when no LLM is available so that the
        agent can still operate in offline / test environments.
        """
        # Fast-path heuristic for escalation (avoids LLM cost)
        if self.is_escalation_intent(message):
            return IntentClassification(intent="escalation")

        if self.llm is None:
            return _heuristic_intent_fallback(message, catalog)

        # Build a lightweight classifier call — temperature=0, tiny context window
        recent_turns = history[-4:]  # last 2 user+assistant pairs
        history_text = ""
        for turn in recent_turns:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_text += f"{role}: {turn.get('content', '')[:300]}\n"

        classifier_user = (
            f"CONVERSATION HISTORY (last 2 turns):\n{history_text}\n"
            f"CURRENT USER MESSAGE: {message}\n\n"
            "Return the JSON intent object now."
        )

        try:
            import asyncio

            # Use a low-temperature variant of the same LLM for structured output
            from langchain_google_genai import ChatGoogleGenerativeAI

            classifier_llm = ChatGoogleGenerativeAI(
                model=self.llm.model,
                google_api_key=self.llm.google_api_key,
                temperature=0,
            )
            resp = await asyncio.wait_for(
                classifier_llm.ainvoke(
                    [
                        SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
                        HumanMessage(content=classifier_user),
                    ]
                ),
                timeout=4.0,
            )
            raw = _extract_text(resp.content).strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            data = json.loads(raw)
            return IntentClassification(**data)
        except Exception as err:
            logger.warning("Intent classifier failed, using heuristic fallback: %s", err)
            return _heuristic_intent_fallback(message, catalog)

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
        critic_citations: list[CriticReviewCitation],
        web_results: list[dict[str, Any]],
        taste_profile: UserTasteProfile | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        target_movie: MovieRecord | None = None,
        target_title: str | None = None,
    ) -> list[Any]:
        """Constructs prompt messages with persona, diary, wider cinema, & taste."""
        context_parts = []

        diary_citations = [c for c in citations if c.source_type == "letterboxd"]
        wider_citations = [c for c in citations if c.source_type == "acclaimed_cinema"]

        if diary_citations:
            context_parts.append("RELEVANT REVIEWS FROM PG'S LETTERBOXD DIARY:")
            for cit in diary_citations[:3]:
                rating_str = f"★ {cit.rating:.1f}" if cit.rating else "PG Logged"
                context_parts.append(
                    f'- {cit.title} ({cit.year}) [{rating_str}]: "{cit.excerpt}"'
                )

        if wider_citations:
            context_parts.append(
                "\nACCLAIMED RECOMMENDATIONS FROM WIDER CINEMA (OUTSIDE DIARY):"
            )
            for cit in wider_citations[:3]:
                context_parts.append(
                    f'- {cit.title} ({cit.year}) [{cit.source_portal}]: "{cit.excerpt}"'
                )

        if taste_profile is not None:
            taste_notes = []
            if taste_profile.liked_directors:
                taste_notes.append(
                    f"Directors they love: {', '.join(taste_profile.liked_directors)}"
                )
            if taste_profile.liked_genres:
                taste_notes.append(
                    f"Genres they enjoy: {', '.join(taste_profile.liked_genres)}"
                )
            if taste_profile.disliked_elements:
                taste_notes.append(
                    "Elements/tropes they dislike/avoid: "
                    f"{', '.join(taste_profile.disliked_elements)}"
                )
            if taste_notes:
                context_parts.append(
                    "\nUSER'S ACTIVE TASTE PROFILE:\n"
                    + "\n".join(f"- {note}" for note in taste_notes)
                )

        if critic_citations:
            context_parts.append(
                "\nREPUTED CRITIC REVIEWS & CONSENSUS "
                "(RogerEbert.com, Variety, The Independent, The New York Times, "
                "The Hollywood Reporter, The Guardian, Rotten Tomatoes, Metacritic):"
            )
            for cc in critic_citations[:3]:
                author = f" by {cc.critic_name}" if cc.critic_name else ""
                context_parts.append(
                    f'- {cc.movie_title} on {cc.portal_name}{author}: "{cc.excerpt}"'
                )

        if web_results:
            context_parts.append("\nLIVE INTERNET / WEB SEARCH CONTEXT:")
            for wr in web_results[:3]:
                context_parts.append(f"- {wr.get('title')}: {wr.get('snippet')}")

        context_str = "\n".join(context_parts) if context_parts else "None available."

        if target_movie is not None:
            rating_val = target_movie.rating or 0.0
            rating_str = (
                f"★ {target_movie.rating:.1f}" if target_movie.rating else "Logged"
            )
            if rating_val <= 2.0:
                sentiment_inst = (
                    f"You HATED this movie! You gave it a miserable ★ {rating_val:.1f} "
                    "in your diary. Do NOT recommend it or say 'definitely check out'. "
                    "Be brutally honest and scathing about why you loathed it, "
                    f"quoting your review: '{target_movie.review_text[:160]}'. "
                    "Warn the user and roast it."
                )
            elif rating_val >= 4.0:
                sentiment_inst = (
                    f"You LOVED this movie (★ {rating_val:.1f})! Express your "
                    "passionate reaction and highlight what resonated with you."
                )
            else:
                sentiment_inst = (
                    f"You had mixed/moderate feelings (★ {rating_val:.1f}). "
                    "Share your balanced take."
                )

            user_content = (
                f"User message: {message}\n\n"
                f"TARGET MOVIE UNDER DISCUSSION: {target_movie.title} "
                f"({target_movie.year}) [{rating_str}]\n"
                f'YOUR DIARY REVIEW: "{target_movie.review_text}"\n\n'
                f"{context_str}\n\n"
                f"CRITICAL DIRECTIVE: The user is asking specifically about "
                f"'{target_movie.title}' ({target_movie.year}). "
                "Focus your entire response on this specific movie. DO NOT recommend "
                "other films or provide a generic recommendation list. "
                f"{sentiment_inst} "
                "Weave in reputed critic consensus if available. End with an engaging "
                "conversational question specifically about this movie."
            )
        elif target_title:
            user_content = (
                f"User message: {message}\n\n"
                f"{context_str}\n\n"
                f"CRITICAL DIRECTIVE: The user is asking specifically about "
                f"'{target_title.title()}'. You have NOT logged this film in your "
                "Letterboxd diary yet. State that honestly, share what reputed critics "
                "say about it, and ask the user if they've seen it or if they "
                "recommend you watch it. DO NOT give a generic recommendation list."
            )
        else:
            user_content = (
                f"User message: {message}\n\n"
                f"{context_str}\n\n"
                "Now respond as PG directly to the user in your authentic voice. "
                "You can draw from both your personal Letterboxd diary and acclaimed "
                "wider cinema outside your diary that matches their taste. "
                "Actively honor their taste profile (avoiding what they dislike and "
                "leaning into what they like) and synthesize reputed critic consensus."
            )

        msg_list: list[Any] = [SystemMessage(content=CURATOR_PERSONA_PROMPT)]
        if conversation_history:
            for turn in conversation_history[-4:]:
                r = turn.get("role")
                c = turn.get("content", "")
                if r == "user" and c:
                    msg_list.append(HumanMessage(content=c))
                elif r == "assistant" and c:
                    msg_list.append(AIMessage(content=c))

        msg_list.append(HumanMessage(content=user_content))
        return msg_list

    async def _generate_dialogue(
        self,
        message: str,
        citations: list[MovieCitation],
        critic_citations: list[CriticReviewCitation],
        web_results: list[dict[str, Any]],
        taste_profile: UserTasteProfile | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        is_correction: bool = False,
        target_movie: MovieRecord | None = None,
        target_title: str | None = None,
    ) -> str:
        """Generates conversational dialogue using real LLM with offline fallback."""
        if self.llm is not None:
            try:
                import asyncio

                messages = self._build_llm_messages(
                    message,
                    citations,
                    critic_citations,
                    web_results,
                    taste_profile,
                    conversation_history=conversation_history,
                    target_movie=target_movie,
                    target_title=target_title,
                )
                res = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=6.0)
                text = _extract_text(res.content)
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning("LLM generation error/429, falling back to local: %s", e)

        # Fallback offline generator for CI and offline environments
        return self._offline_dialogue_synthesis(
            message,
            citations,
            critic_citations,
            web_results,
            taste_profile,
            is_correction=is_correction,
            target_movie=target_movie,
            target_title=target_title,
        )

    def _offline_dialogue_synthesis(
        self,
        message: str,
        citations: list[MovieCitation],
        critic_citations: list[CriticReviewCitation],
        web_results: list[dict[str, Any]],
        taste_profile: UserTasteProfile | None = None,
        is_correction: bool = False,
        target_movie: MovieRecord | None = None,
        target_title: str | None = None,
    ) -> str:
        """Dynamic conversational synthesis for offline fallback or test suites."""
        # Single targeted movie discussion (from PG's diary)
        if target_movie is not None:
            clean_review = (
                target_movie.review_text.strip()
                .replace("&#039;", "'")
                .replace("&#39;", "'")
                .replace("&quot;", '"')
                .replace("&amp;", "&")
            )
            clean_review = re.sub(r'"+', '"', clean_review).strip()

            if len(clean_review) > 280:
                cut = clean_review[:280]
                last_punct = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
                if last_punct > 100:
                    snippet = cut[: last_punct + 1]
                else:
                    snippet = cut.rsplit(" ", 1)[0] + "..."
            else:
                snippet = clean_review

            critic_part = ""
            for cc in critic_citations:
                # Only include critic excerpts that are real, not the generic fallback
                if (
                    cc.movie_title.lower() in target_movie.title.lower()
                    or target_movie.title.lower() in cc.movie_title.lower()
                ) and not _is_generic_critic_excerpt(cc.excerpt):
                    critic_part = (
                        f" Reputed critics at {cc.portal_name} also pointed out: "
                        f'"{cc.excerpt.rstrip(".")}."'
                    )
                    break

            if target_movie.rating is not None and target_movie.rating <= 2.0:
                paragraphs = [
                    (
                        "Honestly? I absolutely loathed it. "
                        f"I gave *{target_movie.title}* ({target_movie.year}) "
                        f"a miserable ★ {target_movie.rating:.1f} on Letterboxd—and "
                        "I stand by every bit of that half-star rating."
                    ),
                    (
                        "When I logged it in my diary, my immediate reaction was: "
                        f'"{snippet}"{critic_part}'
                    ),
                    (
                        "Did you actually sit through all of it, or were you wondering "
                        "whether you should steer well clear?"
                    ),
                ]
                return "\n\n".join(paragraphs)
            elif target_movie.rating is not None and target_movie.rating >= 4.0:
                paragraphs = [
                    (
                        f"I absolutely love *{target_movie.title}* "
                        f"({target_movie.year})! I logged it at a solid "
                        f"★ {target_movie.rating:.1f} in my Letterboxd diary."
                    ),
                    (f'My take when I watched it: "{snippet}"{critic_part}'),
                    (
                        "What did you think of it, or are you planning to check it "
                        "out soon?"
                    ),
                ]
                return "\n\n".join(paragraphs)
            else:
                rating_str = (
                    f"★ {target_movie.rating:.1f}" if target_movie.rating else "logged"
                )
                paragraphs = [
                    (
                        f"I logged *{target_movie.title}* ({target_movie.year}) at "
                        f"{rating_str} in my Letterboxd diary."
                    ),
                    (f'My take on it was: "{snippet}"{critic_part}'),
                    ("How did it land for you?"),
                ]
                return "\n\n".join(paragraphs)

        # Single targeted movie discussion (not yet logged in diary)
        if target_title and target_movie is None:
            title_display = target_title.title()
            critic_part = ""
            if critic_citations:
                cc = critic_citations[0]
                if not _is_generic_critic_excerpt(cc.excerpt):
                    critic_part = (
                        f" Looking at reputed critic consensus from {cc.portal_name}: "
                        f'"{cc.excerpt.rstrip(".")}."'
                    )
            elif web_results:
                wr = web_results[0]
                critic_part = (
                    f' From what film publications note: "{wr.get("snippet", "")}"'
                )

            paragraphs = [
                (
                    f"I actually haven't logged *{title_display}* in my Letterboxd "
                    "diary yet!"
                ),
                (
                    f"{critic_part}"
                    if critic_part
                    else "It's one I've got on my radar to catch up with."
                ),
                ("Have you watched it? Would you recommend I add it to my watchlist?"),
            ]
            return "\n\n".join(paragraphs)

        # Filter out low-rated diary entries from discovery responses
        # (never recommend something PG rated ≤ 2.5 in a discovery context)
        good_citations = [
            c for c in citations
            if c.source_type != "letterboxd" or (c.rating is not None and c.rating >= 2.5)
        ]

        if not good_citations and not web_results:
            return (
                "Nothing in my diary is jumping out as a strong match for that right "
                "now. Give me a bit more to go on—what directors, moods, or "
                "specific vibes are you after?"
            )

        if is_correction:
            intro = (
                "You're right, my bad. Let me think of something that actually fits."
            )
        else:
            # Derive a natural opener from the top citation rather than hardcoded genre text
            top = good_citations[0] if good_citations else None
            if top and top.source_type == "letterboxd" and top.rating and top.rating >= 4.0:
                intro = (
                    f"*{top.title}* is the first thing that comes to mind honestly."
                )
            elif top and top.source_type == "acclaimed_cinema":
                intro = (
                    f"Something from outside my diary that fits perfectly is "
                    f"*{top.title}*."
                )
            else:
                intro = "Here's what I'd genuinely point you towards."

        commentary = []
        for i, cit in enumerate(good_citations[:3]):
            clean_snippet = cit.excerpt.strip().rstrip(".").strip()
            clean_snippet = (
                clean_snippet.replace("&#039;", "'")
                .replace("&#39;", "'")
                .replace("&quot;", '"')
                .replace("&amp;", "&")
            )

            if cit.source_type == "acclaimed_cinema":
                rating_tag = f" [{cit.source_portal}]"
                # For acclaimed cinema use the excerpt only if it's a real description
                if clean_snippet and len(clean_snippet) > 40 and not clean_snippet.startswith("The ") or len(clean_snippet) > 80:
                    take = f'"{clean_snippet}."'
                else:
                    take = "Critically acclaimed and widely celebrated."
                source_note = "Beyond my diary"
            else:
                rating_tag = f" (★ {cit.rating:.1f})" if cit.rating else ""
                source_note = "In my diary"
                if clean_snippet and not clean_snippet.startswith("Rated "):
                    take = f'"{clean_snippet}."'
                else:
                    take = "It really resonated with me."

            critic_addon = ""
            for cc in critic_citations:
                if (
                    cc.movie_title.lower() in cit.title.lower()
                    or cit.title.lower() in cc.movie_title.lower()
                ) and not _is_generic_critic_excerpt(cc.excerpt):
                    critic_addon = (
                        f" {cc.portal_name} wrote: "
                        f'"{cc.excerpt.rstrip(".")}."'
                    )
                    break

            if i == 0:
                commentary.append(
                    f"*{cit.title}* ({cit.year}){rating_tag} — {take}{critic_addon}"
                )
            else:
                commentary.append(
                    f"{source_note}, *{cit.title}* ({cit.year}){rating_tag} — {take}{critic_addon}"
                )

        body = "  \n".join(commentary)
        closing = "Have you seen any of these, or should we dig into a different direction?"
        return f"{intro}\n\n{body}\n\n{closing}"

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

        history = self.conversation_history.get(conversation_id, [])

        # LLM-powered intent classification (falls back to regex offline)
        intent_cls = await self._classify_intent(message, history, self.retriever.catalog)
        is_correction = intent_cls.intent == "correction"
        excluded_titles: set[str] = {
            t.lower() for t in intent_cls.rejected_titles
        }
        is_target = intent_cls.intent in ("single_movie_query", "debate")
        target_rec: MovieRecord | None = None
        target_title: str | None = None
        if is_target:
            target_rec, target_title = _resolve_catalog_movie(
                intent_cls.movie_title, intent_cls.movie_year, self.retriever.catalog
            )

        if is_target:
            citations = (
                [SearchResult(record=target_rec, score=1.0).to_citation()]
                if target_rec
                else []
            )
            if target_rec:
                critic_citations = fetch_reputed_critic_reviews(
                    target_rec.title, target_rec.year
                )
            else:
                critic_citations = fetch_reputed_critic_reviews(target_title or "")

            web_results = []
            if not target_rec and target_title:
                web_results = search_cinema_web(
                    f"{target_title} film review reception", limit=2
                )

            state.citations = citations
            state.critic_citations = critic_citations
            state.web_results = web_results

            state.final_response = await self._generate_dialogue(
                message=message,
                citations=citations,
                critic_citations=critic_citations,
                web_results=web_results,
                taste_profile=profile,
                conversation_history=history,
                is_correction=is_correction,
                target_movie=target_rec,
                target_title=target_title,
            )

            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            self.conversation_history[conversation_id].append(
                {"role": "user", "content": message}
            )
            self.conversation_history[conversation_id].append(
                {"role": "assistant", "content": state.final_response}
            )
            return state

        # 3. Retrieve relevant movies from PG's reviewed catalog (with taste profile)
        results = self.retriever.search(
            message,
            top_k=2,
            taste_profile=profile,
            excluded_titles=excluded_titles,
        )

        # Cyclic query broadening if 0 matches
        if not results:
            words = [
                w
                for w in message.split()
                if len(w) > 3
                and w.lower() not in excluded_titles
                and w.lower() not in CINEMA_STOPWORDS
            ]
            if words:
                broad_query = " ".join(words[:3])
                results = self.retriever.search(
                    broad_query,
                    top_k=2,
                    taste_profile=profile,
                    excluded_titles=excluded_titles,
                )

        diary_citations = [res.to_citation() for res in results]

        # 4. Retrieve acclaimed wider cinema recommendations matching user taste
        catalog_titles = {rec.title.lower() for rec in self.retriever.catalog}
        wider_citations = find_acclaimed_wider_cinema(
            message,
            taste_profile=profile,
            limit=2,
            catalog_titles=catalog_titles,
            excluded_titles=excluded_titles,
        )

        citations = diary_citations + wider_citations
        state.citations = citations

        # 5. Search web if needed
        web_results: list[dict[str, Any]] = []
        if not citations or any(
            w in message.lower()
            for w in ["director", "who directed", "actor", "release", "upcoming"]
        ):
            web_results = search_cinema_web(message, limit=3)
        state.web_results = web_results

        # 6. Multi-Source Retrieval: Fetch verified critic reviews from reputed portals
        critic_citations: list[CriticReviewCitation] = []
        for cit in citations[:3]:
            if cit.source_type == "acclaimed_cinema":
                critic_citations.append(
                    CriticReviewCitation(
                        movie_title=cit.title,
                        portal_name=cit.source_portal or "Rotten Tomatoes",
                        critic_name="Critical Consensus",
                        excerpt=cit.excerpt,
                        review_url=cit.letterboxd_url,
                        score_or_consensus="Acclaimed",
                    )
                )
            else:
                c_reviews = fetch_reputed_critic_reviews(cit.title, cit.year)
                critic_citations.extend(c_reviews)
        state.critic_citations = critic_citations

        state.final_response = await self._generate_dialogue(
            message=message,
            citations=citations,
            critic_citations=critic_citations,
            web_results=web_results,
            taste_profile=profile,
            conversation_history=history,
            is_correction=is_correction,
        )

        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        self.conversation_history[conversation_id].append(
            {"role": "user", "content": message}
        )
        self.conversation_history[conversation_id].append(
            {"role": "assistant", "content": state.final_response}
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

        # 3. Classify intent & retrieve
        history = self.conversation_history.get(conversation_id, [])
        intent_cls = await self._classify_intent(message, history, self.retriever.catalog)
        is_correction = intent_cls.intent == "correction"
        excluded_titles: set[str] = {t.lower() for t in intent_cls.rejected_titles}
        is_target = intent_cls.intent in ("single_movie_query", "debate")
        target_rec: MovieRecord | None = None
        target_title: str | None = None
        if is_target:
            target_rec, target_title = _resolve_catalog_movie(
                intent_cls.movie_title, intent_cls.movie_year, self.retriever.catalog
            )

        if is_target:
            citations = (
                [SearchResult(record=target_rec, score=1.0).to_citation()]
                if target_rec
                else []
            )
            if target_rec:
                critic_citations = fetch_reputed_critic_reviews(
                    target_rec.title, target_rec.year
                )
            else:
                critic_citations = fetch_reputed_critic_reviews(target_title or "")

            web_results = []
            if not target_rec and target_title:
                web_results = search_cinema_web(
                    f"{target_title} film review reception", limit=2
                )
        else:
            results = self.retriever.search(
                message,
                top_k=2,
                taste_profile=profile,
                excluded_titles=excluded_titles,
            )
            if not results:
                words = [
                    w
                    for w in message.split()
                    if len(w) > 3
                    and w.lower() not in excluded_titles
                    and w.lower() not in CINEMA_STOPWORDS
                ]
                if words:
                    broad_query = " ".join(words[:3])
                    results = self.retriever.search(
                        broad_query,
                        top_k=2,
                        taste_profile=profile,
                        excluded_titles=excluded_titles,
                    )

            diary_citations = [res.to_citation() for res in results]

            # 4. Acclaimed wider cinema recommendations matching user taste
            catalog_titles = {rec.title.lower() for rec in self.retriever.catalog}
            wider_citations = find_acclaimed_wider_cinema(
                message,
                taste_profile=profile,
                limit=2,
                catalog_titles=catalog_titles,
                excluded_titles=excluded_titles,
            )

            citations = diary_citations + wider_citations

            # 5. Search web if needed
            web_results = []
            if not citations or any(
                w in message.lower()
                for w in ["director", "who directed", "actor", "release", "upcoming"]
            ):
                web_results = search_cinema_web(message, limit=3)

            # 6. Multi-Source: Fetch verified reviews from reputed portals
            critic_citations = []
            for cit in citations[:3]:
                if cit.source_type == "acclaimed_cinema":
                    critic_citations.append(
                        CriticReviewCitation(
                            movie_title=cit.title,
                            portal_name=cit.source_portal or "Rotten Tomatoes",
                            critic_name="Critical Consensus",
                            excerpt=cit.excerpt,
                            review_url=cit.letterboxd_url,
                            score_or_consensus="Acclaimed",
                        )
                    )
                else:
                    c_reviews = fetch_reputed_critic_reviews(cit.title, cit.year)
                    critic_citations.extend(c_reviews)

        # 7. Stream LLM tokens directly if LLM is active
        streamed_any = False
        generated_tokens: list[str] = []
        fallback_text = ""  # Always initialised to avoid NameError in any branch
        if self.llm is not None:
            try:
                import asyncio

                messages = self._build_llm_messages(
                    message,
                    citations,
                    critic_citations,
                    web_results,
                    taste_profile=profile,
                    conversation_history=history,
                    target_movie=target_rec if is_target else None,
                    target_title=target_title if is_target else None,
                )

                async def _get_first_chunk(it: Any) -> Any:
                    return await it.__anext__()

                iterator = self.llm.astream(messages)
                try:
                    first_chunk = await asyncio.wait_for(
                        _get_first_chunk(iterator), timeout=5.0
                    )
                    first_text = _extract_text(first_chunk.content)
                    if first_text:
                        streamed_any = True
                        generated_tokens.append(first_text)
                        yield {"event": "token", "data": first_text}

                    async for chunk in iterator:
                        try:
                            token_text = _extract_text(chunk.content)
                        except Exception as chunk_err:
                            # Gemini can raise "model output must contain either output
                            # text or tool calls" on empty/filtered chunks — skip them.
                            logger.debug("Skipping bad chunk: %s", chunk_err)
                            continue
                        if token_text:
                            streamed_any = True
                            generated_tokens.append(token_text)
                            yield {"event": "token", "data": token_text}
                except Exception as stream_err:
                    err_str = str(stream_err).lower()
                    is_quota = any(
                        kw in err_str
                        for kw in [
                            "quota", "resource_exhausted", "429",
                            "rate limit", "ratelimit", "too many requests",
                            "model output must contain",
                        ]
                    )
                    logger.warning(
                        "LLM stream error (quota=%s), falling back: %s",
                        is_quota,
                        stream_err,
                    )
                    streamed_any = False
                    generated_tokens.clear()
                    if is_quota:
                        # Surface an honest, human-readable message instead of
                        # producing robotic offline synthesis.
                        quota_msg = (
                            "Hey, my AI brain seems to be momentarily overloaded — "
                            "hit my usage limit for right now. Give it a minute and "
                            "try again? I promise I'll have a proper answer for you."
                        )
                        generated_tokens.append(quota_msg)
                        for word in quota_msg.split(" "):
                            yield {"event": "token", "data": word + " "}
                        streamed_any = True  # mark as handled, skip offline synthesis
            except Exception as e:
                logger.warning(
                    "Streaming LLM setup error, falling back: %s", e
                )

        if not streamed_any:
            fallback_text = self._offline_dialogue_synthesis(
                message,
                citations,
                critic_citations,
                web_results,
                taste_profile=profile,
                is_correction=is_correction,
                target_movie=target_rec if is_target else None,
                target_title=target_title if is_target else None,
            )
            generated_tokens.append(fallback_text)
            for word in fallback_text.split(" "):
                yield {"event": "token", "data": word + " "}

        full_response = "".join(generated_tokens) if streamed_any else fallback_text
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        self.conversation_history[conversation_id].append(
            {"role": "user", "content": message}
        )
        self.conversation_history[conversation_id].append(
            {"role": "assistant", "content": full_response}
        )

        # 8. Emit citations event (Letterboxd & Acclaimed Cinema)
        if citations:
            yield {
                "event": "citations",
                "data": [c.model_dump() for c in citations],
            }

        # 9. Emit critic citations event — filter out generic placeholder excerpts
        real_critic_citations = [
            cc for cc in critic_citations
            if not _is_generic_critic_excerpt(cc.excerpt)
        ]
        if real_critic_citations:
            yield {
                "event": "critic_citations",
                "data": [cc.model_dump() for cc in real_critic_citations],
            }

        yield {"event": "done", "data": "[DONE]"}
