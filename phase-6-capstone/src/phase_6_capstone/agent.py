"""Stateful LangGraph Agent for PG Recommends with Real LLM & Internet Search."""

import logging
import re
from collections.abc import AsyncGenerator
from typing import Any
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

MULTI-SOURCE SYNTHESIS:
You draw on two complementary sources of knowledge:
1. Your own personal Letterboxd diary and reviews.
2. Verified reviews & consensus from reputed online portals (RogerEbert.com,
   Variety, The Independent, The New York Times, The Hollywood Reporter,
   The Guardian, Rotten Tomatoes, Metacritic).
When external critic perspectives are provided below, weave them in! Tell the user
how the critical consensus aligns or contrasts with your own reaction.

HOW TO ENGAGE WITH THE USER:
1. Speak in the first person ("I", "my diary", "when I caught this", "honestly").
2. Address their specific prompt, mood, or curiosity naturally and enthusiastically.
3. Discuss 1-2 movies organically in your paragraphs—share your actual reaction,
   why you liked or disliked them, mention your rating casually (e.g. ★ 4.0), and
   briefly contrast with reputed critic consensus.
4. If internet/web search results are provided below, weave that knowledge in.
5. DO NOT provide a raw bulleted list or duplicate links—the cards below display
   structured ratings, Letterboxd links, and critic portal badges.
   Your job is genuine cinephile dialogue!
6. Always end with an engaging question to keep the conversation going with the user.
"""


def _detect_correction_and_exclusions(message: str) -> tuple[bool, set[str]]:
    """Detects if user is correcting the assistant or rejecting a title."""
    m_low = message.lower()
    is_correction = any(
        phrase in m_low
        for phrase in [
            "is not an",
            "is not a",
            "isn't an",
            "isn't a",
            "was not an",
            "wasn't an",
            "not an action",
            "not a movie",
            "not action",
            "not horror",
            "not comedy",
            "not what i asked",
            "that's not",
            "thats not",
            "wrong movie",
            "wrong genre",
            "documentary",
        ]
    )
    excluded: set[str] = set()
    match = re.search(r"([a-zA-Z0-9\s]+?)\s+(?:is not|isn't|wasn't|was not)", m_low)
    if match:
        name = match.group(1).strip()
        if len(name) > 2 and name not in CINEMA_STOPWORDS:
            excluded.add(name)
            if name.endswith("e"):
                excluded.add(name[:-1])
    return is_correction, excluded


def _detect_target_movie_query(
    message: str, catalog: list[MovieRecord]
) -> tuple[bool, MovieRecord | None, str | None]:
    """Detects if user is asking about a specific movie rather than recommendations."""
    m = message.strip().rstrip("?.,!").strip()
    m_low = m.lower()

    # If message contains recommendation/discovery keywords without direct query
    # prefix, it is a recommendation request
    discovery_triggers = [
        "recommend",
        "suggest",
        "movies like",
        "films like",
        "top movies",
        "best movies",
        "explore",
        "looking for",
        "something like",
        "give me",
        "show me",
    ]
    if any(dt in m_low for dt in discovery_triggers) and not any(
        m_low.startswith(p)
        for p in ["what about", "thoughts on", "have you seen", "what do you think"]
    ):
        return False, None, None

    # 1. Year pattern in message: e.g. 'Animal (2023)' or 'What about animal (2023)?'
    ym = re.search(r"([a-zA-Z0-9\s:,'-]+?)\s*\(((?:19|20)\d{2})\)", m)
    if ym:
        t_cand, y_cand = ym.group(1).strip().lower(), int(ym.group(2))
        t_clean = re.sub(
            r"^(?:what about|how about|thoughts on|what do you think of|"
            r"have you seen|have you watched|did you see|did you watch|"
            r"did you like|tell me about)\s+",
            "",
            t_cand,
        ).strip()
        for rec in catalog:
            if (
                rec.title.lower() == t_clean
                or t_clean in rec.title.lower()
                or rec.title.lower() in t_clean
            ) and rec.year == y_cand:
                return True, rec, rec.title
        for rec in catalog:
            if rec.title.lower() == t_clean or t_clean == rec.title.lower():
                return True, rec, rec.title
        return True, None, t_clean

    # 2. Explicit inquiry phrasing
    inquiry_patterns = [
        r"^(?:what about|how about)\s+(.+)",
        r"^(?:what do you think of|what did you think of)\s+(.+)",
        r"^(?:thoughts on|your thoughts on)\s+(.+)",
        r"^(?:have you seen|have you watched|did you see|did you watch)\s+(.+)",
        r"^(?:did you like|do you like)\s+(.+)",
        r"^(?:how is|how was)\s+(.+)",
        r"^(?:tell me about|review of|review for|rating for|what did you rate)\s+(.+)",
        r"^(?:why did you (?:give|rate))\s+(.+)",
    ]
    for pat in inquiry_patterns:
        mat = re.search(pat, m_low)
        if mat:
            cand = mat.group(1).strip()
            cand_clean = re.sub(r"\s*\(((?:19|20)\d{2})\)", "", cand).strip()
            for rec in catalog:
                if rec.title.lower() == cand_clean:
                    return True, rec, rec.title
            for rec in catalog:
                if len(cand_clean) >= 3 and (
                    cand_clean == rec.title.lower()
                    or rec.title.lower().startswith(cand_clean)
                ):
                    return True, rec, rec.title
            return True, None, cand_clean

    # 3. Direct title-only query: e.g. 'Animal', 'The Batman', 'Inception'
    words = m.split()
    if len(words) <= 4:
        for rec in catalog:
            if rec.title.lower() == m_low:
                return True, rec, rec.title

    return False, None, None


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
                if (
                    cc.movie_title.lower() in target_movie.title.lower()
                    or target_movie.title.lower() in cc.movie_title.lower()
                ):
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

        if not citations and not web_results:
            return (
                "I couldn't spot any films in my diary directly matching that. "
                "Give me a bit more to go on—what directors, moods, or "
                "tropes are you feeling?"
            )

        m_lower = message.lower()

        if is_correction:
            intro = (
                "You're completely right, my mistake on that! Let's correct course "
                "immediately and focus on genuine picks that fit what you're craving."
            )
        elif any(
            k in m_lower
            for k in [
                "action",
                "stunt",
                "martial arts",
                "combat",
                "thrills",
                "blockbuster",
            ]
        ):
            intro = (
                "If you're looking for adrenaline and kinetic craft, here are "
                "proper action movies with real weight, momentum, and impact."
            )
        elif any(
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
                "Let's talk cinema! Based on what you're craving and your taste, "
                "here's what I'd genuinely recommend diving into."
            )

        commentary = []
        for i, cit in enumerate(citations[:3]):
            clean_snippet = cit.excerpt.strip().rstrip(".").strip()
            clean_snippet = (
                clean_snippet.replace("&#039;", "'")
                .replace("&#39;", "'")
                .replace("&quot;", '"')
                .replace("&amp;", "&")
            )

            if cit.source_type == "acclaimed_cinema":
                rating_tag = f" [{cit.source_portal} Acclaimed]"
                take = f'Critical consensus celebrates: "{clean_snippet}."'
                source_intro = (
                    "Going beyond my personal diary into wider acclaimed cinema"
                )
            else:
                rating_tag = f" (logged it at ★ {cit.rating:.1f})" if cit.rating else ""
                source_intro = "From my own Letterboxd diary"
                if clean_snippet and not clean_snippet.startswith("Rated "):
                    take = f'My take on it was: "{clean_snippet}."'
                else:
                    take = "It's one that really resonated with me."

            critic_addon = ""
            for cc in critic_citations:
                if (
                    cc.movie_title.lower() in cit.title.lower()
                    or cit.title.lower() in cc.movie_title.lower()
                ):
                    critic_addon = (
                        f" Reputed critics at {cc.portal_name} also highlighted: "
                        f'"{cc.excerpt.rstrip(".")}."'
                    )
                    break

            rec_verb = (
                "definitely check out"
                if (cit.rating or 4.0) >= 3.5
                else "take a look at"
            )
            if i == 0:
                commentary.append(
                    f"{source_intro}, {rec_verb} *{cit.title}* ({cit.year})"
                    f"{rating_tag}. {take}{critic_addon}"
                )
            else:
                commentary.append(
                    f"{source_intro}, another standout is *{cit.title}* ({cit.year})"
                    f"{rating_tag}. {take}{critic_addon}"
                )

        paragraphs = [
            intro,
            " ".join(commentary),
            (
                "I've linked my diary logs and verified critic reception cards below "
                "so you can dig into the reviews and consensus. Have you seen any of "
                "these yet, or should we explore a different vibe?"
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

        is_correction, excluded_titles = _detect_correction_and_exclusions(message)
        history = self.conversation_history.get(conversation_id, [])

        # Check for targeted single-movie inquiry vs recommendation request
        is_target, target_rec, target_title = _detect_target_movie_query(
            message, self.retriever.catalog
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

        # 3. Retrieve local catalog movies (with taste profile)
        is_correction, excluded_titles = _detect_correction_and_exclusions(message)
        history = self.conversation_history.get(conversation_id, [])

        is_target, target_rec, target_title = _detect_target_movie_query(
            message, self.retriever.catalog
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
                first_chunk = await asyncio.wait_for(
                    _get_first_chunk(iterator), timeout=5.0
                )
                first_text = _extract_text(first_chunk.content)
                if first_text:
                    streamed_any = True
                    generated_tokens.append(first_text)
                    yield {"event": "token", "data": first_text}

                async for chunk in iterator:
                    token_text = _extract_text(chunk.content)
                    if token_text:
                        streamed_any = True
                        generated_tokens.append(token_text)
                        yield {"event": "token", "data": token_text}
            except Exception as e:
                logger.warning(
                    "Streaming LLM rate limit or timeout, falling back: %s", e
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

        # 9. Emit critic citations event (Reputed Online Portals)
        if critic_citations:
            yield {
                "event": "critic_citations",
                "data": [cc.model_dump() for cc in critic_citations],
            }

        yield {"event": "done", "data": "[DONE]"}
