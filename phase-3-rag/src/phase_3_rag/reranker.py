"""Cross-Encoder Reranker for two-stage RAG retrieval.

Cross-encoders evaluate Query and Document simultaneously via cross-attention,
scoring the deep contextual relevance of each candidate chunk.
"""

import json
import re
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel, Field

from phase_3_rag.bm25 import tokenize
from phase_3_rag.chunking import Chunk
from phase_3_rag.config import get_settings


class RerankResult(BaseModel):
    """Result of cross-encoder reranking for a single chunk."""

    chunk: Chunk = Field(description="The underlying document chunk")
    relevance_score: float = Field(
        description="Cross-encoder relevance score (0.0 to 1.0)"
    )
    reranked_rank: int = Field(description="1-based rank after cross-encoding")
    original_rank: int = Field(description="1-based candidate rank before reranking")
    reason: str = Field(
        default="", description="Reasoning or explanation for relevance score"
    )


class BaseReranker(ABC):
    """Abstract interface for all cross-encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        *,
        top_k: int = 8,
    ) -> list[RerankResult]:
        """Rerank candidate chunks against the query and return top-k results."""
        pass


class HeuristicCrossEncoder(BaseReranker):
    """Fast, deterministic cross-attention heuristic reranker.

    Computes query-chunk cross-features:
    1. Exact query phrase matching.
    2. Query term coverage (recall of query terms in chunk).
    3. Query term proximity and density.
    4. Technical acronym and identifier preservation.

    Useful for offline testing, CI environments, and low-latency benchmarks.
    """

    def score_pair(self, query: str, chunk_text: str) -> tuple[float, str]:
        """Compute contextual cross-encoder score between 0.0 and 1.0."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0, "Empty query"

        text_lower = chunk_text.lower()
        chunk_tokens = tokenize(chunk_text)
        if not chunk_tokens:
            return 0.0, "Empty chunk"

        # Feature 1: Exact query substring match
        clean_q = " ".join(q_tokens)
        phrase_bonus = 0.35 if clean_q in text_lower else 0.0

        # Feature 2: Query term coverage (recall)
        present_terms = [t for t in set(q_tokens) if t in text_lower]
        coverage_ratio = len(present_terms) / len(set(q_tokens))

        # Feature 3: Term frequency density (frequency of query terms / chunk size)
        matches_count = sum(chunk_tokens.count(t) for t in present_terms)
        density = min(matches_count / (len(chunk_tokens) + 10) * 10, 0.25)

        # Feature 4: Acronym and numeric preservation (e.g., LCH4, ERV, 85, 72)
        code_tokens = [t for t in q_tokens if re.search(r"[0-9_-]", t) or len(t) <= 4]
        code_bonus = 0.0
        if code_tokens:
            matched_codes = [t for t in code_tokens if t in text_lower]
            code_bonus = 0.20 * (len(matched_codes) / len(code_tokens))

        raw_score = (coverage_ratio * 0.40) + phrase_bonus + density + code_bonus
        final_score = min(max(raw_score, 0.0), 1.0)

        reason = (
            f"Coverage: {coverage_ratio:.0%}, PhraseMatch: {bool(phrase_bonus)}, "
            f"Matches: {matches_count}"
        )
        return round(final_score, 4), reason

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        *,
        top_k: int = 8,
    ) -> list[RerankResult]:
        """Score all candidates and return top-k sorted descending."""
        scored: list[tuple[float, int, Chunk, str]] = []
        for orig_rank, chunk in enumerate(candidates, start=1):
            score, reason = self.score_pair(query, chunk.text)
            scored.append((score, orig_rank, chunk, reason))

        # Sort descending by cross-encoder score
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RerankResult] = []
        for new_rank, (score, orig_rank, chunk, reason) in enumerate(
            scored[:top_k], start=1
        ):
            results.append(
                RerankResult(
                    chunk=chunk,
                    relevance_score=score,
                    reranked_rank=new_rank,
                    original_rank=orig_rank,
                    reason=reason,
                )
            )
        return results


class GeminiCrossEncoder(BaseReranker):
    """LLM-based Cross-Encoder using Gemini for full cross-attention scoring.

    Sends candidate (query, passage) pairs to Gemini with structured schema
    to grade contextual relevance on a 0.0 to 1.0 scale.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 15,
    ) -> None:
        self.settings = get_settings()
        self.client = client
        self.model = model or self.settings.gemini_model
        self.api_key = api_key or self.settings.gemini_api_key
        self.batch_size = batch_size
        self._fallback = HeuristicCrossEncoder()

    def _score_batch(
        self,
        query: str,
        batch: list[tuple[int, Chunk]],
        client: httpx.Client,
    ) -> dict[str, tuple[float, str]]:
        """Send a batch of candidates to Gemini for cross-encoder scoring."""
        passages_text = []
        for orig_rank, chunk in batch:
            passages_text.append(f"[ID: {chunk.chunk_id}]\n{chunk.text.strip()}\n")
        joined_passages = "\n---\n".join(passages_text)

        prompt = (
            "You are a high-precision Cross-Encoder Reranker for technical manuals.\n"
            "Evaluate each candidate passage against the user query.\n"
            "Assign a relevance score from 0.0 (unrelated) to 1.0 (directly and "
            "completely answers the query).\n"
            "Return ONLY a JSON array of objects with fields:\n"
            "  - chunk_id (string)\n"
            "  - relevance_score (float between 0.0 and 1.0)\n"
            "  - reason (concise explanation)\n\n"
            f"Query: {query}\n\n"
            f"Candidate Passages:\n{joined_passages}\n\n"
            "JSON Output:"
        )

        url = f"{self.settings.gemini_base_url}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key or "",
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return {}

        raw_json = candidates[0]["content"]["parts"][0].get("text", "[]")
        parsed = json.loads(raw_json)

        scores: dict[str, tuple[float, str]] = {}
        for item in parsed:
            cid = item.get("chunk_id", "")
            score = float(item.get("relevance_score", 0.0))
            reason = item.get("reason", "")
            scores[cid] = (score, reason)
        return scores

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        *,
        top_k: int = 8,
    ) -> list[RerankResult]:
        """Rerank candidates using Gemini cross-attention with heuristic fallback."""
        if not candidates:
            return []

        should_close = False
        client = self.client
        if client is None:
            client = httpx.Client(timeout=self.settings.timeout_seconds)
            should_close = True

        try:
            scores: dict[str, tuple[float, str]] = {}
            candidate_pairs = list(enumerate(candidates, start=1))

            # Process in batches of 15 to balance latency and context limits
            for i in range(0, len(candidate_pairs), self.batch_size):
                batch = candidate_pairs[i : i + self.batch_size]
                try:
                    batch_scores = self._score_batch(query, batch, client)
                    scores.update(batch_scores)
                except Exception:
                    # Fall back to heuristic for this batch if LLM call fails
                    for _, c in batch:
                        h_score, h_reason = self._fallback.score_pair(query, c.text)
                        scores[c.chunk_id] = (h_score, f"Fallback: {h_reason}")

            scored_candidates: list[tuple[float, int, Chunk, str]] = []
            for orig_rank, chunk in candidate_pairs:
                score, reason = scores.get(chunk.chunk_id, (0.0, "Unscored"))
                scored_candidates.append((score, orig_rank, chunk, reason))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            results: list[RerankResult] = []
            for new_rank, (score, orig_rank, chunk, reason) in enumerate(
                scored_candidates[:top_k], start=1
            ):
                results.append(
                    RerankResult(
                        chunk=chunk,
                        relevance_score=score,
                        reranked_rank=new_rank,
                        original_rank=orig_rank,
                        reason=reason,
                    )
                )
            return results

        finally:
            if should_close:
                client.close()
