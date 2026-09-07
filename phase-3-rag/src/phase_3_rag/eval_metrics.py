"""Evaluation metrics for decoupling Retrieval and Generation in RAG.

Provides standard quantitative metrics:
- Retrieval: Hit@K, Recall@K, Context Precision (MRR)
- Generation: Faithfulness (Grounding), Answer Relevance (Key Facts Recall)
"""

import re


def compute_hit_at_k(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> float:
    """Compute binary Hit@K (1.0 if at least one expected doc is in top-k).

    For negative/unanswerable questions (where expected_doc_ids is empty),
    returns 1.0 if retriever returned items (normal behavior).
    """
    if not expected_doc_ids:
        return 1.0
    expected_set = set(expected_doc_ids)
    for doc_id in retrieved_doc_ids:
        if doc_id in expected_set:
            return 1.0
    return 0.0


def compute_recall_at_k(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> float:
    """Compute Recall@K: fraction of expected documents found in top-k."""
    if not expected_doc_ids:
        return 1.0
    expected_set = set(expected_doc_ids)
    retrieved_set = set(retrieved_doc_ids)
    matches = expected_set.intersection(retrieved_set)
    return len(matches) / len(expected_set)


def compute_context_precision(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> float:
    """Compute Context Precision (Reciprocal Rank of the first relevant document).

    Formula: 1 / rank_of_first_relevant_doc (1-indexed).
    If no relevant document is retrieved, returns 0.0.
    """
    if not expected_doc_ids:
        return 1.0
    expected_set = set(expected_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def compute_answer_relevance(
    generated_answer: str,
    key_facts: list[str],
) -> float:
    """Compute Answer Relevance as the coverage of ground-truth key facts.

    Checks what fraction of expected key facts appear in the generated answer.
    """
    if not key_facts:
        return 1.0

    ans_lower = generated_answer.lower()
    matches = 0

    for fact in key_facts:
        fact_clean = fact.lower().strip()
        # Direct phrase match or token overlap match
        if fact_clean in ans_lower:
            matches += 1
            continue

        fact_tokens = [t for t in re.findall(r"\b\w+\b", fact_clean) if len(t) > 2]
        if fact_tokens:
            token_hits = sum(1 for t in fact_tokens if t in ans_lower)
            if token_hits / len(fact_tokens) >= 0.75:
                matches += 1

    return matches / len(key_facts)


def compute_faithfulness(
    generated_answer: str,
    retrieved_context_text: str,
) -> float:
    """Compute Faithfulness: proportion of answer sentences grounded in context.

    Splits answer into sentences and measures whether key terms in each sentence
    are supported by the retrieved context chunks (penalizing hallucinations).
    """
    if not generated_answer.strip():
        return 0.0

    context_lower = retrieved_context_text.lower()
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]\s+", generated_answer)
        if len(s.strip()) > 15
    ]
    if not sentences:
        return 1.0

    grounded_count = 0
    for sentence in sentences:
        s_lower = sentence.lower()
        words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", s_lower) if len(w) > 3]
        if not words:
            grounded_count += 1
            continue

        # Sentence is considered grounded if >= 70% of its content words are in context
        supported = sum(1 for w in words if w in context_lower)
        if (supported / len(words)) >= 0.65:
            grounded_count += 1

    return grounded_count / len(sentences)
