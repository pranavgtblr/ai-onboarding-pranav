"""Phase 1: LLM Fundamentals package."""

from phase_1_llm_fundamentals.embeddings import (
    EmbeddingSettings,
    SentenceItem,
    SimilarityMatrixResult,
    compute_similarity_matrix,
    cosine_similarity,
    dot_product,
    embed_corpus,
    embed_single_text,
    vector_magnitude,
)
from phase_1_llm_fundamentals.temperature_experiment import (
    ExperimentOutput,
    PromptRunResult,
    Settings,
    TemperatureSummary,
    execute_temperature_experiment,
)
from phase_1_llm_fundamentals.tokenizer import (
    TokenStats,
    calculate_token_stats,
    compare_tokenization,
    decode_tokens_to_pieces,
    tokenize_text,
)

__all__ = [
    "TokenStats",
    "tokenize_text",
    "decode_tokens_to_pieces",
    "calculate_token_stats",
    "compare_tokenization",
    "Settings",
    "PromptRunResult",
    "TemperatureSummary",
    "ExperimentOutput",
    "execute_temperature_experiment",
    "EmbeddingSettings",
    "SentenceItem",
    "SimilarityMatrixResult",
    "dot_product",
    "vector_magnitude",
    "cosine_similarity",
    "embed_single_text",
    "embed_corpus",
    "compute_similarity_matrix",
]
