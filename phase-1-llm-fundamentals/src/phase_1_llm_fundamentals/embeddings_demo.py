"""CLI demonstration for text embeddings and 10x10 cosine similarity matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_1_llm_fundamentals.embeddings import (
    EmbeddingSettings,
    SentenceItem,
    SimilarityMatrixResult,
    compute_similarity_matrix,
    embed_corpus,
)

# 10 Curated Sentences covering Pair A, Pair B, and thematic clusters:
CURATED_SENTENCES: list[SentenceItem] = [
    # Pair A: Same meaning, ZERO shared words
    SentenceItem(
        index=0,
        label="S1 (Pair A1)",
        text="The infant is asleep.",
        category="Infant Sleep",
    ),
    SentenceItem(
        index=1,
        label="S2 (Pair A2)",
        text="A newborn baby rests quietly.",
        category="Infant Sleep",
    ),
    # Pair B: Shared exact product code SKU-98421, different meanings
    SentenceItem(
        index=2,
        label="S3 (Pair B1)",
        text=(
            "Initiate a refund for defective widget SKU-98421 returned by the customer."
        ),
        category="Customer Support",
    ),
    SentenceItem(
        index=3,
        label="S4 (Pair B2)",
        text=(
            "Review the tensile strength specifications and steel "
            "alloy blueprint for component SKU-98421."
        ),
        category="Mechanical Engineering",
    ),
    # Thematic cluster: Culinary / Baking
    SentenceItem(
        index=4,
        label="S5 (Food 1)",
        text="Whisk the egg yolks with fresh cream and sugar until smooth.",
        category="Cooking",
    ),
    SentenceItem(
        index=5,
        label="S6 (Food 2)",
        text="Beat the milk, butter, and vanilla extract in a large glass bowl.",
        category="Cooking",
    ),
    # Thematic cluster: Software Development
    SentenceItem(
        index=6,
        label="S7 (Code 1)",
        text="Write a unit test for the REST API endpoint using pytest.",
        category="Software",
    ),
    SentenceItem(
        index=7,
        label="S8 (Code 2)",
        text="Debug the async route handler in FastAPI to prevent memory leaks.",
        category="Software",
    ),
    # Diverse domains
    SentenceItem(
        index=8,
        label="S9 (Astronomy)",
        text=(
            "Astronomers detected a supermassive black hole at the "
            "center of the distant galaxy."
        ),
        category="Astronomy",
    ),
    SentenceItem(
        index=9,
        label="S10 (Finance)",
        text=(
            "The central bank raised benchmark interest rates to "
            "combat rising inflation."
        ),
        category="Finance",
    ),
]


def print_matrix_table(result: SimilarityMatrixResult) -> None:
    """Print formatted 10x10 cosine similarity matrix in terminal."""
    print("=" * 115)
    print(
        f" 10x10 COSINE SIMILARITY MATRIX "
        f"(Model: {result.model} | Dimensions: {result.dimensions})"
    )
    print("=" * 115)
    print("\n--- Sentences Legend ---")
    for s in result.sentences:
        print(f'[{s.label:<13}] ({s.category:<18}): "{s.text}"')

    print("\n" + "=" * 115)
    header = " " * 15 + " | " + " | ".join(f"S{i + 1:<4}" for i in range(10))
    print(header)
    print("-" * 115)

    for i, row in enumerate(result.matrix):
        label = result.sentences[i].label
        scores = " | ".join(f"{score:.4f}" for score in row)
        print(f"{label:<15} | {scores}")
    print("=" * 115)

    print("\n--- Pair A Analysis (Zero-Word Overlap Semantic Equivalence) ---")
    print(f'  Sentence 1: "{result.pair_a_analysis["sentence_1"]}"')
    print(f'  Sentence 2: "{result.pair_a_analysis["sentence_2"]}"')
    print(f"  Shared Words: {result.pair_a_analysis['shared_words']} (Count: 0)")
    print(f"  Cosine Similarity: {result.pair_a_analysis['cosine_similarity']:.4f}")
    print(f"  Insight: {result.pair_a_analysis['phenomenon']}")

    print("\n--- Pair B Analysis (Shared Product Code SKU-98421, Divergent Intent) ---")
    print(f'  Sentence 1: "{result.pair_b_analysis["sentence_1"]}"')
    print(f'  Sentence 2: "{result.pair_b_analysis["sentence_2"]}"')
    print(f"  Shared Words: {result.pair_b_analysis['shared_words']}")
    print(f"  Cosine Similarity: {result.pair_b_analysis['cosine_similarity']:.4f}")
    print(f"  Insight: {result.pair_b_analysis['phenomenon']}")


def save_matrix_results(
    result: SimilarityMatrixResult,
    output_dir: Path | str = "outputs",
) -> tuple[Path, Path]:
    """Persist matrix and analysis to JSON and Markdown."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "embeddings_matrix.json"
    md_path = out_path / "embeddings_matrix.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# 10x10 Sentence Embeddings Cosine Similarity Matrix\n\n")
        f.write(f"- **Embedding Model:** `{result.model}`\n")
        f.write(f"- **Embedding Dimensions:** `{result.dimensions}`\n\n")
        f.write("## Sentences Index\n\n")
        for s in result.sentences:
            f.write(f'- **{s.label}** (`{s.category}`): *"{s.text}"*\n')

        f.write("\n---\n\n## Cosine Similarity Matrix\n\n")
        header_cols = " | ".join(f"**S{i + 1}**" for i in range(10))
        f.write(f"| Sentence | {header_cols} |\n")
        f.write("| :--- | " + " | ".join([":---:"] * 10) + " |\n")

        for i, row in enumerate(result.matrix):
            scores_col = " | ".join(
                f"**{val:.4f}**" if i == j else f"{val:.4f}"
                for j, val in enumerate(row)
            )
            f.write(f"| **{result.sentences[i].label}** | {scores_col} |\n")

        f.write("\n---\n\n## Key Findings & Anomaly Analysis\n\n")
        f.write("### 1. Pair A: Zero-Word Overlap Equivalence (S1 vs S2)\n\n")
        f.write(f'- **S1:** *"{result.pair_a_analysis["sentence_1"]}"*\n')
        f.write(f'- **S2:** *"{result.pair_a_analysis["sentence_2"]}"*\n')
        f.write("- **Shared Words Count:** `0` (Zero lexical overlap)\n")
        f.write(
            f"- **Cosine Similarity:** "
            f"**`{result.pair_a_analysis['cosine_similarity']:.4f}`**\n"
        )
        f.write(f"- **Analysis:** {result.pair_a_analysis['phenomenon']}\n\n")

        f.write("### 2. Pair B: Shared Product Code (S3 vs S4)\n\n")
        f.write(
            f'- **S3 (Customer Support):** *"{result.pair_b_analysis["sentence_1"]}"*\n'
        )
        f.write(
            f"- **S4 (Mechanical Engineering):** "
            f'*"{result.pair_b_analysis["sentence_2"]}"*\n'
        )
        f.write("- **Shared Identifiers:** `SKU-98421`, `for`\n")
        f.write(
            f"- **Cosine Similarity:** "
            f"**`{result.pair_b_analysis['cosine_similarity']:.4f}`**\n"
        )
        f.write(f"- **Analysis:** {result.pair_b_analysis['phenomenon']}\n")

    return json_path, md_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Text embeddings and 10x10 cosine similarity matrix benchmark"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save JSON and Markdown matrix outputs",
    )
    args = parser.parse_args()

    print("Generating embeddings for 10 sentences via Gemini Embedding API...")
    texts = [s.text for s in CURATED_SENTENCES]
    settings = EmbeddingSettings()
    embeddings = embed_corpus(texts, settings=settings)

    print("Computing 10x10 cosine similarity matrix...")
    matrix_result = compute_similarity_matrix(
        sentences=CURATED_SENTENCES,
        embeddings=embeddings,
        model_name=settings.gemini_embedding_model,
    )

    json_path, md_path = save_matrix_results(matrix_result, output_dir=args.output_dir)
    print_matrix_table(matrix_result)
    print(f"\nMatrix results saved to:\n- {json_path}\n- {md_path}")


if __name__ == "__main__":
    main()
