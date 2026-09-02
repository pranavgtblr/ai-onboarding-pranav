"""CLI demonstration of multilingual tokenization and cost disparity analysis."""

from __future__ import annotations

import argparse

from phase_1_llm_fundamentals.tokenizer import (
    calculate_cost_impact,
    compare_tokenization,
)

# Standard semantically equivalent paragraph across multiple languages:
# "Artificial intelligence is transforming how we build software,
# communicate across cultures, and solve complex global challenges.
# Large language models process human text through tokenization,
# converting words and subwords into numerical representations
# for neural networks."
SAMPLE_PARAGRAPHS: dict[str, str] = {
    "English": (
        "Artificial intelligence is transforming how we build software, "
        "communicate across cultures, and solve complex global challenges. "
        "Large language models process human text through tokenization, "
        "converting words and subwords into numerical representations "
        "for neural networks."
    ),
    "Malayalam": (
        "കൃത്രിമ ബുദ്ധി നാം സോഫ്റ്റ്‌വെയർ നിർമ്മിക്കുന്നതിലും "
        "വിവിധ സംസ്കാരങ്ങളുമായി ആശയവിനിമയം നടത്തുന്നതിലും "
        "സങ്കീർണ്ണമായ ആഗോള വെല്ലുവിളികൾ പരിഹരിക്കുന്നതിലും "
        "വിപ്ലവകരമായ മാറ്റങ്ങൾ വരുത്തുന്നു. "
        "വലിയ ഭാഷാ മോഡലുകൾ ടോക്കണൈസേഷൻ വഴി മനുഷ്യ ഭാഷയെ "
        "ന്യൂറൽ നെറ്റ്‌വർക്കുകൾക്കായുള്ള സംഖ്യാ രൂപങ്ങളാക്കി മാറ്റുന്നു."
    ),
    "Hindi": (
        "कृत्रिम बुद्धिमत्ता हमारे सॉफ्टवेयर बनाने, संस्कृतियों के बीच संवाद "
        "करने और जटिल वैश्विक चुनौतियों को हल करने के तरीके को बदल रही है। "
        "बड़े भाषा मॉडल टोकनाइज़ेशन के माध्यम से मानव पाठ को प्रोसेस करते हैं, "
        "शब्दों और उप-शब्दों को न्यूरल नेटवर्क के लिए संख्यात्मक "
        "प्रतिनिधित्व में बदलते हैं।"
    ),
    "Spanish": (
        "La inteligencia artificial está transformando la forma en que "
        "construimos software, nos comunicamos entre culturas y resolvemos "
        "complejos desafíos globales. Los modelos de lenguaje grandes procesan "
        "el texto humano a través de la tokenización, convirtiendo palabras "
        "y subpalabras en representaciones numéricas para redes neuronales."
    ),
    "French": (
        "L'intelligence artificielle transforme la façon dont nous concevons "
        "les logiciels, communiquons à travers les cultures et résolvons des "
        "défis mondiaux complexes. Les grands modèles de langage traitent le "
        "texte humain grâce à la tokenisation, convertissant les mots et "
        "sous-mots en représentations numériques pour les réseaux de neurones."
    ),
    "Japanese": (
        "人工知能は、私たちがソフトウェアを構築し、異文化間で"
        "コミュニケーションを取り、複雑な地球規模の課題を"
        "解決する方法を変革しています。大規模言語モデルは"
        "トークン化を通じて人間のテキストを処理し、ニューラル"
        "ネットワーク用に単語やサブワードを数値表現に変換します。"
    ),
    "Arabic": (
        "يعمل الذكاء الاصطناعي على تحويل الطريقة التي نبني بها البرمجيات، "
        "ونتواصل عبر الثقافات، ونحل التحديات العالمية المعقدة. "
        "تعالج النماذج اللغوية الكبيرة النصوص البشرية من خلال الترميز، "
        "حيث تحول الكلمات والكلمات الفرعية إلى تمثيلات رقمية للشبكات العصبية."
    ),
}


def print_comparison_table(encoding_name: str = "cl100k_base") -> None:
    """Run tokenization comparison and print formatted results table."""
    stats = compare_tokenization(SAMPLE_PARAGRAPHS, encoding_name=encoding_name)
    costs = calculate_cost_impact(stats, request_volume=1_000_000)

    print("=" * 105)
    print(f" MULTILINGUAL TOKENIZATION BENCHMARK (Tokenizer: {encoding_name})")
    print("=" * 105)
    header = (
        f"{'Language':<12} | {'Tokens':<8} | {'Chars':<7} | {'Bytes':<7} | "
        f"{'Chars/Tok':<10} | {'Bytes/Tok':<10} | {'Fertility':<10} | "
        f"{'1M Cost ($)':<12}"
    )
    print(header)
    print("-" * 105)

    for lang, stat in stats.items():
        cost_info = costs[lang]
        fertility_str = f"{stat.fertility_ratio:.2f}x"
        cost_str = f"${cost_info['monthly_total_cost_usd']:,.2f}"
        row = (
            f"{lang:<12} | {stat.token_count:<8} | {stat.char_count:<7} | "
            f"{stat.byte_count:<7} | {stat.chars_per_token:<10.2f} | "
            f"{stat.bytes_per_token:<10.2f} | {fertility_str:<10} | {cost_str:<12}"
        )
        print(row)

    print("=" * 105)
    print("\n--- Subword Token Fragmentation Inspection ---")
    for lang in ["English", "Malayalam", "Hindi", "Spanish"]:
        stat = stats[lang]
        preview = " | ".join(f"[{p}]" for p in stat.token_pieces[:12])
        print(f"\n[{lang}] ({stat.token_count} total tokens):")
        print(f"First tokens preview: {preview} ...")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Multilingual Tokenization & Cost Disparity Analysis"
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        choices=["cl100k_base", "o200k_base", "p50k_base", "r50k_base"],
        help="tiktoken encoding to test (default: cl100k_base used in GPT-4)",
    )
    parser.add_argument(
        "--compare-all-encodings",
        action="store_true",
        help="Compare cl100k_base (GPT-4) vs o200k_base (GPT-4o)",
    )

    args = parser.parse_args()

    if args.compare_all_encodings:
        for enc in ["cl100k_base", "o200k_base"]:
            print_comparison_table(encoding_name=enc)
            print("\n")
    else:
        print_comparison_table(encoding_name=args.encoding)


if __name__ == "__main__":
    main()
