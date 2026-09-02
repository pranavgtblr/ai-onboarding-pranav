# Phase 1: LLM Fundamentals

## Task 1.1: Multilingual Tokenization & Cost Disparity Analysis

This module analyzes how Large Language Model tokenizers (Byte Pair Encoding - BPE) process different human languages and scripts, measuring token counts, fertility ratios, and their operational and financial impact on multilingual AI applications.

---

## Experimental Setup

### Standard Multilingual Corpus
The following semantically equivalent paragraph was translated across 7 languages (English, Malayalam, Hindi, Spanish, French, Japanese, and Arabic):

> **English:**  
> *"Artificial intelligence is transforming how we build software, communicate across cultures, and solve complex global challenges. Large language models process human text through tokenization, converting words and subwords into numerical representations for neural networks."*

---

## Token Counts & Fertility Ratios

### 1. GPT-4 / GPT-3.5 Tokenizer (`cl100k_base` — 100k Vocabulary)

| Language | Script Family | Token Count | Character Count | Byte Count | Chars / Token | Bytes / Token | Fertility Ratio | 1M Reqs Cost Proj. ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | Latin (Native) | **42** | 273 | 273 | 6.50 | 6.50 | **1.00x** (Baseline) | **$525.00** |
| **Spanish** | Latin (Extended) | **73** | 331 | 336 | 4.53 | 4.60 | **1.74x** | **$912.50** |
| **French** | Latin (Extended) | **82** | 337 | 347 | 4.11 | 4.23 | **1.95x** | **$1,025.00** |
| **Japanese** | Kanji / Kana | **147** | 128 | 384 | 0.87 | 2.61 | **3.50x** | **$1,837.50** |
| **Arabic** | Arabic | **162** | 249 | 461 | 1.54 | 2.85 | **3.86x** | **$2,025.00** |
| **Hindi** | Devanagari | **289** | 287 | 759 | 0.99 | 2.63 | **6.88x** | **$3,612.50** |
| **Malayalam** | Indic (Dravidian) | **495** | 297 | 833 | 0.60 | 1.68 | **11.79x** | **$6,187.50** |

---

### 2. GPT-4o Tokenizer (`o200k_base` — 200k Vocabulary)

| Language | Script Family | Token Count | Character Count | Byte Count | Chars / Token | Bytes / Token | Fertility Ratio | 1M Reqs Cost Proj. ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | Latin (Native) | **41** | 273 | 273 | 6.66 | 6.66 | **1.00x** (Baseline) | **$512.50** |
| **Spanish** | Latin (Extended) | **62** | 331 | 336 | 5.34 | 5.42 | **1.51x** | **$775.00** |
| **French** | Latin (Extended) | **68** | 337 | 347 | 4.96 | 5.10 | **1.66x** | **$850.00** |
| **Arabic** | Arabic | **74** | 249 | 461 | 3.37 | 6.23 | **1.80x** | **$925.00** |
| **Hindi** | Devanagari | **88** | 287 | 759 | 3.26 | 8.62 | **2.15x** | **$1,100.00** |
| **Japanese** | Kanji / Kana | **102** | 128 | 384 | 1.25 | 3.77 | **2.49x** | **$1,275.00** |
| **Malayalam** | Indic (Dravidian) | **107** | 297 | 833 | 2.78 | 7.79 | **2.61x** | **$1,337.50** |

---

## Key Observations & Token Fragmentation

### Why Does Fragmentation Occur?
1. **Vocabulary Imbalance during Pre-training:** BPE merge tables are constructed by maximizing compression over training corpora dominated by English text.
2. **Whole Words vs. Subword Fallback:**
   - English words like `"transforming"`, `"intelligence"`, `"software"` exist as single tokens in the vocabulary.
   - Non-Latin scripts (Devanagari, Malayalam, Arabic) lack corresponding subwords in older vocabularies like `cl100k_base`. As a result, characters are split into individual unicode bytes (`\xe0\xb4\x95...`), taking **3 to 6 tokens per single character**.
3. **Subword Inspection (`cl100k_base`):**
   - **English:** `[Art] | [ificial] | [ intelligence] | [ is] | [ transforming] ...` (42 tokens total)
   - **Malayalam:** `[b'\xe0\xb4'] | [b'\x95'] | [b'\xe0\xb5'] | [b'\x83'] | [b'\xe0\xb4'] | [b'\xa4'] ...` (495 tokens total)
4. **Vocabulary Expansion in Modern Tokenizers (`o200k_base`):**
   - Expanding vocabulary to 200,000 tokens compressed Malayalam from **495 tokens (11.79x)** down to **107 tokens (2.61x)** and Hindi from **289 tokens (6.88x)** down to **88 tokens (2.15x)**.

---

## What This Means for Cost & Architecture on a Multilingual Client

When deploying an LLM-powered system for multilingual or non-English users, token fertility creates critical business and technical consequences:

### 1. The "Language Tax" (Direct API Pricing Disparity)
LLM providers bill strictly on **tokens consumed and generated**, not on characters, words, or informational units.
- An English query/response roundtrip costs **$X**.
- The exact same conversation in Spanish costs **1.5x - 1.7x** more.
- The exact same conversation in Hindi costs **2.15x - 6.88x** more.
- The exact same conversation in Malayalam costs **2.61x - 11.79x** more.

**Cost Projection at 1 Million Requests / Month:**  
*(Assumes \$2.50 / 1M input tokens + \$10.00 / 1M output tokens)*
- **English Client:** **\$512.50 - \$525.00**
- **Spanish Client:** **\$775.00 - \$912.50** (+48% to +74%)
- **Hindi Client:** **\$1,100.00 - \$3,612.50** (+114% to +588%)
- **Malayalam Client:** **\$1,337.50 - \$6,187.50** (+161% to +1078%)

A company serving South Asian or non-English customers faces significantly higher operational burn rate per user for the exact same semantic workload.

### 2. Effective Context Window Shrinkage
LLMs have a finite context window (e.g. 8k, 32k, 128k tokens).
- A 128k token window holds $\approx 85,000$ words of English technical documentation.
- The same 128k window holds only $\approx 7,200$ words of Malayalam in `cl100k_base` ($\approx 33,000$ words in `o200k_base`).
- RAG systems (Retrieval-Augmented Generation) exhaust context memory much faster, forcing smaller chunk budgets and reducing retrieval quality.

### 3. Latency & Generation Throughput Degradation
Autoregressive LLM generation generates tokens sequentially ($O(N)$ forward passes).
- At 80 tokens/second generation speed:
  - English completion (42 tokens): $\approx \mathbf{0.52\text{ seconds}}$
  - Hindi completion (289 tokens): $\approx \mathbf{3.61\text{ seconds}}$
  - Malayalam completion (495 tokens): $\approx \mathbf{6.18\text{ seconds}}$
- Multilingual end users experience noticeable latency penalties and slower streaming output.

### 4. Architectural Mitigation Strategies for Production
1. **Adopt Modern Multilingual Tokenizers:** Standardize on modern model families (e.g. GPT-4o `o200k_base`, Gemini 1.5/2.0, or Llama 3 with 128k vocabulary) which dramatically reduce non-English token inflation.
2. **Translate-at-Boundary Pipeline:** For storage, vector search indexing, and complex chain-of-thought agent reasoning, translate non-English user queries to English at the API gateway, perform retrieval & internal reasoning in English, and synthesize the final answer in the user's native language.
3. **Adaptive Chunking in RAG:** Use character- or semantic-length chunking rather than fixed-token chunking to prevent multilingual chunks from containing too few words.

---

## Running the Tokenizer & Tests

### 1. Sync Dependencies
```bash
cd phase-1-llm-fundamentals
uv sync
```

### 2. Run Tokenizer CLI Benchmark
```bash
# Default cl100k_base (GPT-4) benchmark
uv run tokenize-demo

# Compare cl100k_base vs o200k_base (GPT-4o)
uv run tokenize-demo --compare-all-encodings
```

### 3. Run Automated Tests & Linters
```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```
