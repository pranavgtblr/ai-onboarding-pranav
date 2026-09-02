# Phase 1: LLM Fundamentals

## Task 1.1: Multilingual Tokenization & Cost Disparity Analysis

This module analyzes how Large Language Model tokenizers (Byte Pair Encoding - BPE) process different human languages and scripts, measuring token counts, fertility ratios, and their operational and financial impact on multilingual AI applications.

---

### Experimental Setup & Ratios

#### Tested Paragraph
> *"Artificial intelligence is transforming how we build software, communicate across cultures, and solve complex global challenges. Large language models process human text through tokenization, converting words and subwords into numerical representations for neural networks."*

#### Benchmark Comparison

| Tokenizer | Language | Script Family | Token Count | Chars / Token | Fertility Ratio | 1M Reqs Cost Proj. (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **cl100k_base** (GPT-4) | **English** | Latin (Native) | **42** | 6.50 | **1.00x** (Baseline) | **$525.00** |
| | **Spanish** | Latin (Extended) | **73** | 4.53 | **1.74x** | **$912.50** |
| | **French** | Latin (Extended) | **82** | 4.11 | **1.95x** | **$1,025.00** |
| | **Japanese** | Kanji / Kana | **147** | 0.87 | **3.50x** | **$1,837.50** |
| | **Arabic** | Arabic | **162** | 1.54 | **3.86x** | **$2,025.00** |
| | **Hindi** | Devanagari | **289** | 0.99 | **6.88x** | **$3,612.50** |
| | **Malayalam** | Indic (Dravidian) | **495** | 0.60 | **11.79x** | **$6,187.50** |
| **o200k_base** (GPT-4o) | **English** | Latin (Native) | **41** | 6.66 | **1.00x** (Baseline) | **$512.50** |
| | **Spanish** | Latin (Extended) | **62** | 5.34 | **1.51x** | **$775.00** |
| | **French** | Latin (Extended) | **68** | 4.96 | **1.66x** | **$850.00** |
| | **Arabic** | Arabic | **74** | 3.37 | **1.80x** | **$925.00** |
| | **Hindi** | Devanagari | **88** | 3.26 | **2.15x** | **$1,100.00** |
| | **Japanese** | Kanji / Kana | **102** | 1.25 | **2.49x** | **$1,275.00** |
| | **Malayalam** | Indic (Dravidian) | **107** | 2.78 | **2.61x** | **$1,337.50** |

---

## Task 1.2: Temperature Sampling & Output Variance Experiment

This experiment evaluates the effect of the temperature hyperparameter across 15 repeated generations (5 runs per temperature: 0.0, 0.7, 1.2) using the Gemini API on an identical creative prompt:

> **Prompt:** *"In exactly 3 sentences, describe what happens when a deep-sea submarine discovers an uncharted ancient underwater civilization."*

Raw outputs and metrics are persisted in [`outputs/temperature_experiment_results.json`](outputs/temperature_experiment_results.json) and [`outputs/temperature_experiment_results.md`](outputs/temperature_experiment_results.md).

---

### Empirical Results Summary

| Temperature (T) | Total Runs | Unique Outputs | Avg Word Count | Avg Pairwise Jaccard Sim | Output Behavior Profile |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0** | 5 | 5 / 5 | 64.6 words | **0.1900** | Highly structured, predictable phrasing, repetitive motifs |
| **0.7** | 5 | 5 / 5 | 63.0 words | **0.1730** | Balanced, fluent, moderate narrative variation |
| **1.2** | 5 | 5 / 5 | 69.4 words | **0.1410** | High lexical diversity, unexpected adjectives, novel plot twists |

---

### What Changes vs. What Does Not

#### 1. What Changes (Variance & Diversity)
- **Lexical Diversity & Vocabulary Breadth:**
  - At T = 0.0: Descriptions rely on standard high-probability tokens: *"towering bioluminescent spires"*, *"robotic arms"*, *"ancient city"*.
  - At T = 1.2: Descriptions introduce rich, low-probability adjectives and specific world-building vocabulary: *"barnacle-encrusted basalt structures"*, *"descent sphere"*, *"iridescent stone"*, *"cyclopean ruins"*, *"tractor beam"*.
- **Narrative Trajectory & Plot Twists:**
  - At T = 0.0: The story almost always follows the same archetype: lights shine -> robotic arms scrape silt -> city gates grind open peacefully.
  - At T = 0.7: Introduces varied events: defense runes awaken, seismic shifts, glowing aqueducts.
  - At T = 1.2: Introduces divergent plot conflicts: holographic star charts, communications severed, terrifying alien encounters, and active defensive barriers.
- **Sentence Structure & Cadence:**
  - T = 0.0 uses standard Subject-Verb-Object participial clauses (*"As the submarine's floodlights sweep..."*).
  - T = 1.2 introduces varied syntactical inversions (*"Floodlights from the descent sphere sweep across..."*).

#### 2. What Does NOT Change (Invariants)
- **Constraint Compliance:** All runs across all temperatures faithfully adhered to the prompt's 3-sentence constraint.
- **Core Semantic Anchors:** Every response preserved the primary entities: submarine/submersible, ocean depth/abyss, illumination/floodlights, and ancient architectural ruins.
- **Syntactic Coherence & Grammar:** Even at T = 1.2, modern models maintain valid grammatical and semantic coherence without degrading into gibberish.

---

## Task 1.3: Text Embeddings & 10x10 Cosine Similarity Matrix

This experiment embeds 10 curated sentences into 3,072-dimensional dense vector representations using Google's `gemini-embedding-001` model, calculates the complete 10x10 Cosine Similarity Matrix, and analyzes semantic equivalence vs. lexical overlap anomalies.

---

### The 10 Curated Sentences

1. **S1 (Pair A1 - Infant Sleep):** *"The infant is asleep."*
2. **S2 (Pair A2 - Infant Sleep):** *"A newborn baby rests quietly."*
3. **S3 (Pair B1 - Customer Support):** *"Initiate a refund for defective widget SKU-98421 returned by the customer."*
4. **S4 (Pair B2 - Mechanical Engineering):** *"Review the tensile strength specifications and steel alloy blueprint for component SKU-98421."*
5. **S5 (Food 1 - Cooking):** *"Whisk the egg yolks with fresh cream and sugar until smooth."*
6. **S6 (Food 2 - Cooking):** *"Beat the milk, butter, and vanilla extract in a large glass bowl."*
7. **S7 (Code 1 - Software):** *"Write a unit test for the REST API endpoint using pytest."*
8. **S8 (Code 2 - Software):** *"Debug the async route handler in FastAPI to prevent memory leaks."*
9. **S9 (Astronomy):** *"Astronomers detected a supermassive black hole at the center of the distant galaxy."*
10. **S10 (Finance):** *"The central bank raised benchmark interest rates to combat rising inflation."*

---

### Full 10x10 Cosine Similarity Matrix

| Sentence | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 (Pair A1)** | **1.0000** | **0.8188** | 0.5308 | 0.5090 | 0.5453 | 0.5519 | 0.5093 | 0.5014 | 0.5357 | 0.5568 |
| **S2 (Pair A2)** | **0.8188** | **1.0000** | 0.5190 | 0.5077 | 0.5449 | 0.5235 | 0.5034 | 0.4792 | 0.5305 | 0.5215 |
| **S3 (Pair B1)** | 0.5308 | 0.5190 | **1.0000** | **0.5947** | 0.5205 | 0.5235 | 0.5400 | 0.4967 | 0.4803 | 0.4992 |
| **S4 (Pair B2)** | 0.5090 | 0.5077 | **0.5947** | **1.0000** | 0.5083 | 0.5255 | 0.5160 | 0.4883 | 0.5102 | 0.5136 |
| **S5 (Food 1)** | 0.5453 | 0.5449 | 0.5205 | 0.5083 | **1.0000** | **0.7042** | 0.4886 | 0.4481 | 0.5141 | 0.5536 |
| **S6 (Food 2)** | 0.5519 | 0.5235 | 0.5235 | 0.5255 | **0.7042** | **1.0000** | 0.4987 | 0.4480 | 0.4893 | 0.5486 |
| **S7 (Code 1)** | 0.5093 | 0.5034 | 0.5400 | 0.5160 | 0.4886 | 0.4987 | **1.0000** | **0.6029** | 0.4573 | 0.4900 |
| **S8 (Code 2)** | 0.5014 | 0.4792 | 0.4967 | 0.4883 | 0.4481 | 0.4480 | **0.6029** | **1.0000** | 0.4728 | 0.4697 |
| **S9 (Astronomy)** | 0.5357 | 0.5305 | 0.4803 | 0.5102 | 0.5141 | 0.4893 | 0.4573 | 0.4728 | **1.0000** | 0.5507 |
| **S10 (Finance)** | 0.5568 | 0.5215 | 0.4992 | 0.5136 | 0.5536 | 0.5486 | 0.4900 | 0.4697 | 0.5507 | **1.0000** |

---

### In-Depth Explanation of Key Results

#### 1. Pair A: High Semantic Similarity With Zero Shared Words (S1 vs S2)
- **S1:** *"The infant is asleep."*
- **S2:** *"A newborn baby rests quietly."*
- **Shared Words:** `0` (Zero lexical overlap)
- **Cosine Similarity:** **`0.8188`** (The highest similarity in the entire matrix between distinct sentences)
- **Why this happens:**
  - Keyword search engines (like SQL `LIKE '%infant%'` or Elasticsearch BM25) give this pair a score of **0.0** because there are zero matching tokens.
  - Embedding models project sentences into a 3,072-dimensional space where words like *infant/baby*, *asleep/rests*, and *quietly* occupy neighboring coordinates.
  - The model maps the holistic conceptual meaning of the sentence rather than surface string tokens.

#### 2. Pair B: Shared Exact Product Code with Divergent Meaning (S3 vs S4)
- **S3 (Customer Support):** *"Initiate a refund for defective widget SKU-98421 returned by the customer."*
- **S4 (Mechanical Engineering):** *"Review the tensile strength specifications and steel alloy blueprint for component SKU-98421."*
- **Shared Identifiers:** `SKU-98421`
- **Cosine Similarity:** **`0.5947`**
- **Why this happens:**
  - The shared token `SKU-98421` is rare and alphanumeric, so it carries substantial attention weight and nudges the vectors closer together (giving 0.5947, which is higher than background baseline ~0.48).
  - However, the contextual embeddings of the surrounding words (*refund/defective/customer* vs *tensile strength/steel alloy/blueprint*) anchor the sentences into completely different semantic neighborhoods.
  - As a result, the similarity is significantly lower than Pair A (0.8188), proving that modern dense embeddings prioritize surrounding context and intent over isolated keyword collisions.

#### 3. Why This Matters for Production RAG (Retrieval-Augmented Generation)
- **Dense Vector Search Strength:** Captures user intent when users phrase queries with different vocabulary (e.g. searching *"baby sleeping"* retrieves *"infant asleep"*).
- **Dense Vector Search Weakness:** May struggle with exact product lookups, part numbers, or error codes where exact alphanumeric matching is critical.
- **Production Solution (Hybrid Search):** Combine **Dense Vector Search (Cosine Similarity)** for semantic intent with **Sparse BM25 Keyword Search** for exact identifiers (SKUs, IDs, log codes) using Reciprocal Rank Fusion (RRF).

---

## Running the CLI Tools & Tests

### 1. Run Tokenizer Benchmark (Task 1.1)
```bash
uv run tokenize-demo --compare-all-encodings
```

### 2. Run Temperature Sampling Experiment (Task 1.2)
```bash
uv run temperature-demo
```

### 3. Run Embeddings Similarity Matrix (Task 1.3)
```bash
uv run embed-demo
```

### 4. Run Automated Test Suite
```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run pyright
```
