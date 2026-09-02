# Phase 1: LLM Fundamentals

## Task 1.1: Multilingual Tokenization & Cost Disparity Analysis

This module analyzes how Large Language Model tokenizers (Byte Pair Encoding - BPE) process different human languages and scripts, measuring token counts, fertility ratios, and their operational and financial impact on multilingual AI applications.

---

### Experimental Setup & Ratios

#### Tested Paragraph
> *"Artificial intelligence is transforming how we build software, communicate across cultures, and solve complex global challenges. Large language models process human text through tokenization, converting words and subwords into numerical representations for neural networks."*

#### Benchmark Comparison

| Tokenizer | Language | Script Family | Token Count | Chars / Token | Fertility Ratio | 1M Reqs Cost Proj. ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`cl100k_base`** (GPT-4) | **English** | Latin (Native) | **42** | 6.50 | **1.00x** (Baseline) | **$525.00** |
| | **Spanish** | Latin (Extended) | **73** | 4.53 | **1.74x** | **$912.50** |
| | **French** | Latin (Extended) | **82** | 4.11 | **1.95x** | **$1,025.00** |
| | **Japanese** | Kanji / Kana | **147** | 0.87 | **3.50x** | **$1,837.50** |
| | **Arabic** | Arabic | **162** | 1.54 | **3.86x** | **$2,025.00** |
| | **Hindi** | Devanagari | **289** | 0.99 | **6.88x** | **$3,612.50** |
| | **Malayalam** | Indic (Dravidian) | **495** | 0.60 | **11.79x** | **$6,187.50** |
| **`o200k_base`** (GPT-4o) | **English** | Latin (Native) | **41** | 6.66 | **1.00x** (Baseline) | **$512.50** |
| | **Spanish** | Latin (Extended) | **62** | 5.34 | **1.51x** | **$775.00** |
| | **French** | Latin (Extended) | **68** | 4.96 | **1.66x** | **$850.00** |
| | **Arabic** | Arabic | **74** | 3.37 | **1.80x** | **$925.00** |
| | **Hindi** | Devanagari | **88** | 3.26 | **2.15x** | **$1,100.00** |
| | **Japanese** | Kanji / Kana | **102** | 1.25 | **2.49x** | **$1,275.00** |
| | **Malayalam** | Indic (Dravidian) | **107** | 2.78 | **2.61x** | **$1,337.50** |

---

## Task 1.2: Temperature Sampling & Output Variance Experiment

This experiment evaluates the effect of the **temperature** hyperparameter ($T = 0.0, 0.7, 1.2$) across 15 repeated generations (5 runs per temperature) using the Gemini API on an identical creative prompt:

> **Prompt:** *"In exactly 3 sentences, describe what happens when a deep-sea submarine discovers an uncharted ancient underwater civilization."*

Raw outputs and metrics are persisted in [`outputs/temperature_experiment_results.json`](outputs/temperature_experiment_results.json) and [`outputs/temperature_experiment_results.md`](outputs/temperature_experiment_results.md).

---

### Empirical Results Summary

| Temperature ($T$) | Total Runs | Unique Outputs | Avg Word Count | Avg Pairwise Jaccard Sim | Output Behavior Profile |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`0.0`** | 5 | 5 / 5 | 64.6 words | **0.1900** | Highly structured, predictable phrasing, repetitive motifs |
| **`0.7`** | 5 | 5 / 5 | 63.0 words | **0.1730** | Balanced, fluent, moderate narrative variation |
| **`1.2`** | 5 | 5 / 5 | 69.4 words | **0.1410** | High lexical diversity, unexpected adjectives, novel plot twists |

---

### What Changes vs. What Does Not

#### 1. What Changes (Variance & Diversity)
- **Lexical Diversity & Vocabulary Breadth:**
  - At $T = 0.0$: Descriptions rely on standard high-probability tokens: *"towering bioluminescent spires"*, *"robotic arms"*, *"ancient city"*.
  - At $T = 1.2$: Descriptions introduce rich, low-probability adjectives and specific world-building vocabulary: *"barnacle-encrusted basalt structures"*, *"descent sphere"*, *"iridescent stone"*, *"cyclopean ruins"*, *"tractor beam"*.
- **Narrative Trajectory & Plot Twists:**
  - At $T = 0.0$: The story almost always follows the same archetype: lights shine $\rightarrow$ robotic arms scrape silt $\rightarrow$ city gates grind open peacefully.
  - At $T = 0.7$: Introduces varied events: defense runes awaken, seismic shifts, glowing aqueducts.
  - At $T = 1.2$: Introduces divergent plot conflicts: holographic star charts, communications severed, terrifying alien encounters, and active defensive barriers.
- **Sentence Structure & Cadence:**
  - $T=0.0$ uses standard Subject-Verb-Object participial clauses (*"As the submarine's floodlights sweep..."*).
  - $T=1.2$ introduces varied syntactical inversions (*"Floodlights from the descent sphere sweep across..."*).

#### 2. What Does NOT Change (Invariants)
- **Constraint Compliance:** All runs across all temperatures faithfully adhered to the prompt’s 3-sentence constraint.
- **Core Semantic Anchors:** Every response preserved the primary entities: submarine/submersible, ocean depth/abyss, illumination/floodlights, and ancient architectural ruins.
- **Syntactic Coherence & Grammar:** Even at $T = 1.2$, modern models maintain valid grammatical and semantic coherence without degrading into gibberish.

---

### Deep-Dive: The Mathematics of Temperature

LLMs generate text by predicting a raw score (logit $z_i$) for every token $i$ in their vocabulary $V$. Temperature $T$ scales these logits before passing them into the **Softmax** function:

$$P(w_i) = \frac{e^{z_i / T}}{\sum_{j \in V} e^{z_j / T}}$$

```text
Logits [z1, z2, z3] ──> [ Divide by T ] ──> [ Softmax ] ──> Probability Distribution ──> Sampling
```

1. **At $T \to 0$ (Greedy / Argmax Decoding):**
   - The highest logit is exaggerated towards $P(w_{\text{max}}) \approx 1.0$ while all other probabilities collapse towards $0$.
   - The model selects the most mathematically probable token at each step.
   - *Why is modern $T=0$ not 100% bitwise identical across API calls?* Parallel GPU execution non-determinism (floating-point summation non-associativity across distributed tensor cores and Mixture-of-Experts routing).
2. **At $T = 0.7$ (Default / Balanced Sampling):**
   - Retains the model's natural probability distribution with slight sharpening.
   - High-probability words are favored, but natural linguistic variation is preserved.
3. **At $T \ge 1.0$ (High Entropy / Creative Sampling):**
   - Divides logits by a number $> 1.0$, flattening the probability distribution curve.
   - Tail tokens (rarer synonyms, unusual concepts) become significantly more likely to be sampled.

---

### Engineering Decision Matrix: When to Use Which Temperature

| Task Type | Recommended Temperature | Rationale |
| :--- | :---: | :--- |
| **Structured Data / JSON Extraction** | **`0.0`** | Requires strict schema adherence; minimizes hallucinated keys or syntax errors. |
| **Code Generation & SQL Queries** | **`0.0 - 0.2`** | Demands exact syntax, deterministic logic, and deterministic test reproducibility. |
| **Classification & Routing** | **`0.0`** | Deterministic categorical decisions without creative drift. |
| **RAG Q&A & Document Summarization** | **`0.2 - 0.5`** | Factual grounding on retrieved context while maintaining natural prose. |
| **Conversational Chatbots / Copilots** | **`0.7`** | Natural conversational cadence; balances helpfulness with conversational warmth. |
| **Creative Writing & Brainstorming** | **`0.9 - 1.2`** | High novelty, unusual metaphors, out-of-the-box conceptual combinations. |
| **Adversarial / Red-Teaming Tests** | **`1.2 - 1.5`** | Uncovers model edge cases, hallucination modes, and boundary behaviors. |

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

### 3. Run Quality Checks & Pytest Suite
```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run pyright
```
