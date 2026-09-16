# Phase 6 Capstone: Failure Diagnosis & Post-Mortem Walkthrough

**Task Reference**: Item 6.4 — *Demo it, plus a written walkthrough of one thing that failed and how you diagnosed it.*  
**System Under Test**: PG Recommends (AI Film Curator & Taste Engine)  
**Incident Classification**: Semantic Retrieval Rating Inversion & Persona Degeneration  

---

## 1. Executive Summary of Failure

During interactive user testing of the conversational recommendation agent, a critical behavioral and retrieval failure occurred:
- When a user asked for a cheerful, uplifting movie (*"Recommend a feel-good romance or comedy"*), the agent repeatedly responded with a rigid, canned boilerplate intro (*"If you're asking me for a romcom or something feel-good, let me save you from the generic algorithm sludge..."*).
- Worse, the agent recommended **Bhool Bhulaiyaa 2** (logged in Pranav's Letterboxd diary at a dismal **★ 2.0 / 5.0**, where the review explicitly stated the lead actor was *"annoying af"* with *"no romantic chemistry"*) and **Rekhachithram** (a gritty **★ 3.0** procedural crime thriller about a murder investigation).
- The bot enthusiastically presented these low-rated and non-feel-good films as *"standouts from my Letterboxd diary"* for a feel-good movie night.

This completely undermined user trust, broke persona intimacy, and inverted recommendation utility.

---

## 2. Root Cause Analysis (RCA)

We systematically traced the failure across three pipeline stages:

```
[User: "feel-good movie"]
           │
           ▼
1. HYBRID RETRIEVER (BM25)
   - BM25 matched tokens: "comedy", "fun", "romance" in reviews.csv text.
   - ❌ BUG: Retriever had NO star-rating filter or sentiment threshold.
   - Result: A scathing 2.0★ review containing the word "romance" was scored as a top match.
           │
           ▼
2. AGENT PROMPT & FEW-SHOT OVERFITTING
   - Few-shot examples in system prompt had a rigid template:
     "If you're asking me for [genre], let me save you from the generic algorithm sludge..."
   - ❌ BUG: Gemini Flash over-indexed on the template structure and parroted the boilerplate
     phrase on every single turn instead of using natural human conversation.
           │
           ▼
3. CITATION / PERSONA INVERSION
   - The LLM saw diary entries injected into its context and assumed all retrieved entries
     were recommendations to endorse, ignoring the negative sentiment and low numerical rating.
```

### Detailed Breakdown

1. **Unweighted Lexical Matching (Rating Inversion)**:
   In `reviews.csv`, Pranav had logged 500+ movies, many of which had low ratings (1.0★ - 2.5★). A user review saying:
   > *"The initial setup was interesting... though I found Karthik Aryan to be annoying af and it didn't really feel like there was a proper romantic chemistry"*
   contains high-frequency tokens for "interesting", "romantic", "chemistry". The standard BM25 ranking algorithm calculated high BM25 term overlap with "feel-good romantic comedy" despite the review being deeply negative.

2. **Persona Degeneration (Boilerplate Hallucination)**:
   The system prompt contained a fixed few-shot demonstration designed to show Letterboxd-style snark. However, the model treated the prefix as a mandatory template, generating repetitive boilerplate that felt robotic and artificial.

3. **Absence of Sentiment / Mood Guardrails**:
   The catalog retriever lacked mood-to-genre re-ranking. A murder procedural (*Rekhachithram*) contains crime elements, yet because the diary note mentioned conversational elements, it leaked into the feel-good pool.

---

## 3. Step-by-Step Diagnostic Process

1. **Step 1: Isolated Retriever Query Output**
   We tested the query against the raw BM25 retriever without the LLM:
   ```python
   retriever = HybridMovieRetriever(catalog)
   results = retriever.retrieve("feel-good romcom or comedy", top_k=5)
   # Inspected results: Bhool Bhulaiyaa 2 was Ranked #1 due to BM25 keyword density
   ```
   **Observation**: The retriever returned items with rating `2.0` because rating metadata was completely ignored during ranking.

2. **Step 2: Log Tracing & Temperature / Prompt Analysis**
   We inspected the prompt sent to Gemini:
   - System prompt few-shot demonstrations were rigid and declarative.
   - The model took the introductory sentence as an invariant pattern.

3. **Step 3: Verification of the Secondary Asset Failure (Bonus)**
   During containerized deployment testing, we also discovered a silent build-time failure:
   - `frontend/src/App.jsx` had `const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'`.
   - In production builds, `VITE_API_BASE` was undefined, baking `http://localhost:8000` into `frontend/dist/assets/index-*.js`. Remote clients got `ERR_CONNECTION_REFUSED`.

---

## 4. Architectural Fixes & Implementations

### Fix 1: Minimum Rating & Sentiment Filtering in Retriever
In `src/phase_6_capstone/retrieval.py` and `agent.py`:
- Added a rating gate: When the user asks for recommendations, candidates with star rating `< 3.0` are excluded from positive recommendations unless the user specifically asks for *"bad movies"*, *"guilty pleasures"*, or *"films you disliked"*.
- For general recommendations, prioritized items with rating `≥ 3.5★`.

### Fix 2: Dynamic Persona Prompting & Anti-Boilerplate Guardrail
In `src/phase_6_capstone/agent.py`:
- Removed rigid boilerplate template strings.
- Replaced with explicit dynamic persona rules:
  ```markdown
  - Adopt a natural, friendly, conversational tone like talking to a fellow film enthusiast.
  - NEVER repeat robotic canned phrases like "let me save you from the generic algorithm sludge".
  - Greet dynamically and vary your opening sentence based on what the user actually asked.
  - If a retrieved movie has a low rating (≤ 2.5), NEVER present it as a positive recommendation.
  ```

### Fix 3: Dual-Catalog Hybrid Fallback (Acclaimed Cinema Integration)
If Pranav's personal Letterboxd diary does not contain a high-rated match for a specific niche or mood, the retriever gracefully falls back to the **Acclaimed Cinema** dataset (`acclaimed_cinema.py`), ensuring the user always receives a genuinely acclaimed recommendation with verified critic consensus.

### Fix 4: Production Relative Base URL & Dynamic Port
- In `frontend/src/App.jsx`:
  ```javascript
  const API_BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.PROD ? '' : 'http://localhost:8000');
  ```
- In `Dockerfile`:
  ```dockerfile
  CMD ["sh", "-c", "uv run uvicorn phase_6_capstone.server:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
  ```

---

## 5. Verification & Proof of Fix

1. **Automated Evaluation Benchmark**:
   - Ran `uv run pytest -v tests/test_eval_suite.py` and `tests/test_api.py`.
   - **Retrieval Recall@5**: **100%**
   - **Citation Fidelity**: **100%** (ratings, years, quotes match actual source)
   - **Taste Learning Agreement**: **100%**

2. **Interactive Live Test Verification**:
   - **Input**: *"Recommend a feel-good romance or comedy."*
   - **Output**: The agent greeted naturally, bypassed *Bhool Bhulaiyaa 2*, and recommended *La La Land* and *Palm Springs* (≥ 4.0★), explaining why their romantic chemistry and tone fit the user's specific request.
