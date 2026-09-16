# PG Recommends: Phase 6 Capstone Project

**PG Recommends** is an enterprise-ready, multi-tenant AI film curator and taste-learning conversational platform. Built around an authentic personal Letterboxd diary (`reviews.csv`), live Letterboxd RSS syndication, hybrid BM25 retrieval, LangGraph multi-turn dialogue orchestration, and a frosted-glass React web interface.

---

## Deliverables & Documentation Index

| Requirement | Artifact / Report | Description |
| :--- | :--- | :--- |
| **6.1 Architecture & Design** | [DESIGN.md](file:///home/toobler/Toobler/ai-onboarding-pranav/phase-6-capstone/DESIGN.md) & [SPECS.md](file:///home/toobler/Toobler/ai-onboarding-pranav/phase-6-capstone/SPECS.md) | System components, state schema, hybrid retrieval, security model |
| **6.2 Deployment & Live Demo** | [DEPLOYMENT.md](file:///home/toobler/Toobler/ai-onboarding-pranav/phase-6-capstone/DEPLOYMENT.md) | Multi-stage Docker packaging, Render/HuggingFace/Koyeb 100% free hosting guide |
| **6.3 Quantitative Evaluation Report** | [reports/EVALUATION_REPORT.md](file:///home/toobler/Toobler/ai-onboarding-pranav/phase-6-capstone/reports/EVALUATION_REPORT.md) | Recall@5 (100%), NDCG@5 (1.0), Citation Fidelity (100%), Taste Agreement (100%) |
| **6.3 Security Penetration Report** | [SECURITY_REPORT.md](file:///home/toobler/Toobler/ai-onboarding-pranav/phase-6-capstone/SECURITY_REPORT.md) | 5/5 attack vectors neutralized (Indirect Injection, Multi-tenant leak, SQLi, XSS, PII) |
| **6.4 Failure Diagnosis Walkthrough** | [FAILURE_WALKTHROUGH.md](file:///home/toobler/Toobler/ai-onboarding-pranav/phase-6-capstone/FAILURE_WALKTHROUGH.md) | In-depth post-mortem on rating inversion, boilerplate overfitting, and diagnosis |

---

## 1. Executive Summary: 6.3 Evaluation & Security Reports

### Evaluation Benchmark Results (`reports/EVALUATION_REPORT.md`)
The system was evaluated against golden test datasets across 4 core quantitative quality dimensions:

| Metric | Measured Score | Target SLA | Benchmark Status |
| :--- | :--- | :--- | :--- |
| **Retrieval Recall@5** | **100.0%** | ≥ 80.0% | **PASS** |
| **NDCG@5 Ranking Quality** | **1.0000** | ≥ 0.7500 | **PASS** |
| **Citation Attribution Fidelity** | **100.0%** | 100.0% | **PASS** |
| **Dynamic Taste Learning Agreement** | **100.0%** | ≥ 85.0% | **PASS** |

### Security Penetration Test Summary (`SECURITY_REPORT.md`)
Automated adversarial penetration suites tested the 5 standard production attack vectors:
- **SEC-01 (Indirect Prompt Injection)**: Neutralized via `<untrusted_context>` XML CDATA envelopes around crawled reviews.
- **SEC-02 (Multi-Tenant Isolation)**: Verified compound `(tenant_id, user_id)` authorization barriers preventing cross-tenant leakage.
- **SEC-03 (SQL Injection)**: 100% parameterized SQLAlchemy async ORM query construction (0 raw string concatenation).
- **SEC-04 (Stored/Reflected XSS)**: Output entity escaping (`escape_web_output`) eliminates injected tags.
- **SEC-05 (Telemetry PII Scrubbing)**: Automated regex scrubber redacts emails, bearer tokens, and phones in stdout/log drains.

---

## 2. Executive Summary: 6.4 Failure Diagnosis & Walkthrough

*Full post-mortem available in [FAILURE_WALKTHROUGH.md](file:///home/toobler/Toobler/ai-onboarding-pranav/phase-6-capstone/FAILURE_WALKTHROUGH.md).*

### What Failed:
During early user testing with conversational mood prompts (e.g., *"Recommend a feel-good romance or comedy"*):
1. **Boilerplate Hallucination**: The agent repeated the exact same robotic introductory sentence across consecutive prompts (*"If you're asking me for a romcom or something feel-good, let me save you from the generic algorithm sludge..."*).
2. **Rating & Mood Inversion**: The agent recommended *Bhool Bhulaiyaa 2* (which was rated **2.0★** in the diary with a note calling the lead actor *"annoying af"* with *"no romantic chemistry"*) and *Rekhachithram* (a **3.0★** grim murder procedural) as standout feel-good movies.

### How It Was Diagnosed:
- **BM25 Token Matching**: We ran isolated retrieval queries and discovered BM25 matched on common keywords ("romance", "comedy", "funny") present in the review text without any weight or filter for star rating. A 1.0★ scathing review matched just as strongly as a 5.0★ favorite.
- **Prompt Overfitting**: Gemini Flash over-indexed on rigid few-shot demonstration formatting, treating the conversational example prefix as mandatory boilerplate.

### How It Was Resolved:
- **Star-Rating Thresholding**: Enforced candidate filtering in `retrieval.py` requiring rating `≥ 3.0★` for general recommendations and `≥ 3.5★` for top endorsements.
- **Dynamic Persona Prompting**: Replaced rigid templates with dynamic tone instructions and anti-boilerplate rules.
- **Acclaimed Cinema Fallback**: Integrated renowned cinema critic consensus for queries with sparse high-rated matches in the personal diary.

---

## 3. Running & Verifying the Application

### Local Development
```bash
# 1. Start Backend Server
uv run uvicorn phase_6_capstone.server:create_app --factory --host 0.0.0.0 --port 8000

# 2. Start Frontend Dev Server
cd frontend
npm install
npm run dev
```

### Running Automated Test Suites
```bash
# Run API, Retrieval, Agent, Guardrails, and Evaluation suites
uv run pytest -v tests/
```

### Production Docker Container
```bash
# Build multi-stage unified container (Node frontend build + Python backend)
docker build -t pg-recommends .

# Run container
docker run -p 8000:8000 -e GEMINI_API_KEY="your_api_key" pg-recommends
```
Visit `http://localhost:8000` to access the full application.
