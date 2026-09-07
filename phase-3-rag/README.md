# Phase 3: Retrieval-Augmented Generation (RAG) Systems

Welcome to **Phase 3** of the AI Engineering Onboarding!

In this phase, we progress from raw model generation to grounded knowledge retrieval. We explore how external unstructured and structured knowledge is ingested, chunked, embedded, indexed, and retrieved to eliminate hallucinations and give LLMs private, up-to-date domain memory.

---

## Task 3.1: Naive RAG Baseline (Control)

### Objective
Build the baseline Naive RAG pipeline:
1. **Corpus**: 20 technical operations documents (*Project Odyssey Mars Base*).
2. **Chunking**: Fixed 500-token chunks using `tiktoken` (`cl100k_base`).
3. **Vector Search**: Embed chunks with Google's `gemini-embedding-001` and retrieve the top-5 nearest neighbors via cosine similarity.
4. **Prompt Stuffing**: Inject the top-5 retrieved chunks into the prompt context and synthesize a grounded answer using `gemini-3.5-flash-lite`.
5. **Control Benchmark**: This implementation serves as the permanent control baseline against which all advanced RAG methods in Phase 3 (hierarchical chunking, reranking, hybrid search, semantic routing, and query rewriting) will be compared.

---

### What Happens Under the Hood? (Frontend / JS Analogy)

In frontend development, think of Naive RAG as a **Semantic Search & Filter Pipeline**:

1. **Ingestion & Indexing (Client Cache / Search Index)**:
   - Break large text files into 500-token pages.
   - Run each page through an embedding function to turn text into a 3,072-dimensional vector.
2. **Retrieval (`Array.filter().sort()`)**:
   - Convert user query into an embedding vector.
   - Compute the angle (cosine similarity) between query vector and all chunk vectors.
   - Take the top 5 closest chunks.
3. **Context Stuffing & Generation**:
   - Concatenate the 5 chunks into a string.
   - Send `prompt + context` to the LLM.

---

### Running & Verifying

#### 1. Ask a Technical Question
```bash
uv run naive-rag --query "What is the emergency coolant pressure threshold for the secondary reactor?"
```

#### 2. Detailed Chunk Inspection (Verbose Mode)
```bash
uv run naive-rag --query "When was Protocol Omega established and what triggers it?" --verbose
```

#### 3. Run Test Suite
```bash
uv run pytest -v
```
