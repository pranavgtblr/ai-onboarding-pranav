# LLM Fundamentals: Core Architectural Questions & Answers

## 1. Why does the same prompt cost more on message 20 than on message 1?

LLM APIs (such as OpenAI and Google Gemini) are completely **stateless HTTP services**. The model retains no memory or session state between API calls.

To create the experience of an ongoing conversation, the client application must append the entire prior conversation history (messages 1 through 19, including user questions, assistant responses, and system prompts) into the payload sent with message 20. 

Because providers charge for every input token processed on every request:
- **Message 1:** You pay for the system prompt + user message 1.
- **Message 20:** You pay for the system prompt + all 19 previous exchanges + user message 20.

Token consumption and input costs scale quadratically over the lifetime of a chat session.

---

## 2. Why can't the model tell you what it doesn't know?

An LLM is an autoregressive next-token prediction engine, not a verified knowledge graph or database:

1. **Absence of an Internal Knowledge Boundary:** The model does not maintain an index of "known" versus "unknown" facts. Its knowledge exists solely as diffuse numerical weights across billions of neural connections. It cannot run a query and receive a `404 Not Found`.
2. **Plausibility Over Truth:** The model optimizes for token probability (what words most naturally follow the prompt), not epistemic validity. A fabricated hallucination and a verified fact often share the exact same high probability and fluent sentence structure.
3. **Training & Alignment Bias:** Human training data and RLHF (Reinforcement Learning from Human Feedback) reward helpful, direct answers. Without specific prompt constraints or tool-assisted verification, the model defaults to completing the text pattern rather than self-auditing its training boundaries.

---

## 3. A user asks "what's the SKU for the Model 7 clasp?" - why might pure vector search miss it?

Pure vector search (dense embeddings) maps text into a continuous semantic coordinate space based on holistic conceptual meaning, which introduces two failure modes for this query:

1. **Semantic Closeness of Disparate Product Versions:** In dense vector space, "Model 7 clasp", "Model 8 clasp", and "Model 6 clasp" map to virtually identical coordinates (cosine similarity often exceeding 0.95). Dense embeddings capture the broad concept ("hardware fastener / clasp"), but compress away minor alphanumeric distinctions like single digits.
2. **Lack of Exact Alphanumeric Matching:** Vector search is designed to recognize synonyms (e.g., matching "infant" to "baby"), not to perform exact string or substring equality. Rare tokens like part numbers and model identifiers do not carry unique semantic neighborhoods.

**Production Solution:** Real-world RAG systems use **Hybrid Search** (combining sparse BM25 keyword matching for exact identifiers like "Model 7" with dense vector search for conceptual meaning like "clasp").

---

## 4. Step by step, what happens between "the model decides to use a tool" and "the user sees an answer"?

When an LLM uses function calling, the model never executes external code directly. The process is an orchestrator-mediated loop:

1. **Tool Call Emission:** During autoregressive generation, the model determines that an external tool is required based on tool definitions provided in the prompt. It outputs a structured JSON schema specifying the function name and arguments (e.g., `{"name": "lookup_sku", "arguments": {"model": "Model 7", "item": "clasp"}}`) with a termination stop reason (`tool_calls`).
2. **Control Yield to Application Runtime:** The model halts generation. The API response returns the structured tool call to the client/application code.
3. **Local Tool Execution:** The host application runtime validates the arguments (e.g., via Pydantic or Zod) and executes the actual code (such as querying a SQL database, invoking a REST API, or running a calculator).
4. **Tool Result Injection:** The host application formats the execution return value into a new message with the role `"tool"` (or `"function"`) and appends it to the active conversation history.
5. **Second Model Invocation:** The application makes a second API request to the LLM, passing the complete conversation history including the new tool result.
6. **Final Synthesis:** The model reads the tool output as context, interprets the data, and generates a natural language response for the user (or decides to trigger an additional tool if further data is needed).
