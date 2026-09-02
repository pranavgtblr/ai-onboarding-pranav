# 10x10 Sentence Embeddings Cosine Similarity Matrix

- **Embedding Model:** `gemini-embedding-001`
- **Embedding Dimensions:** `3072`

## Sentences Index

- **S1 (Pair A1)** (`Infant Sleep`): *"The infant is asleep."*
- **S2 (Pair A2)** (`Infant Sleep`): *"A newborn baby rests quietly."*
- **S3 (Pair B1)** (`Customer Support`): *"Initiate a refund for defective widget SKU-98421 returned by the customer."*
- **S4 (Pair B2)** (`Mechanical Engineering`): *"Review the tensile strength specifications and steel alloy blueprint for component SKU-98421."*
- **S5 (Food 1)** (`Cooking`): *"Whisk the egg yolks with fresh cream and sugar until smooth."*
- **S6 (Food 2)** (`Cooking`): *"Beat the milk, butter, and vanilla extract in a large glass bowl."*
- **S7 (Code 1)** (`Software`): *"Write a unit test for the REST API endpoint using pytest."*
- **S8 (Code 2)** (`Software`): *"Debug the async route handler in FastAPI to prevent memory leaks."*
- **S9 (Astronomy)** (`Astronomy`): *"Astronomers detected a supermassive black hole at the center of the distant galaxy."*
- **S10 (Finance)** (`Finance`): *"The central bank raised benchmark interest rates to combat rising inflation."*

---

## Cosine Similarity Matrix

| Sentence | **S1** | **S2** | **S3** | **S4** | **S5** | **S6** | **S7** | **S8** | **S9** | **S10** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 (Pair A1)** | **1.0000** | 0.8188 | 0.5308 | 0.5090 | 0.5453 | 0.5519 | 0.5093 | 0.5014 | 0.5357 | 0.5568 |
| **S2 (Pair A2)** | 0.8188 | **1.0000** | 0.5190 | 0.5077 | 0.5449 | 0.5235 | 0.5034 | 0.4792 | 0.5305 | 0.5215 |
| **S3 (Pair B1)** | 0.5308 | 0.5190 | **1.0000** | 0.5947 | 0.5205 | 0.5235 | 0.5400 | 0.4967 | 0.4803 | 0.4992 |
| **S4 (Pair B2)** | 0.5090 | 0.5077 | 0.5947 | **1.0000** | 0.5083 | 0.5255 | 0.5160 | 0.4883 | 0.5102 | 0.5136 |
| **S5 (Food 1)** | 0.5453 | 0.5449 | 0.5205 | 0.5083 | **1.0000** | 0.7042 | 0.4886 | 0.4481 | 0.5141 | 0.5536 |
| **S6 (Food 2)** | 0.5519 | 0.5235 | 0.5235 | 0.5255 | 0.7042 | **1.0000** | 0.4987 | 0.4480 | 0.4893 | 0.5486 |
| **S7 (Code 1)** | 0.5093 | 0.5034 | 0.5400 | 0.5160 | 0.4886 | 0.4987 | **1.0000** | 0.6029 | 0.4573 | 0.4900 |
| **S8 (Code 2)** | 0.5014 | 0.4792 | 0.4967 | 0.4883 | 0.4481 | 0.4480 | 0.6029 | **1.0000** | 0.4728 | 0.4697 |
| **S9 (Astronomy)** | 0.5357 | 0.5305 | 0.4803 | 0.5102 | 0.5141 | 0.4893 | 0.4573 | 0.4728 | **1.0000** | 0.5507 |
| **S10 (Finance)** | 0.5568 | 0.5215 | 0.4992 | 0.5136 | 0.5536 | 0.5486 | 0.4900 | 0.4697 | 0.5507 | **1.0000** |

---

## Key Findings & Anomaly Analysis

### 1. Pair A: Zero-Word Overlap Equivalence (S1 vs S2)

- **S1:** *"The infant is asleep."*
- **S2:** *"A newborn baby rests quietly."*
- **Shared Words Count:** `0` (Zero lexical overlap)
- **Cosine Similarity:** **`0.8188`**
- **Analysis:** Dense semantic capture: High similarity score despite zero lexical overlap, demonstrating that embeddings map conceptual meaning rather than keyword tokens.

### 2. Pair B: Shared Product Code (S3 vs S4)

- **S3 (Customer Support):** *"Initiate a refund for defective widget SKU-98421 returned by the customer."*
- **S4 (Mechanical Engineering):** *"Review the tensile strength specifications and steel alloy blueprint for component SKU-98421."*
- **Shared Identifiers:** `SKU-98421`, `for`
- **Cosine Similarity:** **`0.5947`**
- **Analysis:** Lexical collision with semantic divergence: Shows how a unique product ID token interacts with contrasting context (customer refund vs engineering spec).
