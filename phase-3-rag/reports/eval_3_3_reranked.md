# Evaluation Report: Phase 3.3: Two-Stage Reranked RAG

- **Evaluated Questions**: 40
- **Top-K Context Chunks**: 8
- **Average Latency**: 3.560s
- **Total Tokens Consumed**: 59737 prompt + 2485 completion

## 1. Summary Scorecard

| Stage | Metric | Score | Scale / Goal |
| :--- | :--- | :--- | :--- |
| **Retrieval** | **Hit@8** | **95.0%** | 100% (Target doc in top-k) |
| **Retrieval** | **Recall@8** | **93.8%** | 100% (Target coverage) |
| **Retrieval** | **Context Precision (MRR)** | **0.8831** | 0.0 to 1.0 (Rank quality) |
| **Generation** | **Faithfulness** | **61.9%** | 100% (Grounded claims) |
| **Generation** | **Answer Relevance** | **37.7%** | 100% (Key facts hit) |

## 2. Per-Question Detailed Breakdown

| ID | Type | Target Docs | Retrieved Docs | Hit | Recall | Precision | Faithfulness | Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q01 | factoid | doc_01 | doc_01,doc_15,doc_19,doc_20,doc_04,doc_18,doc_01,doc_01 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q02 | numeric | doc_01 | doc_05,doc_01,doc_01,doc_05,doc_01,doc_01,doc_11,doc_17 | 1 | 1.00 | 0.50 | 1.00 | 0.00 |
| q03 | factoid | doc_02 | doc_02,doc_12,doc_15,doc_10,doc_11,doc_05,doc_10,doc_06 | 1 | 1.00 | 1.00 | 0.67 | 0.33 |
| q04 | numeric | doc_02 | doc_02,doc_04,doc_13,doc_08,doc_15,doc_04,doc_05,doc_01 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q05 | factoid | doc_03 | doc_03,doc_14,doc_14,doc_07,doc_16,doc_18,doc_05,doc_10 | 1 | 1.00 | 1.00 | 0.00 | 0.25 |
| q06 | numeric | doc_03 | doc_03,doc_06,doc_07,doc_05,doc_09,doc_07,doc_14,doc_01 | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| q07 | numeric | doc_04 | doc_04,doc_05,doc_20,doc_18,doc_03,doc_11,doc_05,doc_17 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q08 | procedural | doc_04 | doc_17,doc_17,doc_03,doc_05,doc_04,doc_06,doc_17,doc_04 | 1 | 1.00 | 0.20 | 0.00 | 0.33 |
| q09 | factoid | doc_05 | doc_05,doc_08,doc_08,doc_05,doc_05,doc_07,doc_07,doc_01 | 1 | 1.00 | 1.00 | 1.00 | 0.33 |
| q10 | numeric | doc_05 | doc_05,doc_05,doc_04,doc_05,doc_18,doc_10,doc_13,doc_11 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q11 | factoid | doc_06 | doc_06,doc_05,doc_05,doc_06,doc_12,doc_02,doc_15,doc_01 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q12 | procedural | doc_06 | doc_05,doc_03,doc_07,doc_16,doc_17,doc_16,doc_11,doc_17 | 0 | 0.00 | 0.00 | 0.00 | 0.75 |
| q13 | numeric | doc_07 | doc_07,doc_02,doc_01,doc_06,doc_14,doc_05,doc_08,doc_07 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q14 | factoid | doc_07 | doc_07,doc_11,doc_20,doc_16,doc_16,doc_20,doc_07,doc_09 | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| q15 | numeric | doc_08 | doc_13,doc_05,doc_10,doc_03,doc_17,doc_07,doc_02,doc_08 | 1 | 1.00 | 0.12 | 0.00 | 0.33 |
| q16 | procedural | doc_08 | doc_08,doc_18,doc_01,doc_08,doc_18,doc_19,doc_16,doc_05 | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| q17 | factoid | doc_09 | doc_09,doc_13,doc_05,doc_10,doc_14,doc_16,doc_04,doc_07 | 1 | 1.00 | 1.00 | 0.00 | 0.25 |
| q18 | numeric | doc_09 | doc_09,doc_01,doc_07,doc_09,doc_05,doc_06,doc_13,doc_19 | 1 | 1.00 | 1.00 | 0.00 | 0.67 |
| q19 | factoid | doc_10 | doc_10,doc_20,doc_19,doc_03,doc_11,doc_05,doc_12,doc_04 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q20 | numeric | doc_10 | doc_10,doc_10,doc_03,doc_19,doc_16,doc_20,doc_10,doc_08 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q21 | numeric | doc_11 | doc_11,doc_03,doc_01,doc_10,doc_18,doc_17,doc_06,doc_19 | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| q22 | numeric | doc_11 | doc_07,doc_11,doc_20,doc_16,doc_06,doc_11,doc_06,doc_04 | 1 | 1.00 | 0.50 | 0.33 | 0.33 |
| q23 | factoid | doc_12 | doc_08,doc_08,doc_08,doc_01,doc_19,doc_19,doc_04,doc_19 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| q24 | numeric | doc_12 | doc_12,doc_18,doc_11,doc_15,doc_01,doc_11,doc_18,doc_12 | 1 | 1.00 | 1.00 | 1.00 | 0.67 |
| q25 | procedural | doc_13 | doc_13,doc_05,doc_11,doc_18,doc_05,doc_11,doc_16,doc_06 | 1 | 1.00 | 1.00 | 0.75 | 0.00 |
| q26 | factoid | doc_13 | doc_13,doc_13,doc_01,doc_03,doc_01,doc_11,doc_18,doc_12 | 1 | 1.00 | 1.00 | 0.00 | 0.00 |
| q27 | numeric | doc_14 | doc_14,doc_05,doc_07,doc_18,doc_05,doc_18,doc_19,doc_10 | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| q28 | factoid | doc_14 | doc_14,doc_14,doc_08,doc_14,doc_18,doc_06,doc_02,doc_19 | 1 | 1.00 | 1.00 | 0.75 | 1.00 |
| q29 | factoid | doc_15 | doc_15,doc_18,doc_10,doc_16,doc_13,doc_09,doc_11,doc_04 | 1 | 1.00 | 1.00 | 0.00 | 0.00 |
| q30 | numeric | doc_15 | doc_15,doc_06,doc_04,doc_02,doc_01,doc_01,doc_17,doc_03 | 1 | 1.00 | 1.00 | 0.00 | 0.00 |
| q31 | factoid | doc_16 | doc_16,doc_05,doc_16,doc_17,doc_07,doc_05,doc_10,doc_11 | 1 | 1.00 | 1.00 | 1.00 | 0.33 |
| q32 | procedural | doc_16 | doc_16,doc_16,doc_11,doc_09,doc_16,doc_13,doc_16,doc_19 | 1 | 1.00 | 1.00 | 0.75 | 1.00 |
| q33 | numeric | doc_17 | doc_17,doc_17,doc_17,doc_08,doc_05,doc_11,doc_01,doc_05 | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| q34 | factoid | doc_17 | doc_17,doc_17,doc_08,doc_17,doc_17,doc_19,doc_18,doc_06 | 1 | 1.00 | 1.00 | 0.00 | 0.25 |
| q35 | procedural | doc_18 | doc_18,doc_18,doc_11,doc_07,doc_11,doc_10,doc_08,doc_06 | 1 | 1.00 | 1.00 | 1.00 | 0.25 |
| q36 | procedural | doc_18 | doc_18,doc_18,doc_20,doc_09,doc_01,doc_06,doc_10,doc_10 | 1 | 1.00 | 1.00 | 0.50 | 0.00 |
| q37 | factoid | doc_19 | doc_19,doc_19,doc_05,doc_01,doc_19,doc_05,doc_08,doc_15 | 1 | 1.00 | 1.00 | 1.00 | 0.33 |
| q38 | multi_hop | doc_19,doc_01 | doc_19,doc_19,doc_20,doc_15,doc_07,doc_18,doc_06,doc_18 | 1 | 0.50 | 1.00 | 1.00 | 0.33 |
| q39 | factoid | doc_20 | doc_20,doc_04,doc_04,doc_10,doc_18,doc_01,doc_01,doc_18 | 1 | 1.00 | 1.00 | 0.00 | 0.00 |
| q40 | negative | None | doc_10,doc_17,doc_17,doc_10,doc_10,doc_18,doc_18,doc_13 | 1 | 1.00 | 1.00 | 0.00 | 0.00 |
