# Benchmark Report: Adaptive 4-Route Omni-Router (Task 3.22)

**Date:** 2026-09-10T06:34:13Z  
**Pipeline:** `Adaptive Omni-Router (Task 3.21 & 3.22)`  
**Total Evaluated Questions:** 20 (Balanced across 4 routes)  

## 1. Executive Summary

| Metric | Score | Target | Status |
| :--- | :---: | :---: | :---: |
| **Overall Routing Accuracy** | **100.0%** | >= 90.0% | PASS |
| **Execution Success Rate** | **100.0%** | 100.0% | PASS |
| **Citation Compliance Rate** | **100.0%** | 100.0% | PASS |

## 2. Per-Route Performance Breakdown

| Knowledge Source Modality | Questions | Route Accuracy | Avg Retrieval Latency |
| :--- | :---: | :---: | :---: |
| **`DIRECT_LLM`** | 5 | 100.0% | 0.0 ms |
| **`LOCAL_CORPUS`** | 5 | 100.0% | 3759.51 ms |
| **`STRUCTURED_DB`** | 5 | 100.0% | 6382.3 ms |
| **`WEB_SEARCH`** | 5 | 100.0% | 16225.64 ms |

## 3. Detailed Question-by-Question Results

| ID | Question | Expected | Predicted | Match | Citations | Source Used |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `DIRECT-01` | Write a Python function to check if a word is a palindrome. | `DIRECT_LLM` | `DIRECT_LLM` | PASS | Parametric | Direct Parametric LLM |
| `DIRECT-02` | Calculate 128 * 4 + 56 - 12. | `DIRECT_LLM` | `DIRECT_LLM` | PASS | Parametric | Direct Parametric LLM |
| `DIRECT-03` | Hello! How are you doing today? | `DIRECT_LLM` | `DIRECT_LLM` | PASS | Parametric | Direct Parametric LLM |
| `DIRECT-04` | Explain the difference between mutable and immutable types in Python. | `DIRECT_LLM` | `DIRECT_LLM` | PASS | Parametric | Direct Parametric LLM |
| `DIRECT-05` | Translate 'Good morning, my friend' into Spanish. | `DIRECT_LLM` | `DIRECT_LLM` | PASS | Parametric | Direct Parametric LLM |
| `LOCAL-01` | What is the nominal cabin atmospheric pressure limit for the ECLSS? | `LOCAL_CORPUS` | `LOCAL_CORPUS` | PASS | 3 cited | Local Engineering Corpus (Documents) |
| `LOCAL-02` | What propellant mixture does the MDAS propulsion system use? | `LOCAL_CORPUS` | `LOCAL_CORPUS` | PASS | 2 cited | Local Engineering Corpus (Documents) |
| `LOCAL-03` | What is the maximum safe operating voltage of the primary power bus? | `LOCAL_CORPUS` | `LOCAL_CORPUS` | PASS | 3 cited | Local Engineering Corpus (Documents) |
| `LOCAL-04` | What fluid mixture is used in the habitat thermal control subsystem? | `LOCAL_CORPUS` | `LOCAL_CORPUS` | PASS | 3 cited | Local Engineering Corpus (Documents) |
| `LOCAL-05` | What is the emergency minimum cabin atmospheric pressure limit? | `LOCAL_CORPUS` | `LOCAL_CORPUS` | PASS | 3 cited | Local Engineering Corpus (Documents) |
| `DB-01` | How many total customers are in the database? | `STRUCTURED_DB` | `STRUCTURED_DB` | PASS | 2 cited | Structured Relational Database (SQL) |
| `DB-02` | Show all pending orders and their total amounts. | `STRUCTURED_DB` | `STRUCTURED_DB` | PASS | 2 cited | Structured Relational Database (SQL) |
| `DB-03` | List all appointments scheduled with doctor name and status. | `STRUCTURED_DB` | `STRUCTURED_DB` | PASS | 3 cited | Structured Relational Database (SQL) |
| `DB-04` | What is the stock quantity of all products currently in inventory? | `STRUCTURED_DB` | `STRUCTURED_DB` | PASS | 2 cited | Structured Relational Database (SQL) |
| `DB-05` | How many customers are located in New York or Chicago? | `STRUCTURED_DB` | `STRUCTURED_DB` | PASS | 2 cited | Structured Relational Database (SQL) |
| `WEB-01` | What are the latest features released in Python 3.13? | `WEB_SEARCH` | `WEB_SEARCH` | PASS | 2 cited | Live Web Search Engine |
| `WEB-02` | What is the current weather forecast for Tokyo today? | `WEB_SEARCH` | `WEB_SEARCH` | PASS | 2 cited | Live Web Search Engine |
| `WEB-03` | Who won the latest Arsenal football match yesterday? | `WEB_SEARCH` | `WEB_SEARCH` | PASS | 2 cited | Live Web Search Engine |
| `WEB-04` | What is the latest stock price of NVIDIA today? | `WEB_SEARCH` | `WEB_SEARCH` | PASS | 2 cited | Live Web Search Engine |
| `WEB-05` | What are the breaking technology news headlines this week? | `WEB_SEARCH` | `WEB_SEARCH` | PASS | 2 cited | Live Web Search Engine |
