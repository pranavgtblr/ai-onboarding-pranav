# ⚖️ LLM-as-a-Judge Calibration & Agreement Report (Task 5.3)

**Evaluation Timestamp:** `2026-09-14T09:29:48Z`  
**Evaluator Model:** `gemini-3.5-flash-lite` (`google_genai`)  
**Dataset Size:** 30 Hand-Labeled Ground-Truth Samples  

## 1. Executive Summary Scorecard

| Metric | Value | Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Raw Agreement Rate** | **100.0%** | >= 85.0% | ✅ PASS |
| **Cohen's Kappa (κ)** | **1.0000** | >= 0.70 | ✅ PASS |
| **Reliability Tier** | **Near-Perfect / Strong Agreement** | Substantial | ✅ PASS |
| **F1 Score** | **1.0000** | >= 0.85 | ✅ PASS |
| **Precision (PASS)** | **100.0%** | >= 85.0% | ✅ PASS |
| **Recall (PASS)** | **100.0%** | >= 85.0% | ✅ PASS |

## 2. Confusion Matrix

```text
                        HUMAN ANNOTATOR
                      PASS          FAIL
JUDGE     PASS   TP = 15        FP = 0    (Leniency Error)
VERDICT   FAIL   FN = 0         TN = 15   (Harshness Error)
```

### Bias Analysis:
- **Leniency Bias (False Positive Rate):** 0.0% (0/30). Judge accepts ungrounded extrapolations.
- **Harshness Bias (False Negative Rate):** 0.0% (0/30). Judge penalizes valid synonymous phrasing.

## 3. Sample-by-Sample Evaluation Table

| ID | Category | Human | Judge | Agreement | Faithfulness | Hallucinations |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `SAMPLE-01` | grounded_factoid | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-02` | grounded_numeric | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-03` | subtle_hallucination | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `liquid hydrogen` |
| `SAMPLE-04` | grounded_procedural | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-05` | numerical_distortion | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `84 kwh` |
| `SAMPLE-06` | grounded_synthesis | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-07` | unsupported_extrapolation | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `reverse osmosis` |
| `SAMPLE-08` | grounded_factoid | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-09` | off_topic_evasion | **FAIL** | **FAIL** | ✅ MATCH | 1/5 | None |
| `SAMPLE-10` | grounded_catalog | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-11` | fabricated_spec | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `bluetooth 5.4` |
| `SAMPLE-12` | grounded_negative | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-13` | hallucinated_entity | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `kobe titanium` |
| `SAMPLE-14` | grounded_factoid | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-15` | contradictory_statement | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `pneumatic seal automatically` |
| `SAMPLE-16` | grounded_comparison | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-17` | omission_distortion | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `cotton gardening gloves` |
| `SAMPLE-18` | grounded_factoid | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-19` | subtle_hallucination | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `european space agency` |
| `SAMPLE-20` | grounded_definition | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-21` | subtle_hallucination | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `mojo syntax` |
| `SAMPLE-22` | temporal_hallucination | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `1971` |
| `SAMPLE-23` | grounded_tabular | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-24` | irrelevant_hallucination | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `otis` |
| `SAMPLE-25` | grounded_instructional | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-26` | partial_factual_error | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `50 cm` |
| `SAMPLE-27` | grounded_factoid | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-28` | ungrounded_justification | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `damaged during transport` |
| `SAMPLE-29` | grounded_status | **PASS** | **PASS** | ✅ MATCH | 5/5 | None |
| `SAMPLE-30` | subtle_hallucination | **FAIL** | **FAIL** | ✅ MATCH | 2/5 | `kyber-1024` |

## 4. Disagreement Breakdown

**Zero disagreements detected.** The LLM judge achieved 100% concordance with human ground truth on this dataset.
