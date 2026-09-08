# Comparative PDF Parser Benchmark: pypdf vs. PyMuPDF Layout-Aware

This report provides an empirical, side-by-side engineering evaluation comparing **pypdf** (standard pure-Python PDF reader) against **PyMuPDF (Layout-Aware)** across three critical failure modes encountered in production RAG systems.

---

## 1. Executive Summary & Failure Mode Matrix

| Challenge | Symptom | pypdf | PyMuPDF | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Column** | Text interleaves | ❌ Scrambled | ✅ Clean | Hallucinated facts; broken sentence semantics |
| **Headers/Footers** | Boilerplate in chunks | ❌ Polluted | ✅ Clean | Context window waste; search noise |
| **Tables/Specs** | Cells flatten to text | ❌ Flattened | ✅ Structured | LLM cannot associate values with column headers |

---

## 2. Quantitative Performance & Throughput Benchmark

| Document | pypdf Latency | PyMuPDF Latency | Latency Delta | pypdf Words | PyMuPDF Words | Tables |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `multi_column_sample.pdf` | 7.1ms | 25.8ms (+18.7ms) | 0.3x | 102 | 96 | 0 |
| `polluted_header_footer_sample.pdf` | 11.1ms | 42.1ms (+31.0ms) | 0.3x | 225 | 153 | 0 |
| `table_sample.pdf` | 9.7ms | 25.0ms (+15.3ms) | 0.4x | 78 | 197 | 1 |

---

## 3. Deep-Dive Failure Mode Analysis

### Failure Mode: Multi Column (`multi_column_sample.pdf`)

**Diagnostic Summary**: pypdf extracts text strictly by descending Y-coordinate order. When two columns share horizontal Y-coordinates, words from Column 1 and Column 2 are interleaved horizontally into the same sentence. PyMuPDF layout-aware segmentation isolates Column 1 completely.

- **pypdf Status**: **❌ FAILS**
- **PyMuPDF Status**: **✅ PASSES**

#### Side-by-Side Text Comparison:

**[1] pypdf Raw Output:**
```text
Odyssey Mission Technical Digest: Dual-System Summary
COLUMN 1: ROVER EXPLORATION COLUMN 2: NUCLEAR REACTOR
The Martian rover Odyssey traversed 14 Primary Stirling reactor unit Beta
kilometers across the crater floor. generated 9.8 kilowatts electrical.
Soil samples revealed magn
```

**[2] PyMuPDF Layout-Aware Output:**
```markdown
COLUMN 1: ROVER EXPLORATION The Martian rover Odyssey traversed 14 kilometers across the crater floor. Soil samples revealed magnesium sulfate deposits and hydrated silica beds. Wheels exhibited minor abrasive wear on the titanium grousers, but torque margins remained above 85 pe
```

---

### Failure Mode: Header Footer Pollution (`polluted_header_footer_sample.pdf`)

**Diagnostic Summary**: pypdf blindly ingests running headers (3 times) and running footers (3 times). In a 200-token RAG chunking setup, this repetitive boilerplate pollutes chunks, consumes context window budget, and artificially triggers search hits. PyMuPDF layout filtering suppresses margin noise.

- **pypdf Status**: **❌ FAILS**
- **PyMuPDF Status**: **✅ PASSES**

#### Side-by-Side Text Comparison:

**[1] pypdf Raw Output:**
```text
=== RESTRICTED // ODYSSEY BASE OPERATIONS // DO NOT DISTRIBUTE ===
Section 1: Atmospheric Scrubbing Protocols
The habitat life support system relies on dual regenerative amine beds for carbon
dioxide adsorption. Every 30 minutes, the desorbing bed is
```

**[2] PyMuPDF Layout-Aware Output:**
```markdown
Section 1: Atmospheric Scrubbing Protocols
The habitat life support system relies on dual regenerative amine beds for carbon
dioxide adsorption. Every 30 minutes, the desorbing bed is heated to 120 degrees
Celsius under partial vacuum, releasing gase
```

---

### Failure Mode: Table Flattening (`table_sample.pdf`)

**Diagnostic Summary**: pypdf extracts table contents as an unstructured line stream: column headers and cell values collapse together without delimiters, destroying relational column-value mappings. PyMuPDF identifies table vector lines and formats relational data into GitHub Flavored Markdown.

- **pypdf Status**: **❌ FAILS**
- **PyMuPDF Status**: **✅ PASSES**

#### Side-by-Side Text Comparison:

**[1] pypdf Raw Output:**
```text
Critical Subsystem Operating Limits Table
Engineering specification matrix testing relational extraction fidelity.
Subsystem Category Sensor Metric Nominal Value Unit Critical Limit
Life Support Habitat Pressure 101.3 kPa < 95.0 kPa
Life Support O2 Partial Pressure 21.0 kPa < 19.5 kPa
Power Systems
```

**[2] PyMuPDF Layout-Aware Output:**
```markdown
### Extracted Tables

**Table 1**

| Subsystem Category | Sensor Metric | Nominal Value | Unit | Critical Limit |
| :--- | :--- | :--- | :--- | :--- |
| Life Support | Habitat Pressure | 101.3 | kPa | < 95.0 kPa |
| Life Support | O2 Partial Pressure | 21.0 | kPa | < 19.5 kPa |
| Power Systems | Stirling Bus Voltage | 120.0 | VDC | < 110.0 VDC |
|
```

---

## 4. Architectural Recommendations for Enterprise RAG

1. **Never use naive Y-coordinate extraction for multi-column documents**: Horizontal line extraction merges unrelated columns into syntactically corrupt sentences.
2. **Implement deterministic header/footer margin bounding filters**: Suppress running headers and page footers to prevent repetitive noise from wasting chunk tokens.
3. **Convert tables to Markdown before chunking**: Preserve tabular relationships so cross-encoders and LLMs can answer exact numeric questions accurately.
