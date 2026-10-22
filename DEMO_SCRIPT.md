# Maintenance Copilot — 3-Minute Hackathon Demo Script
**NEXAI 2026 Hackathon | Agentic RAG over Aviation Maintenance Logbooks**

---

## 1. The 30-Second Hook & Problem Statement

> **"Welcome judges. Every day, aircraft maintenance technicians face cryptic squawk reports written in dense jargon and inconsistent abbreviations across thousands of historical logbooks. If a technician asks a standard LLM for advice, the model hallucinate maintenance procedures—a critical safety violation.**
>
> **Enter Maintenance Copilot: an autonomous, evidence-gated agentic RAG assistant that queries 6,169 genuine aviation logbooks. It never invents records, enforces strict citation validation, distinguishes historical technician actions from approved FAA procedures, and safely abstains when evidence is missing."**

---

## 2. Live 2-Minute Demonstration Flow

### Demo Step 1: Specific Discrepancy & Instant Provenance
- **Action**: Click the sample button **`Cyl #2 Plug Fouled`** or type:
  `"What actions were taken when cylinder 2 lower spark plug was fouled?"`
- **Click**: `Investigate`
- **What to show the judges**:
  1. **Evidence Status Badge**: Displays `SUFFICIENT` in green.
  2. **Grounded Answer**: Directly identifies past incident where an engine choke on departure was traced to a fouled plug.
  3. **Clickable Citation `[ID: 100004]`**: Click `[ID: 100004]` to open the **Source Logbook Inspector** side drawer. Show judges the authentic raw text:
     - `PROBLEM`: `HAD ENGINE CHOKE & BRIEFLY LOSE POWER ON DEPARTURE. FULL THR`
     - `ACTION`: `PERFORMED ENGINE RUN UP, FOUND CYL 2 LOWER PLUG FOULED.`
  4. **Agent Execution Trace**: Expand the accordion to show the bounded loop: Step 1 `search_records()`, Step 2 `get_record('100004')`. Latency is under 20ms for retrieval.

---

### Demo Step 2: Divergent Repair Paths & Hardware Repairs
- **Action**: Click **`Baffle Crack Repair`** or type:
  `"Stop drilled crack on L/H rear engine baffle"`
- **What to show the judges**:
  1. The agent discovers that technicians took multiple divergent historical approaches for cracked baffles: stop-drilling with doubler patches (`100012`) vs full baffle replacement.
  2. **Policy Compliance**: Highlight the Policy Audit pill showing no causation fallacies and transparent presentation of alternatives.

---

### Demo Step 3: Ambiguous Squawk & Clarification Gate
- **Action**: Click **`Ambiguous Squawk`** or type:
  `"leak"`
- **What to show the judges**:
  1. The system does **not** guess or fabricate an answer.
  2. The deterministic evidence gate classifies the query as **`NEEDS CLARIFICATION`**.
  3. The assistant asks focused questions: *Which subsystem (rocker cover, magneto, intake)? Which cylinder? What phase of flight?*

---

### Demo Step 4: Out-of-Domain Abstention & Zero Hallucination
- **Action**: Click **`Out of Domain Query`** or type:
  `"Boeing 777 autopilot computer failure and yaw damper disconnect"`
- **What to show the judges**:
  1. The evidence gate evaluates the query against the corpus domain.
  2. The badge immediately turns amber: **`INSUFFICIENT EVIDENCE`**.
  3. The copilot politely abstains, stating that historical logbooks cover reciprocating engines, preventing dangerous hallucinations.

---

### Demo Step 5: Safety Disclaimer & Offline Dual-Mode
- **Action**: Point out the persistent amber **Descriptive Safety Notice**:
  > *"Historical logbook entries describe past actions recorded by technicians; they are NOT manufacturer-approved maintenance instructions or airworthiness directives. Always refer to the FAA-approved AMM."*
- **Explain Dual-Mode**: If the Gemini API experiences network outages, rate limits, or cold starts, the system automatically switches to a deterministic **retrieval-only fallback**, still delivering verified records with zero downtime.

---

## 3. Judge Defense Q&A Cheatsheet

### Q1: "How do you guarantee the model doesn't hallucinate fake record IDs?"
> **Answer**: *"We built a deterministic post-generation validator in Python (`src/agent/validator.py`). It parses all `[ID: 100xxx]` citations using regex, checks them against an $O(1)$ database lookup set, and confirms the record was retrieved in the current session. Any hallucinated ID is excised with `[INVALID_CITATION]` before display."*

### Q2: "Why use Hybrid Retrieval instead of pure semantic search?"
> **Answer**: *"Aviation logbooks contain precise alphanumeric part numbers (`MS20470`, `AN4-12A`, `L/H`, `2700 RPM`) that dense embeddings often blur, while BM25 lacks semantic understanding of synonyms. Our empirical benchmark of 40 queries proved that Hybrid Reciprocal Rank Fusion (RRF) achieves 86.7% Hit@5 on held-out test data, outperforming both pure BM25 and pure dense vectors."*

### Q3: "How did you discover abbreviations without labeled training data?"
> **Answer**: *"We created an unsupervised morphology and statistical discovery pipeline (`src/abbreviation/detector.py`). It discovered 332 candidate abbreviations with frequency $\ge 5$, isolating slash tokens (`L/H`), apostrophes (`CK'D`), and acronyms (`CYL`, `IAW`), saving them to an editable JSON glossary (`data/glossary.json`)."*

### Q4: "What happens if Gemini is down or rate-limited?"
> **Answer**: *"We engineered a dual-mode architecture. If the API returns a 429 or network timeout, the application seamlessly switches to offline retrieval mode, displaying genuine historical entries and evidence status without failing."*
