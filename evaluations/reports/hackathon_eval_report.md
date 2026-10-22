# ✈️ Maintenance Copilot — Benchmark Evaluation Report
**Evaluation Date:** 2026-10-22T09:30:02Z  
**LLM Engine:** `gemini-3.5-flash-lite` (Gemini 3.5 Flash Lite)  
**Retrieval Engine:** Hybrid RRF (BM25 + `all-MiniLM-L6-v2`, $k=60$)  
**Dataset Scope:** 6,169 Genuine Aviation Maintenance Logbooks  

---

## 🎯 Executive Summary (Hackathon Core Metrics)

| Metric | DEV Split (20 Queries) | Held-Out TEST (20 Queries) | Overall (40 Queries) |
|---|---|---|---|
| **Retrieval Hit@5 (15 pairs)** | **73.33%** (10/15) | **86.67%** (13/15) | **80.0%** (24/30) |
| **Mean Reciprocal Rank (MRR)** | **0.5889** | **0.7278** | **0.6583** |
| **Citation Precision** | **25.86%** | **34.67%** | **30.83%** |
| **Database Citation Validity** | **100.0%** | **100.0%** | **100.0%** (0 hallucinated IDs) |
| **Correct Refusals (5 unanswerable)** | **100.0%** (5/5) | **20.0%** (1/5) | **60.0%** (6/10) |
| **Average Query Latency** | **2137.84 ms** | **576.44 ms** | **1357.14 ms** |

---

## 📋 Primary Metric Breakdown: DEV Set (15 Pairs + 5 Refusals)

### 1. Retrieval Hit@5 on 15 Question-to-Record Pairs
- **Hit@1:** 53.33%
- **Hit@3:** 60.0%
- **Hit@5:** **73.33%**
- **MRR:** **0.5889**

### 2. Citation Precision
- **Share of Cited Records Supporting Answer:** **25.86%**
- **Total Cited IDs:** 58
- **Database Valid IDs:** 100.0% (all IDs exist in the 6,169 canonical record store)

### 3. Correct Refusals on 5 Questions the Data Cannot Answer
- **Correct Abstention Rate:** **100.0%** (5/5)
- **Hallucinated Citations on Out-of-Domain:** 5 (Zero citations hallucinated)
- Evaluated scenarios: autopilot altitude hold runaway, cabin oxygen mask failure, landing gear actuator overhaul, APU generator fire, thrust reverser cascade lock.

---

## 🔬 Detailed Query-by-Query Log

| Query ID | Split | Category | Query Snippet | Hit@5 | Cited IDs | Precision | Status |
|---|---|---|---|---|---|---|---|
| `DEV_01_EXACT_TECH` | dev | exact_technical_term | removed and replaced fuel servo... | ✅ YES | 100002 | 100.0% | sufficient |
| `DEV_02_EXACT_TECH` | dev | exact_technical_term | oil dipstick oil stick is loose... | ✅ YES | 100044 | 100.0% | sufficient |
| `DEV_03_EXACT_TECH` | dev | exact_technical_term | engine cowl Dzus fasteners need retain... | ✅ YES | 102030 | 100.0% | sufficient |
| `DEV_04_EXACT_TECH` | dev | exact_technical_term | stop drilled crack on L/H rear engine ... | ✅ YES | None | 100.0% | sufficient |
| `DEV_05_ABBREV_EXPANDED` | dev | abbreviation_expanded | left-hand rear engine baffle cracked... | ✅ YES | 101316, 100191, 104069 (+2) | 40.0% | sufficient |
| `DEV_06_ABBREV_EXPANDED` | dev | abbreviation_expanded | right-hand magneto blast tube fell off... | ✅ YES | 100027, 102286, 101150 (+2) | 20.0% | sufficient |
| `DEV_07_ABBREV_EXPANDED` | dev | abbreviation_expanded | cylinder head temperature probe fluctu... | ✅ YES | 104459, 102822, 106164 (+2) | 20.0% | sufficient |
| `DEV_08_ABBREV_EXPANDED` | dev | abbreviation_expanded | service instruction 388C reamed exhaus... | ❌ NO | 105421, 102026, 106081 (+2) | 0.0% | sufficient |
| `DEV_09_ABBREV_EXPANDED` | dev | abbreviation_expanded | engine overspeed revolutions per minut... | ✅ YES | 102759, 106023, 105563 (+2) | 20.0% | sufficient |
| `DEV_10_NOISY_MISSPELLING` | dev | misspelling_noisy | rockr box covr scrws lose... | ❌ NO | 100228, 105152, 106017 (+2) | 0.0% | sufficient |
| `DEV_11_NOISY_MISSPELLING` | dev | misspelling_noisy | intak gasget leeking #2... | ❌ NO | 104700, 103205, 102107 (+2) | 0.0% | insufficient |
| `DEV_12_NOISY_MISSPELLING` | dev | misspelling_noisy | stck exaust valv cylnder #3... | ❌ NO | 103026, 100303, 104571 (+2) | 0.0% | insufficient |
| `DEV_13_PARAPHRASED` | dev | paraphrased | engine was running rough and vibrating... | ✅ YES | 100130, 103912, 100079 (+2) | 40.0% | sufficient |
| `DEV_14_PARAPHRASED` | dev | paraphrased | shutting down the engine when idle ove... | ✅ YES | 100001, 100002, 101007 (+2) | 40.0% | sufficient |
| `DEV_15_PARAPHRASED` | dev | paraphrased | push rod tube shroud seal oil leak at ... | ✅ YES | 103806, 103805, 100386 (+2) | 60.0% | sufficient |
| `DEV_16_UNANSWERABLE` | dev | unanswerable | autopilot altitude hold disconnect and... | — (Refusal) | None | — | ✅ Refused (insufficient) |
| `DEV_17_UNANSWERABLE` | dev | unanswerable | cabin oxygen mask deployment failure a... | — (Refusal) | None | — | ✅ Refused (insufficient) |
| `DEV_18_UNANSWERABLE` | dev | unanswerable | landing gear hydraulic retraction actu... | — (Refusal) | None | — | ✅ Refused (insufficient) |
| `DEV_19_UNANSWERABLE` | dev | unanswerable | auxiliary power unit APU generator bea... | — (Refusal) | None | — | ✅ Refused (insufficient) |
| `DEV_20_UNANSWERABLE` | dev | unanswerable | thrust reverser cascade vane unlock in... | — (Refusal) | 105691, 105692, 104416 (+2) | — | ✅ Refused (insufficient) |
| `TEST_01_EXACT_TECH` | test | exact_technical_term | vacuum tube and oil cooler line rubbin... | ✅ YES | 100065, 104660, 101596 (+2) | 20.0% | sufficient |
| `TEST_02_EXACT_TECH` | test | exact_technical_term | fuel pressure transducer ground path c... | ✅ YES | 100067, 105769, 101633 (+2) | 20.0% | sufficient |
| `TEST_03_EXACT_TECH` | test | exact_technical_term | starter ring gear damaged teeth dresse... | ✅ YES | 100929, 100309, 100274 (+2) | 60.0% | sufficient |
| `TEST_04_EXACT_TECH` | test | exact_technical_term | engine removed to facilitate firewall ... | ✅ YES | 100253, 100186, 100184 (+2) | 60.0% | sufficient |
| `TEST_05_EXACT_TECH` | test | exact_technical_term | dirty sniffler valves cleaned... | ❌ NO | 105093, 105181, 104147 (+2) | 0.0% | sufficient |
| `TEST_06_ABBREV_EXPANDED` | test | abbreviation_expanded | alternator cooling hose chafing agains... | ✅ YES | 100637, 101071, 105980 (+2) | 20.0% | sufficient |
| `TEST_07_ABBREV_EXPANDED` | test | abbreviation_expanded | cylinder intake or exhaust valve stake... | ✅ YES | 102378, 102911, 105312 (+2) | 40.0% | sufficient |
| `TEST_08_ABBREV_EXPANDED` | test | abbreviation_expanded | spark plug access hole button plugs wo... | ❌ NO | 101697, 101475, 101073 (+2) | 0.0% | sufficient |
| `TEST_09_ABBREV_EXPANDED` | test | abbreviation_expanded | front crankshaft oil seal leaking repl... | ✅ YES | 103176, 103175, 105866 (+2) | 40.0% | sufficient |
| `TEST_10_ABBREV_EXPANDED` | test | abbreviation_expanded | torquemeter pressure and vent hoses re... | ✅ YES | 100415, 100416, 100413 (+2) | 100.0% | sufficient |
| `TEST_11_NOISY_MISSPELLING` | test | misspelling_noisy | throtle linkag dog bone lose bering... | ✅ YES | 100603, 103302, 105704 (+2) | 20.0% | sufficient |
| `TEST_12_NOISY_MISSPELLING` | test | misspelling_noisy | low compreshun 20/80 on cilinder 4... | ✅ YES | 103952, 106147, 101231 (+2) | 40.0% | sufficient |
| `TEST_13_PARAPHRASED` | test | paraphrased | re-swaged intake pipe at oil sump conn... | ✅ YES | 101308, 102527, 101650 (+2) | 60.0% | sufficient |
| `TEST_14_PARAPHRASED` | test | paraphrased | propeller low pitch stop adjusted afte... | ✅ YES | 102959, 103165, 100011 (+2) | 20.0% | sufficient |
| `TEST_15_PARAPHRASED` | test | paraphrased | cylinder 3 dry tappet clearance check ... | ✅ YES | 104620, 106099, 101459 (+2) | 20.0% | sufficient |
| `TEST_16_UNANSWERABLE` | test | unanswerable | trailing edge wing flap asymmetry lock... | — (Refusal) | 101613, 104288, 105606 (+2) | — | sufficient |
| `TEST_17_UNANSWERABLE` | test | unanswerable | bleed air precooler temperature contro... | — (Refusal) | 103581, 102164, 104868 (+2) | — | sufficient |
| `TEST_18_UNANSWERABLE` | test | unanswerable | cockpit voice recorder underwater loca... | — (Refusal) | 103309, 105605, 104230 (+2) | — | sufficient |
| `TEST_19_UNANSWERABLE` | test | unanswerable | engine fire extinguishing squib bottle... | — (Refusal) | 105590, 102694, 105360 (+2) | — | sufficient |
| `TEST_20_UNANSWERABLE` | test | unanswerable | emergency evacuation slide deployment ... | — (Refusal) | 102497, 102490, 102498 (+2) | — | ✅ Refused (insufficient) |
