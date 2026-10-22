# Frozen Baseline Benchmark Report — DEV Split Only
**Date:** 2026-10-22T09:25:36Z | **Seed:** 42 | **Split:** Development (N=20)

## 1. DEV Set Empirical Results
| Retrieval Mode | Hit@5 (Count) [95% CI] | MRR [95% CI] | Citation ID Validity | Quote Validation Fail Rate | Correct Abstention | False Refusal | Latency p50 / p95 | Error Rate |
|---|---|---|---|---|---|---|---|---|
| **BM25 (Baseline)** | 80.0% (12/15) [60.0%, 100.0%] | 0.6056 [0.4054, 0.8167] | 100.0% (75/75) | 0.0% (0/15) | 60.0% (3/5) | 13.3% (2/15) | 3.29 ms / 4.22 ms | 0.0% (0/20) |
| **Dense Semantic** | 60.0% (9/15) [33.3%, 86.7%] | 0.4689 [0.2356, 0.6890] | 100.0% (75/75) | 13.3% (2/15) | 60.0% (3/5) | 0.0% (0/15) | 7.75 ms / 148.75 ms | 0.0% (0/20) |
| **Hybrid RRF (Baseline)** | 73.3% (11/15) [46.7%, 93.3%] | 0.5889 [0.3610, 0.8167] | 100.0% (75/75) | 6.7% (1/15) | 60.0% (3/5) | 13.3% (2/15) | 10.91 ms / 13.45 ms | 0.0% (0/20) |
| **Hybrid RRF (Abbr Expanded)** | 73.3% (11/15) [46.7%, 93.3%] | 0.6000 [0.3772, 0.8222] | 100.0% (75/75) | 6.7% (1/15) | 60.0% (3/5) | 13.3% (2/15) | 9.7 ms / 11.59 ms | 0.0% (0/20) |
| **BM25 (Abbr Expanded)** | 80.0% (12/15) [60.0%, 100.0%] | 0.6000 [0.3833, 0.8167] | 100.0% (75/75) | 0.0% (0/15) | 60.0% (3/5) | 13.3% (2/15) | 3.43 ms / 4.35 ms | 0.0% (0/20) |

## 2. Configuration & Model Identifiers
- **Embedding Model:** `all-MiniLM-L6-v2` (Dense 384-d normalized unit vectors)
- **Gemini Model (Configured):** `gemini-2.5-flash`
- **Hybrid Fusion:** Reciprocal Rank Fusion ($k=60$)
- **Evaluation Top-K:** `5`
- **Random Seed:** `42`

## 3. Dataset & Artifact Integrity (SHA-256)
| File | SHA-256 Hash |
|---|---|
| `data/raw/logbooks.csv` | `c78205fd400bf2c4216fbf24fa4135ff9f109ee0fe6b1bb2a6dfb0b3b6661e5e` |
| `data/processed/records.jsonl` | `49375e8b967d68ef0c393a922658adde7e44c865ac6da91df58a1e596cf4cec7` |
| `data/processed/records_lookup.json` | `e0e88246fcc66a0acdeb28e0954f3ac16187ea34e1e12b882ac09a678f285414` |
| `data/processed/embedding_ids.json` | `9d7f544cba90dabf00670af3362602825d4a02b0ca6dfd25502a53c688e02b98` |
| `data/processed/embeddings.npy` | `01aab8bfdc742c2fae38cc1df911f5e37321b77d4223a4b5ebefba3290f2cc83` |
| `data/glossary.json` | `ee868ad6792cc4f41b1ddf2d42d304b3e8f5f3544256700655b2dbac5b1fc44f` |
| `data/eval_queries_40.json` | `89782c4c2bd27e8e3f1250c29bd4fd06c2a1c41a6b36672fa7f03a06e8d2a24f` |
| `.env` | `b006fa101a9c4db83636f197283111bccdc9356b224cf0ea49f6ce168b539ad7` |

## 4. Reliability Analysis & Known Weaknesses
1. **Small Sample Size ($N=20$, $N_{\text{ans}}=15$, $N_{\text{unans}}=5$):**
   - Each answerable query accounts for $1/15 = 6.67\%$ of Hit@5 and MRR.
   - Each unanswerable query accounts for $1/5 = 20.0\%$ of the abstention rate.
2. **Target Imbalance:**
   - Single-target queries (e.g. `DEV_01`, `DEV_02`, `DEV_06`, `DEV_08`) test strict precision.
   - Multi-target queries (e.g. `DEV_10`, `DEV_11`, `DEV_12` with 6-7 targets) disproportionately elevate Hit@5.
3. **Lexical Proximity:**
   - Exact technical queries mirror the verbatim token sequences from `problem_raw` / `action_raw`.
4. **Strict Isolation of Held-Out Test Set:**
   - Test split was entirely excluded from this run to prevent any leakage or tuning.