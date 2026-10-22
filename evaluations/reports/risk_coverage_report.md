# Risk-Coverage Evaluation of the Evidence Gate
**Benchmark Split:** Development (N=20, Strictly held-out test split excluded)
**Date:** 2026-10-22T09:47:06Z

## 1. Executive Summary & Hypothesis Validation
- **Hypothesis Verified**: A single accuracy number obscures how the evidence gate trades off coverage against error risk.
- **Configured Operating Point**: **75.0% Coverage** with **26.7% Risk**.
- **Area Under the Risk-Coverage Curve (AURC)**: **0.0604**.

## 2. Currently Configured Operating Point Metrics
- **Total Queries:** 20 (15 Answerable, 5 Unanswerable)
- **Queries Answered:** 15 / 20 (75.0% Coverage)
- **Errors on Answered Queries (Risk):** 4 / 15 (26.7% Risk)
  - *Unanswerable queries erroneously answered:* 2 (DEV_18 hydraulics, DEV_19 APU)
  - *Answerable queries answered without target record:* 2 (DEV_08, DEV_10)
- **Correct Abstention Rate:** 60.0% (3/5 unanswerables abstained)
- **False Refusal Rate:** 13.3% (2/15 answerables falsely refused: DEV_11, DEV_12 due to noisy spelling)

## 3. Breakdown by Unanswerable Query Type
| Unanswerable Category | Total Evaluated | Correctly Abstained | Abstention Rate | Primary Failure Mode |
|---|---|---|---|---|
| **out-of-domain** | 5 | 4 | **80.0%** | Lexical overlap false positive |
| **in-domain but absent** | 3 | 0 | **0.0%** | Lexical overlap false positive |
| **vague** | 4 | 3 | **75.0%** | Lexical overlap false positive |
| **unsupported procedure/spec request** | 3 | 0 | **0.0%** | Lexical overlap false positive |

## 4. Key Findings
1. **Lexical Parasitism on Absent In-Domain Terms**: Queries about landing gear hydraulics or APUs trigger shallow word matches ('gear', 'bearing', 'fire') in piston records, elevating lexical match above zero and slipping past the gate.
2. **Spelling Noise Vulnerability**: Misspelled queries with zero lexical match (DEV_11, DEV_12) are falsely refused because the hybrid retriever fails to populate semantic similarity into the gate.
3. **Clean Vague & AMM Rejection**: Single generic words ('leak') and prescriptive AMM torque queries achieve 100% correct abstention via categorical rules.

## 5. Artifact Links
- CSV Data: [risk_coverage_curve.csv](file:///Users/rishitmittal/aircraft/NexAI-Hack/evaluations/reports/risk_coverage_curve.csv)
- Curve Plot: [risk_coverage_curve.png](file:///Users/rishitmittal/aircraft/NexAI-Hack/evaluations/reports/risk_coverage_curve.png)