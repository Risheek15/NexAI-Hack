"""
Deterministic evidence-unit segmenter for aviation maintenance records.
Splits original immutable text into stable, addressable evidence units:
  Format: "<record_id>#u<n>"
"""

import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


class EvidenceUnit(BaseModel):
    unit_id: str = Field(..., description="Stable unit identifier e.g. '100004#u1'")
    record_id: str = Field(..., description="Parent 6-digit record ID")
    unit_index: int = Field(..., description="1-based unit sequence within record")
    field: str = Field(..., description="'problem' or 'action'")
    claim_type: str = Field(..., description="'recorded_symptom' or 'recorded_action'")
    text: str = Field(..., description="Verbatim immutable source text segment")


def segment_logbook_record(record_id: str, problem_raw: str, action_raw: str) -> List[EvidenceUnit]:
    """
    Deterministically segments a logbook record into stable evidence units.
    Splits problem and action text by sentence boundaries.
    Guarantees deterministic numbering and 100% exact verbatim preservation of original text.
    """
    clean_id = str(record_id).strip()
    units: List[EvidenceUnit] = []
    idx = 1

    # 1. Segment problem_raw (recorded symptoms / discrepancies)
    prob_clean = problem_raw.strip() if problem_raw else ""
    if prob_clean:
        clauses = [c.strip() for c in _SENTENCE_SPLIT_RE.split(prob_clean) if c.strip()]
        for clause in clauses:
            units.append(EvidenceUnit(
                unit_id=f"{clean_id}#u{idx}",
                record_id=clean_id,
                unit_index=idx,
                field="problem",
                claim_type="recorded_symptom",
                text=clause
            ))
            idx += 1

    # 2. Segment action_raw (recorded corrective actions)
    act_clean = action_raw.strip() if action_raw else ""
    if act_clean:
        clauses = [c.strip() for c in _SENTENCE_SPLIT_RE.split(act_clean) if c.strip()]
        for clause in clauses:
            units.append(EvidenceUnit(
                unit_id=f"{clean_id}#u{idx}",
                record_id=clean_id,
                unit_index=idx,
                field="action",
                claim_type="recorded_action",
                text=clause
            ))
            idx += 1

    return units
