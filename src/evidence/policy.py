"""
Evidence Policy Engine.
Audits generated responses to ensure factual grounding, identifies conflicting evidence,
distinguishes direct historical evidence from inference, enforces mandatory safety disclaimers,
and strictly prevents presenting historical technician actions as approved limits or procedures.
"""

import re
from typing import List, Dict, Any, Set, Tuple, Optional
from pydantic import BaseModel, Field

from src.agent.validator import extract_cited_ids, extract_direct_quotes

_SAFETY_DISCLAIMER_KEYWORDS = [
    "not approved maintenance", "not manufacturer-approved",
    "historical logbook entries describe", "past actions taken by",
    "descriptive log entries", "not approved repair procedures",
    "do not establish approved", "refer to applicable maintenance manual",
    "consult the aircraft maintenance manual", "consult applicable amm"
]

_CAUSATION_WARNING_TERMS = [
    "definitely caused", "proves causation", "the sole cause of",
    "will always cause", "guarantees that"
]

_PRESCRIPTIVE_PATTERNS = [
    re.compile(r"\b(torque\s+(?:spec|specs|specification|specifications|value|limits?))\b", re.IGNORECASE),
    re.compile(r"\b(allowable\s+(?:limits?|wear|tolerances?))\b", re.IGNORECASE),
    re.compile(r"\b(approved\s+(?:procedure|procedures|steps|instructions?|method))\b", re.IGNORECASE),
    re.compile(r"\b(airworthiness\s+(?:determination|release|sign[\s-]*off|approval))\b", re.IGNORECASE),
    re.compile(r"\b(how\s+should\s+i\s+torque|exact\s+torque)\b", re.IGNORECASE),
]

PRESCRIPTIVE_SAFETY_ADVISORY = (
    "> [!WARNING]\n"
    "> **Prescriptive & Airworthiness Boundary Notice**\n"
    "> Historical maintenance records document descriptive past actions by individual technicians. "
    "> They do **not** constitute manufacturer-approved procedures, torque specifications, or airworthiness determinations. "
    "> For approved torque values, limits, and repair procedures, always consult the current **Aircraft Maintenance Manual (AMM)**, "
    "> Engine Overhaul Manual, and authorized Type Certificate Data Sheets (TCDS), and have work certified by a licensed A&P/IA technician."
)


class PolicyAuditResult(BaseModel):
    is_compliant: bool
    has_safety_disclaimer: bool
    abstention_enforced: bool
    clarification_enforced: bool
    is_prescriptive_query: bool = False
    prescriptive_boundary_applied: bool = False
    potential_causation_fallacies: List[str] = Field(default_factory=list)
    conflicting_evidence_notes: List[str] = Field(default_factory=list)
    unsupported_claim_warnings: List[str] = Field(default_factory=list)
    policy_summary: str


def is_prescriptive_or_regulatory_request(query: str) -> Tuple[bool, Optional[str]]:
    """
    Detects whether a query seeks approved maintenance instructions, torque specifications,
    allowable wear limits, or airworthiness determinations that cannot be established
    by historical logbook records alone.
    """
    if not query:
        return False, None
    for pat in _PRESCRIPTIVE_PATTERNS:
        match = pat.search(query)
        if match:
            return True, match.group(0)
    return False, None


def check_safety_disclaimer_present(text: str) -> bool:
    """Checks whether the response explicitly distinguishes historical squawk logs from approved procedures."""
    lower_text = text.lower()
    return any(phrase in lower_text for phrase in _SAFETY_DISCLAIMER_KEYWORDS)


def detect_causation_fallacies(text: str) -> List[str]:
    """Detects unsupported claims of definitive causation from mere historical similarity."""
    lower_text = text.lower()
    found = []
    for term in _CAUSATION_WARNING_TERMS:
        if term in lower_text:
            found.append(f"Response contains strong causation assertion: '{term}'")
    return found


def detect_conflicting_historical_actions(retrieved_records: List[Dict[str, Any]]) -> List[str]:
    """
    Identifies conflicting or divergent corrective actions across historical records for similar squawks.
    (e.g., in some cases tightening screws resolved the issue, whereas in others full component replacement was required).
    """
    conflicts = []
    if len(retrieved_records) < 2:
        return conflicts

    actions = [r.get("action_raw", "").upper() for r in retrieved_records]
    has_tighten = any("TIGHTENED" in a or "TORQUED" in a for a in actions)
    has_replace = any("REPLACED" in a or "INSTALLED NEW" in a for a in actions)
    has_drilled = any("STOP DRILLED" in a or "PATCH" in a for a in actions)

    differing = []
    if has_tighten:
        differing.append("tightening/retorquing existing hardware")
    if has_replace:
        differing.append("part/gasket replacement")
    if has_drilled:
        differing.append("stop-drilling cracks / fabricating patches")

    if len(differing) >= 2:
        conflicts.append(
            f"Historical records report divergent repair paths for similar symptoms: {', '.join(differing)}. "
            f"The response should present these as separate historical alternatives rather than a single consensus."
        )

    return conflicts


def audit_evidence_policy(
    query: str,
    answer_text: str,
    evidence_status: str,
    retrieved_records: List[Dict[str, Any]],
    cited_record_ids: List[str],
    is_fallback: bool = False
) -> PolicyAuditResult:
    """
    Comprehensive audit of answer compliance against the Maintenance Copilot Evidence Policy.
    """
    has_disclaimer = check_safety_disclaimer_present(answer_text)
    causation_warnings = detect_causation_fallacies(answer_text)
    conflicts = detect_conflicting_historical_actions(retrieved_records)
    unsupported_warnings = []

    is_prescriptive, pres_term = is_prescriptive_or_regulatory_request(query)
    prescriptive_boundary_applied = False
    if is_prescriptive:
        lower_ans = answer_text.lower()
        if any(w in lower_ans for w in ["cannot establish", "consult", "amm", "maintenance manual", "authorized", "approved procedure"]):
            prescriptive_boundary_applied = True
        else:
            unsupported_warnings.append(
                f"Query requested prescriptive information ('{pres_term}'), but response failed to apply the authoritative AMM boundary."
            )

    # 1. Abstention check
    abstention_enforced = False
    if evidence_status == "insufficient":
        lower_ans = answer_text.lower()
        if any(w in lower_ans for w in ["insufficient", "cannot support", "no matching", "unavailable", "abstain", "no records"]):
            abstention_enforced = True
        else:
            unsupported_warnings.append(
                "Evidence status is 'insufficient', but the answer attempted to synthesize a factual claim rather than abstaining."
            )

    # 2. Clarification check
    clarification_enforced = False
    if evidence_status == "needs_clarification":
        lower_ans = answer_text.lower()
        if any(w in lower_ans for w in ["clarify", "specify", "too generic", "which component", "which cylinder"]):
            clarification_enforced = True
        else:
            unsupported_warnings.append(
                "Query was flagged as 'needs_clarification', but the answer did not ask a clarifying question."
            )

    # 3. Citation check on factual answers
    if evidence_status == "sufficient" and not cited_record_ids and not is_fallback:
        unsupported_warnings.append(
            "Evidence status is 'sufficient', but no record IDs were cited in the response."
        )

    is_compliant = (
        has_disclaimer
        and len(causation_warnings) == 0
        and len(unsupported_warnings) == 0
    )

    summary_parts = []
    if is_compliant:
        summary_parts.append("Answer is fully grounded and policy-compliant.")
    else:
        summary_parts.append("Policy violations detected:")
        if not has_disclaimer:
            summary_parts.append("Missing mandatory descriptive safety disclaimer.")
        summary_parts.extend(unsupported_warnings)
        summary_parts.extend(causation_warnings)

    return PolicyAuditResult(
        is_compliant=is_compliant,
        has_safety_disclaimer=has_disclaimer,
        abstention_enforced=abstention_enforced,
        clarification_enforced=clarification_enforced,
        is_prescriptive_query=is_prescriptive,
        prescriptive_boundary_applied=prescriptive_boundary_applied,
        potential_causation_fallacies=causation_warnings,
        conflicting_evidence_notes=conflicts,
        unsupported_claim_warnings=unsupported_warnings,
        policy_summary=" ".join(summary_parts)
    )
