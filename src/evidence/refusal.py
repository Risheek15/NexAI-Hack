"""
Refusal & Abstention Formatting Engine.
Generates structured refusal statements when evidence is insufficient or queries are out-of-domain,
and prompts users with focused clarification questions when queries are underspecified.
"""

from typing import List, Dict, Any, Optional

SAFETY_DISCLAIMER = (
    "\n\n*(Note: Historical logbook entries describe past actions recorded by technicians; "
    "they are NOT manufacturer-approved maintenance instructions or airworthiness directives. "
    "Always refer to the FAA-approved Aircraft Maintenance Manual (AMM) and applicable Service Bulletins.)*"
)


def format_abstention_response(
    query: str,
    reason: str,
    closest_records: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Constructs a polite, clear, and grounded refusal statement when historical records
    cannot support an answer.
    """
    lines = [
        f"**Historical Evidence Insufficient to Answer Query**",
        f"",
        f"The historical maintenance logbooks do not contain sufficient evidence to answer: *\"{query}\"*.",
        f"**Reason:** {reason}",
    ]

    if closest_records:
        lines.append("")
        lines.append("For reference, the closest matching historical logbook entries found were:")
        for r in closest_records[:3]:
            lines.append(
                f"- **[ID: {r.get('record_id')}]** Problem: `{r.get('problem_raw')}` | Action: `{r.get('action_raw')}`"
            )

    lines.append(SAFETY_DISCLAIMER)
    return "\n".join(lines)


def format_clarification_response(query: str, reason: str) -> str:
    """
    Constructs a clarification question when the user query is too generic or ambiguous.
    """
    lines = [
        f"**Clarification Requested**",
        f"",
        f"The query *\"{query}\"* is too general or ambiguous to identify a specific historical incident.",
        f"**Reason:** {reason}",
        f"",
        f"**Please clarify:**",
        f"- Which specific component or subsystem was affected (e.g., induction tube, cylinder rocker cover, magneto blast tube, or cooling baffle)?",
        f"- Which cylinder number (e.g. Cylinder #1, #2, #3, or #4)?",
        f"- What specific symptom or phase of flight was observed (e.g. idle cutoff, departure power loss, oil leak, or high CHT)?",
        SAFETY_DISCLAIMER
    ]
    return "\n".join(lines)
