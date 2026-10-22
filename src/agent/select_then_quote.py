"""
Select-Then-Quote Evidence Execution and Quotation Rendering Engine.
Validates unit IDs, enforces session inspection isolation, and renders verbatim quotes.
"""

import re
from typing import List, Dict, Set, Optional, Literal, Any
from pydantic import BaseModel, Field

from src.agent.segmenter import EvidenceUnit, segment_logbook_record

ClaimType = Literal["recorded_symptom", "recorded_action", "recorded_diagnosis", "interpretation"]
_UNIT_ID_RE = re.compile(r"^(\d{6})#u(\d+)$")


class StructuredClaim(BaseModel):
    claim_type: ClaimType = Field(..., description="Classification of the claim")
    unit_ids: List[str] = Field(default_factory=list, description="IDs of supporting evidence units e.g. ['100004#u3']")
    paraphrase: Optional[str] = Field(default=None, description="Optional concise summary or paraphrase (never includes quotes)")


class SelectThenQuoteResponse(BaseModel):
    query: str
    claims: List[StructuredClaim]
    rejected_unit_ids: List[str] = Field(default_factory=list)
    rendered_markdown: str = ""
    is_valid: bool = True
    total_valid_quotes_rendered: int = 0


def build_unit_registry_from_records(records: List[Dict[str, Any]]) -> Dict[str, EvidenceUnit]:
    """Builds a lookup mapping from unit_id -> EvidenceUnit for a collection of records."""
    registry: Dict[str, EvidenceUnit] = {}
    for r in records:
        rec_id = str(r.get("record_id", "")).strip()
        prob = str(r.get("problem_raw", ""))
        act = str(r.get("action_raw", ""))
        units = segment_logbook_record(rec_id, prob, act)
        for u in units:
            registry[u.unit_id] = u
    return registry


def validate_and_render_claims(
    query: str,
    claims: List[StructuredClaim],
    session_retrieved_record_ids: Set[str],
    unit_registry: Dict[str, EvidenceUnit],
    disclaimer: Optional[str] = None
) -> SelectThenQuoteResponse:
    """
    Validates unit IDs, strictly rejects non-retrieved unit IDs,
    renders quotes verbatim from immutable unit text,
    and isolates 'interpretation' claims under a tentative non-authoritative notice.
    """
    rejected_units: List[str] = []
    rendered_sections: List[str] = [
        f"### Historical Maintenance Logbook Evidence for: *{query}*",
        ""
    ]

    fact_lines: List[str] = []
    interpretation_lines: List[str] = []
    quotes_count = 0

    for claim in claims:
        valid_units_for_claim: List[EvidenceUnit] = []
        for uid in claim.unit_ids:
            uid_clean = str(uid).strip()
            match = _UNIT_ID_RE.match(uid_clean)
            if not match:
                rejected_units.append(f"{uid_clean} (malformed_format)")
                continue

            rec_id = match.group(1)
            # Rejection rule: Unit must belong to a record retrieved or inspected in this session
            if rec_id not in session_retrieved_record_ids:
                rejected_units.append(f"{uid_clean} (unretrieved_session_id)")
                continue

            if uid_clean not in unit_registry:
                rejected_units.append(f"{uid_clean} (nonexistent_unit)")
                continue

            valid_units_for_claim.append(unit_registry[uid_clean])

        # Render claim based on type
        if claim.claim_type == "interpretation":
            note = claim.paraphrase or "Tentative technical interpretation."
            cited_str = ", ".join([f"[ID: {u.record_id}]" for u in valid_units_for_claim]) if valid_units_for_claim else "General Analysis"
            interpretation_lines.append(f"- **Tentative Hypothesis ({cited_str}):** {note}")
        else:
            paraphrase_header = f"**{claim.paraphrase}**\n  " if claim.paraphrase else ""

            # Render quotes verbatim from canonical evidence units
            rendered_unit_quotes = []
            for u in valid_units_for_claim:
                # Verbatim quote rendered from immutable text by construction
                rendered_unit_quotes.append(f'- Record [ID: {u.record_id}] ({u.field.upper()} #{u.unit_index}): "{u.text}"')
                quotes_count += 1

            if rendered_unit_quotes:
                fact_lines.append(f"{paraphrase_header}" + "\n  ".join(rendered_unit_quotes))
            elif paraphrase_header:
                fact_lines.append(f"{paraphrase_header}(No valid supporting logbook unit found)")

    if fact_lines:
        rendered_sections.append("#### Verified Historical Logbook Excerpts:")
        rendered_sections.extend(fact_lines)
        rendered_sections.append("")

    if interpretation_lines:
        rendered_sections.extend([
            "#### ⚠️ Tentative Interpretation & Hypotheses (Non-Authoritative):",
            "> [!NOTE]",
            "> The following observations are AI-assisted tentative interpretations and NOT approved maintenance instructions or verified historical facts.",
            ""
        ])
        rendered_sections.extend(interpretation_lines)
        rendered_sections.append("")

    if rejected_units:
        rendered_sections.extend([
            "> [!WARNING]",
            f"> Rejected invalid or unretrieved unit citations: {', '.join(rejected_units)}",
            ""
        ])

    default_disclaimer = (
        "> [!NOTE]\n"
        "> Historical logbook entries reflect past recorded actions, not approved maintenance manual procedures."
    )
    rendered_sections.append(disclaimer or default_disclaimer)

    return SelectThenQuoteResponse(
        query=query,
        claims=claims,
        rejected_unit_ids=rejected_units,
        rendered_markdown="\n".join(rendered_sections),
        is_valid=(len(rejected_units) == 0),
        total_valid_quotes_rendered=quotes_count
    )
