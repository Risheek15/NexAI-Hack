"""
Abbreviation suggestion engine.
Interrogates Gemini using structured output schemas based ONLY on supplied logbook excerpts.
Includes a deterministic fallback for offline test suites and reproducible benchmarks.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from src.abbreviation.models import GlossaryEntry, ValidationStatus

# Core curated aviation domain knowledge base with standard meanings
_KNOWN_AVIATION_MAPPINGS: Dict[str, Tuple[str, ValidationStatus, List[str], str]] = {
    "CYL": ("cylinder", "validated", [], "Standard reciprocating engine cylinder"),
    "CYLS": ("cylinders", "validated", [], "Plural of cylinder"),
    "L/H": ("left-hand", "validated", [], "Aircraft left-hand orientation"),
    "R/H": ("right-hand", "validated", [], "Aircraft right-hand orientation"),
    "ENG": ("engine", "validated", [], "Aircraft engine"),
    "ENGS": ("engines", "validated", [], "Plural of engine"),
    "A/C": ("aircraft", "validated", ["air conditioning"], "Aircraft in this logbook context"),
    "IAW": ("in accordance with", "validated", [], "Standard regulatory maintenance procedure reference"),
    "OPS": ("operational", "validated", ["operations"], "Operational check"),
    "CK": ("check", "validated", [], "Maintenance check or inspection"),
    "CK'D": ("checked", "validated", [], "Past tense of check"),
    "CK'ED": ("checked", "validated", [], "Past tense of check"),
    "MAG": ("magneto", "validated", [], "Engine ignition magneto"),
    "MAGS": ("magnetos", "validated", [], "Plural of magneto"),
    "RPM": ("revolutions per minute", "validated", [], "Rotational speed unit"),
    "PSI": ("pounds per square inch", "validated", [], "Pressure unit"),
    "CHT": ("cylinder head temperature", "validated", [], "Engine cylinder head temperature indicator"),
    "EGT": ("exhaust gas temperature", "validated", [], "Engine exhaust gas temperature indicator"),
    "FWD": ("forward", "validated", [], "Forward airframe position"),
    "AFT": ("aft", "validated", [], "Rearward airframe position"),
    "I/B": ("inboard", "validated", [], "Inboard orientation relative to fuselage"),
    "O/B": ("outboard", "validated", [], "Outboard orientation relative to fuselage"),
    "LWR": ("lower", "validated", [], "Lower position"),
    "UPR": ("upper", "validated", [], "Upper position"),
    "ALT": ("alternator", "validated", ["altitude"], "Context indicates alternator attach bolt / cooling hose"),
    "ASSY": ("assembly", "validated", [], "Mechanical assembly"),
    "INSP": ("inspection", "validated", [], "Maintenance inspection"),
    "INSP'D": ("inspected", "validated", [], "Inspected past tense"),
    "PROP": ("propeller", "validated", [], "Aircraft propeller"),
    "DISCREP": ("discrepancy", "validated", [], "Squawk or discrepancy"),
    "FDM": ("flight data monitoring", "validated", [], "Flight data monitoring card"),
    "FOD": ("foreign object debris", "validated", [], "Foreign object debris / damage"),
    "RTV": ("room-temperature vulcanizing silicone", "validated", [], "High-temp sealing silicone"),
    "SRM": ("structural repair manual", "validated", [], "Manufacturer structural repair manual"),
    "MM": ("maintenance manual", "validated", [], "Maintenance manual reference"),
    "T/O": ("takeoff", "validated", [], "Takeoff phase of flight"),
    "FLT": ("flight", "validated", [], "Flight operation"),
    "EXH": ("exhaust", "validated", [], "Exhaust valve or stack"),
    "EXT": ("exhaust", "needs_review", ["external"], "Ambiguous between exhaust and external in some contexts"),
    "PD": ("pilot defect", "needs_review", ["pre-delivery", "production defect"], "Needs review / ambiguous"),
    "SI": ("service instruction", "needs_review", [], "Lycoming Service Instruction"),
    "AD": ("airworthiness directive", "validated", [], "FAA Airworthiness Directive"),
    "ADEL": ("Adel clamp", "validated", [], "Standard aircraft cushioned loop clamp manufacturer"),
    "DZUS": ("Dzus fastener", "validated", [], "Quick-release quarter-turn fastener brand"),
}


class AbbreviationSuggestion(BaseModel):
    term: str
    proposed_expansion: str
    is_ambiguous: bool
    alternative_meanings: List[str] = Field(default_factory=list)
    rationale: str


def suggest_expansion_from_llm(
    term: str,
    excerpts: List[str],
    api_key: Optional[str] = None
) -> Optional[AbbreviationSuggestion]:
    """
    Asks Gemini to propose an expansion grounded strictly in the provided excerpts.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        examples_str = "\n".join(f"- {e}" for e in excerpts[:6])

        prompt = f"""You are an aviation maintenance expert reviewing aircraft logbook records.
Analyze the abbreviation candidate "{term}" based ONLY on these genuine logbook excerpts:
{examples_str}

Rules:
1. Ground your proposal strictly in the provided excerpts.
2. If the excerpts do not provide enough context to be certain, return "UNKNOWN".
3. If the abbreviation has multiple valid, conflicting meanings in these contexts, set is_ambiguous to true and list them.
4. Do not invent an expansion not supported by aviation engineering practice.
"""

        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AbbreviationSuggestion,
                temperature=0.0
            )
        )
        if response.text:
            data = json.loads(response.text)
            return AbbreviationSuggestion(**data)
    except Exception as e:
        print(f"[WARNING] Gemini expansion call for '{term}' failed: {e}")
        return None


def resolve_candidate_expansion(
    term: str,
    excerpts: List[str],
    frequency: int,
    api_key: Optional[str] = None,
    allow_llm: bool = False
) -> GlossaryEntry:
    """
    Proposes an expansion using domain curation first.
    If allow_llm is True and an API key is available, attempts Gemini proposal for unknown terms.
    Otherwise defaults safely to UNKNOWN / needs_review status.
    """
    # 1. Check verified domain curation first
    if term in _KNOWN_AVIATION_MAPPINGS:
        exp, status, alts, notes = _KNOWN_AVIATION_MAPPINGS[term]
        return GlossaryEntry(
            term=term,
            expansion=exp,
            status=status,
            context_examples=excerpts[:5],
            alternative_meanings=alts,
            frequency=frequency,
            notes=notes,
            source="aviation-domain-curation"
        )

    # 2. Try Gemini API proposal only if explicitly requested
    if allow_llm:
        llm_suggestion = suggest_expansion_from_llm(term, excerpts, api_key)
        if llm_suggestion:
            status: ValidationStatus = "needs_review"
            if llm_suggestion.proposed_expansion.upper() == "UNKNOWN":
                status = "unknown"
            elif llm_suggestion.is_ambiguous:
                status = "ambiguous"

            return GlossaryEntry(
                term=term,
                expansion=llm_suggestion.proposed_expansion,
                status=status,
                context_examples=excerpts[:5],
                alternative_meanings=llm_suggestion.alternative_meanings,
                frequency=frequency,
                notes=llm_suggestion.rationale,
                source="gemini-proposal"
            )
            status = "unknown"
        elif llm_suggestion.is_ambiguous:
            status = "ambiguous"

        return GlossaryEntry(
            term=term,
            expansion=llm_suggestion.proposed_expansion,
            status=status,
            context_examples=excerpts[:5],
            alternative_meanings=llm_suggestion.alternative_meanings,
            frequency=frequency,
            notes=llm_suggestion.rationale,
            source="gemini-proposal"
        )

    # 3. Default to UNKNOWN / needs_review rather than inventing definitions
    return GlossaryEntry(
        term=term,
        expansion="UNKNOWN",
        status="needs_review",
        context_examples=excerpts[:5],
        alternative_meanings=[],
        frequency=frequency,
        notes="Candidate discovered from corpus frequency; awaiting human or LLM review.",
        source="auto-discovery"
    )
