"""
Deterministic Aspect Coverage Checker.
Evaluates whether top retrieved records corroborate each extracted query aspect
(component, symptom, condition, requested_info) using normalized text,
validated glossary expansions, token overlap, and fuzzy lexical matching.
Enforces exact technical identifier constraints (cylinders, RPM, part numbers, orientation).
Strictly counts distinct duplicate clusters, not raw record counts.
"""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Set, Tuple, Literal
from pydantic import BaseModel, Field

from src.retrieval.models import RetrievalResult
from src.agent.aspect_extractor import QueryAspects
from src.abbreviation.models import Glossary
from src.abbreviation.expander import expand_text_with_glossary
from src.normalizer import normalize_text


AspectStatus = Literal["supported", "unsupported", "contradicted", "unknown", "not_applicable"]

# Exact Technical Identifier Patterns
_CYLINDER_PATTERN = re.compile(
    r"(?:#\s*(\d{1,2})|\b(?:cyl|cylinder)s?[\s\.\-#]*(\d{1,2})\b)",
    re.IGNORECASE
)
_RPM_PATTERN = re.compile(r"\b(\d{3,5})\s*(?:rpm|revolutions\s+per\s+minute)\b", re.IGNORECASE)
_PART_NUM_PATTERN = re.compile(r"\b([A-Za-z]{1,4}\d{1,4}-\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)
_LH_PATTERN = re.compile(r"\b(l/?h|left(?:-hand)?)\b", re.IGNORECASE)
_RH_PATTERN = re.compile(r"\b(r/?h|right(?:-hand)?)\b", re.IGNORECASE)

_WORD_PATTERN = re.compile(r"\b[A-Za-z0-9#/'\-]+\b")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "on", "in", "at", "for", "with", "of", "to",
    "is", "was", "are", "were", "be", "been", "by", "all", "its", "it"
}

_TOKEN_SYNONYMS = {
    "cylinder": "cyl",
    "cylinders": "cyl",
    "magneto": "mag",
    "magnetos": "mag",
    "engine": "eng",
    "engines": "eng",
    "sparkplug": "plug",
    "sparkplugs": "plug",
    "spark-plug": "plug",
    "spark-plugs": "plug",
    "fouling": "fouled",
    "foul": "fouled"
}


class AspectDetail(BaseModel):
    """
    Detailed evidence tracking for an individual query aspect.
    """
    aspect_name: str
    extracted_value: Optional[str] = None
    status: AspectStatus = Field(
        ...,
        description="'supported', 'unsupported', 'contradicted', 'unknown', or 'not_applicable'"
    )
    supporting_record_ids: List[str] = Field(default_factory=list)
    supporting_clusters: int = 0
    contradicting_record_ids: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


class AspectCoverageResult(BaseModel):
    """
    Structured outcome of aspect-level evidence coverage analysis.
    """
    aspect_coverage: Dict[str, bool] = Field(
        ...,
        description="Mapping of aspect name -> boolean support status (True if supported or N/A, False if unsupported)"
    )
    aspect_details: Dict[str, AspectDetail] = Field(
        default_factory=dict,
        description="Detailed per-aspect status and supporting evidence tracking"
    )
    unsupported_aspects: List[str] = Field(
        ...,
        description="List of aspects that were present in query but lack evidence in retrieved clusters"
    )
    contradicted_aspects: List[str] = Field(
        default_factory=list,
        description="List of aspects that have conflicting or contradicted records"
    )
    cluster_support_counts: Dict[str, int] = Field(
        ...,
        description="Number of distinct duplicate clusters supporting each aspect"
    )
    total_distinct_clusters: int = Field(
        ...,
        description="Total number of distinct duplicate clusters evaluated"
    )
    aspect_support_ratio: float = Field(
        ...,
        description="Fraction of present query aspects supported by at least min_supporting_clusters"
    )
    reason: str = Field(
        ...,
        description="Deterministic explanatory summary of aspect coverage"
    )


def extract_cylinders(text: str) -> Set[str]:
    """
    Extracts cylinder number designations e.g. 'CYL 2', 'CYL #2', '#2', 'CYL. 2', 'cylinder 12' -> {'2'}, {'12'}.
    Normalizes numeric tokens so #02 and CYL 2 both resolve to '2'.
    """
    if not text:
        return set()
    cyls = set()
    for m in _CYLINDER_PATTERN.finditer(text):
        c1, c2 = m.groups()
        cyl = c1 or c2
        if cyl:
            try:
                cyls.add(str(int(cyl)))
            except ValueError:
                pass
    return cyls


def extract_rpm_values(text: str) -> Set[str]:
    """Extracts explicit RPM values, e.g. '2700 RPM' -> {'2700'}."""
    if not text:
        return set()
    return set(_RPM_PATTERN.findall(text))


def extract_part_numbers(text: str) -> Set[str]:
    """Extracts part/dash numbers, e.g. 'AN4-12A' -> {'AN4-12A'}."""
    if not text:
        return set()
    return {p.upper() for p in _PART_NUM_PATTERN.findall(text)}


def extract_orientations(text: str) -> Set[str]:
    """Extracts left/right orientation."""
    if not text:
        return set()
    ors = set()
    if _LH_PATTERN.search(text):
        ors.add("LH")
    if _RH_PATTERN.search(text):
        ors.add("RH")
    return ors


class TechnicalConstraints:
    """
    Deterministic technical constraint extraction and satisfaction validator.
    Enforces exact boundaries for cylinder numbers, RPM ratings, part numbers, and orientation.
    """
    def __init__(self, text: str):
        self.cylinders = extract_cylinders(text)
        self.rpms = extract_rpm_values(text)
        self.part_numbers = extract_part_numbers(text)
        self.orientations = extract_orientations(text)

    def is_empty(self) -> bool:
        return not (self.cylinders or self.rpms or self.part_numbers or self.orientations)

    def satisfies(self, record_text: str) -> Tuple[bool, Optional[str]]:
        """
        Validates that record_text satisfies any exact constraints present in self,
        and does not contradict them.
        """
        rec_cyls = extract_cylinders(record_text)
        rec_rpms = extract_rpm_values(record_text)
        rec_parts = extract_part_numbers(record_text)
        rec_ors = extract_orientations(record_text)

        # 1. Cylinder constraints: if aspect/query specifically constrains cylinder(s),
        # record MUST contain the requested cylinder(s) and NOT only conflicting ones.
        if self.cylinders:
            if not rec_cyls:
                return False, f"Record does not specify requested cylinder(s) {sorted(self.cylinders)}"
            if not (self.cylinders & rec_cyls):
                return False, f"Cylinder mismatch: requested {sorted(self.cylinders)}, found {sorted(rec_cyls)}"

        # 2. RPM constraints: exact numerical match required
        if self.rpms:
            if not rec_rpms:
                return False, f"Record does not mention requested RPM {sorted(self.rpms)}"
            if not (self.rpms & rec_rpms):
                return False, f"RPM mismatch: requested {sorted(self.rpms)}, found {sorted(rec_rpms)}"

        # 3. Part number constraints
        if self.part_numbers:
            if not rec_parts:
                return False, f"Record does not mention requested part {sorted(self.part_numbers)}"
            if not (self.part_numbers & rec_parts):
                return False, f"Part number mismatch: requested {sorted(self.part_numbers)}, found {sorted(rec_parts)}"

        # 4. Left/Right orientation constraints
        if self.orientations:
            if rec_ors and not (self.orientations & rec_ors):
                return False, f"Orientation mismatch: requested {sorted(self.orientations)}, found {sorted(rec_ors)}"

        return True, None


def _norm_token(t: str) -> str:
    tl = t.lower().strip()
    return _TOKEN_SYNONYMS.get(tl, tl)


def clean_tokens(text: str) -> List[str]:
    """Tokenizes text into lowercase non-stopword tokens with synonym normalization."""
    if not text:
        return []
    words = [_norm_token(t) for t in _WORD_PATTERN.findall(text)]
    return [w for w in words if w not in _STOPWORDS and (len(w) > 1 or w.isdigit())]


def match_aspect_in_text(
    aspect_val: str,
    record_text: str,
    fuzzy_threshold: float = 0.80
) -> bool:
    """
    Deterministically checks if an aspect value is supported by a record's narrative text.
    Combines exact technical constraints (cylinders, RPM, part numbers, orientation),
    exact substring matching, and non-numeric token fuzzy matching.
    """
    if not aspect_val or not record_text:
        return False

    aspect_lower = aspect_val.lower().strip()
    rec_lower = record_text.lower().strip()

    # 1. Technical constraints check
    constraints = TechnicalConstraints(aspect_val)
    if not constraints.is_empty():
        sat, _ = constraints.satisfies(record_text)
        if not sat:
            return False

        # If the aspect value only specified technical identifiers (e.g. '#2', 'CYL 2', '2700 RPM')
        # and those technical constraints are fully satisfied by the record:
        aspect_non_id_tokens = clean_tokens(
            re.sub(_CYLINDER_PATTERN, "", re.sub(_RPM_PATTERN, "", re.sub(_PART_NUM_PATTERN, "", aspect_lower)))
        )
        if not aspect_non_id_tokens:
            return True

    # 2. Exact substring match
    if aspect_lower in rec_lower:
        return True

    # 3. Token-level matching
    aspect_tokens = clean_tokens(aspect_lower)
    if not aspect_tokens:
        return False

    rec_tokens = clean_tokens(rec_lower)
    if not rec_tokens:
        return False

    matched_tokens_count = 0
    for a_tok in aspect_tokens:
        # Numeric tokens must match EXACTLY, never fuzzy
        if any(c.isdigit() for c in a_tok):
            if a_tok in rec_tokens:
                matched_tokens_count += 1
            continue

        # Direct token match
        if a_tok in rec_tokens:
            matched_tokens_count += 1
            continue

        # Fuzzy match allowed ONLY for non-numeric words (e.g. "intak gasget" -> "intake gasket")
        for r_tok in rec_tokens:
            if not any(c.isdigit() for c in r_tok):
                if abs(len(a_tok) - len(r_tok)) <= 2:
                    if SequenceMatcher(None, a_tok, r_tok).ratio() >= fuzzy_threshold:
                        matched_tokens_count += 1
                        break

    if len(aspect_tokens) == 1:
        return matched_tokens_count == 1

    match_ratio = matched_tokens_count / len(aspect_tokens)
    return match_ratio >= 0.50


def check_aspect_coverage(
    query: str,
    aspects: QueryAspects,
    deduped_results: List[RetrievalResult],
    clusters: Dict[str, List[str]],
    glossary: Optional[Glossary] = None,
    min_supporting_clusters: int = 1,
    fuzzy_threshold: float = 0.80
) -> AspectCoverageResult:
    """
    Evaluates evidence coverage across distinct duplicate clusters for each extracted aspect.
    Tracks explicit statuses (supported, unsupported, contradicted, not_applicable)
    and supporting record/unit IDs.

    Args:
        query: Raw query string.
        aspects: Extracted QueryAspects.
        deduped_results: Distinct cluster representatives.
        clusters: Mapping from representative record_id -> list of cluster member record_ids.
        glossary: Optional Glossary for abbreviation expansions.
        min_supporting_clusters: Minimum distinct clusters required to consider an aspect supported.
        fuzzy_threshold: Minimum string similarity ratio for fuzzy token matching.

    Returns:
        AspectCoverageResult with per-aspect coverage, detailed aspect statuses, and explanatory reason.
    """
    aspect_fields = ["component", "symptom", "condition", "requested_info"]
    present_aspects = {f: getattr(aspects, f) for f in aspect_fields if getattr(aspects, f)}

    # If no aspects were extracted, treat as completely unsupported unless query is trivial
    if not present_aspects:
        empty_details = {
            f: AspectDetail(
                aspect_name=f,
                extracted_value=None,
                status="not_applicable",
                supporting_record_ids=[],
                supporting_clusters=0,
                contradicting_record_ids=[],
                reason="Not specified in query."
            )
            for f in aspect_fields
        }
        return AspectCoverageResult(
            aspect_coverage={f: True for f in aspect_fields},
            aspect_details=empty_details,
            unsupported_aspects=[],
            contradicted_aspects=[],
            cluster_support_counts={f: 0 for f in aspect_fields},
            total_distinct_clusters=len(deduped_results),
            aspect_support_ratio=1.0,
            reason="No specific aspects identified in query."
        )

    # Pre-render expanded narrative texts for distinct cluster representatives
    cluster_texts: List[Tuple[RetrievalResult, str]] = []
    for rep in deduped_results:
        raw_combined = f"{rep.problem_normalized} {rep.action_normalized}"
        if glossary is not None:
            expanded = expand_text_with_glossary(raw_combined, glossary)
        else:
            expanded = raw_combined
        cluster_texts.append((rep, expanded))

    total_clusters = len(deduped_results)
    aspect_coverage: Dict[str, bool] = {}
    aspect_details: Dict[str, AspectDetail] = {}
    cluster_support_counts: Dict[str, int] = {}
    unsupported_aspects: List[str] = []
    contradicted_aspects: List[str] = []

    for f in aspect_fields:
        val = getattr(aspects, f)
        if not val:
            # Aspect not present in query: not applicable
            aspect_coverage[f] = True
            cluster_support_counts[f] = 0
            aspect_details[f] = AspectDetail(
                aspect_name=f,
                extracted_value=None,
                status="not_applicable",
                supporting_record_ids=[],
                supporting_clusters=0,
                contradicting_record_ids=[],
                reason="Not present in query."
            )
            continue

        # Check technical constraints of the aspect
        constraints = TechnicalConstraints(val)

        supporting_clusters = 0
        supporting_record_ids: List[str] = []
        contradicting_record_ids: List[str] = []

        for rep, text in cluster_texts:
            cluster_members = clusters.get(rep.record_id, [rep.record_id])
            if match_aspect_in_text(val, text, fuzzy_threshold=fuzzy_threshold):
                supporting_clusters += 1
                supporting_record_ids.extend(cluster_members)
            else:
                # Check if it was rejected due to an explicit constraint contradiction
                if not constraints.is_empty():
                    sat, _ = constraints.satisfies(text)
                    if not sat:
                        contradicting_record_ids.extend(cluster_members)

        cluster_support_counts[f] = supporting_clusters
        is_supported = (supporting_clusters >= min_supporting_clusters)
        aspect_coverage[f] = is_supported

        if is_supported:
            status: AspectStatus = "supported"
            reason_txt = f"Supported by {supporting_clusters} cluster(s) (Record IDs: {', '.join(supporting_record_ids[:5])})."
        elif contradicting_record_ids:
            status = "contradicted"
            contradicted_aspects.append(f)
            unsupported_aspects.append(f)
            reason_txt = f"Contradicted by {len(contradicting_record_ids)} record(s) with conflicting technical identifiers."
        else:
            status = "unsupported"
            unsupported_aspects.append(f)
            reason_txt = f"No retrieved records corroborate '{val}'."

        aspect_details[f] = AspectDetail(
            aspect_name=f,
            extracted_value=val,
            status=status,
            supporting_record_ids=supporting_record_ids,
            supporting_clusters=supporting_clusters,
            contradicting_record_ids=contradicting_record_ids,
            reason=reason_txt
        )

    supported_count = len(present_aspects) - len(unsupported_aspects)
    aspect_support_ratio = supported_count / len(present_aspects) if present_aspects else 1.0

    # Compose deterministic reason string
    supported_descs = [
        f"{f} ('{getattr(aspects, f)}')" for f in present_aspects if f not in unsupported_aspects
    ]
    if not unsupported_aspects:
        reason = (
            f"All query aspects ({', '.join(supported_descs)}) are supported across "
            f"{total_clusters} distinct record cluster(s)."
        )
    elif contradicted_aspects:
        contra_descs = [f"{f} ('{getattr(aspects, f)}')" for f in contradicted_aspects]
        reason = (
            f"Constraint contradiction: retrieved records explicitly contradict {', '.join(contra_descs)}."
        )
    else:
        unsupported_descs = [
            f"{f} ('{getattr(aspects, f)}')" for f in unsupported_aspects
        ]
        if supported_descs:
            reason = (
                f"Partially supported: {', '.join(supported_descs)} supported, but "
                f"retrieved records lack evidence for {', '.join(unsupported_descs)}."
            )
        else:
            reason = (
                f"Insufficient evidence: retrieved records do not support {', '.join(unsupported_descs)}."
            )

    return AspectCoverageResult(
        aspect_coverage=aspect_coverage,
        aspect_details=aspect_details,
        unsupported_aspects=unsupported_aspects,
        contradicted_aspects=contradicted_aspects,
        cluster_support_counts=cluster_support_counts,
        total_distinct_clusters=total_clusters,
        aspect_support_ratio=round(aspect_support_ratio, 4),
        reason=reason
    )
