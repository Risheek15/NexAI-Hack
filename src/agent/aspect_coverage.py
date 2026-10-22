"""
Deterministic Aspect Coverage Checker.
Evaluates whether top retrieved records corroborate each extracted query aspect
(component, symptom, condition, requested_info) using normalized text,
validated glossary expansions, token overlap, and fuzzy lexical matching.
Strictly counts distinct duplicate clusters, not raw record counts.
"""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.retrieval.models import RetrievalResult
from src.agent.aspect_extractor import QueryAspects
from src.abbreviation.models import Glossary
from src.abbreviation.expander import expand_text_with_glossary
from src.normalizer import normalize_text


_CYLINDER_PATTERN = re.compile(r"#\s*(\d+)")
_WORD_PATTERN = re.compile(r"\b[A-Za-z0-9#/'\-]+\b")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "on", "in", "at", "for", "with", "of", "to",
    "is", "was", "are", "were", "be", "been", "by", "all", "its", "it"
}


class AspectCoverageResult(BaseModel):
    """
    Structured outcome of aspect-level evidence coverage analysis.
    """
    aspect_coverage: Dict[str, bool] = Field(
        ...,
        description="Mapping of aspect name -> boolean support status (True if supported or N/A, False if unsupported)"
    )
    unsupported_aspects: List[str] = Field(
        ...,
        description="List of aspects that were present in query but lack evidence in retrieved clusters"
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
    """Extracts cylinder number designations, e.g. #1, #2, #3, #4."""
    if not text:
        return set()
    return set(_CYLINDER_PATTERN.findall(text))


def clean_tokens(text: str) -> List[str]:
    """Tokenizes text into lowercase non-stopword tokens."""
    if not text:
        return []
    words = [t.lower() for t in _WORD_PATTERN.findall(text)]
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def match_aspect_in_text(
    aspect_val: str,
    record_text: str,
    fuzzy_threshold: float = 0.80
) -> bool:
    """
    Deterministically checks if an aspect value is supported by a record's narrative text.
    Combines cylinder constraint checks, exact substring matching, and fuzzy token matching.
    """
    if not aspect_val or not record_text:
        return False

    aspect_lower = aspect_val.lower().strip()
    rec_lower = record_text.lower().strip()

    # 1. Cylinder constraint: if aspect explicitly names cylinder(s) (e.g. #2),
    # the record must match that cylinder or not specify a conflicting one.
    aspect_cyls = extract_cylinders(aspect_lower)
    rec_cyls = extract_cylinders(rec_lower)
    if aspect_cyls and rec_cyls and not (aspect_cyls & rec_cyls):
        # Explicit conflicting cylinder designation (e.g. #2 requested but record is exclusively #3)
        return False

    # 2. Exact substring match
    if aspect_lower in rec_lower:
        return True

    # 3. Token-level matching with fuzzy allowance for spelling noise (e.g. "intak gasget" -> "intake gasket")
    aspect_tokens = clean_tokens(aspect_lower)
    if not aspect_tokens:
        return False

    rec_tokens = clean_tokens(rec_lower)
    if not rec_tokens:
        return False

    matched_tokens_count = 0
    for a_tok in aspect_tokens:
        # Check if cylinder number token
        if a_tok.startswith("#"):
            if a_tok in rec_tokens or any(a_tok[1:] in r_tok for r_tok in rec_tokens):
                matched_tokens_count += 1
                continue

        matched = False
        # Direct word match
        if a_tok in rec_tokens:
            matched = True
        else:
            # Fuzzy match (SequenceMatcher)
            for r_tok in rec_tokens:
                # Fast length prune
                if abs(len(a_tok) - len(r_tok)) <= 2:
                    if SequenceMatcher(None, a_tok, r_tok).ratio() >= fuzzy_threshold:
                        matched = True
                        break

        if matched:
            matched_tokens_count += 1

    # For single-word aspects: must match
    if len(aspect_tokens) == 1:
        return matched_tokens_count == 1

    # For multi-word aspects: require at least 50% of tokens to match
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

    Args:
        query: Raw query string.
        aspects: Extracted QueryAspects.
        deduped_results: Distinct cluster representatives.
        clusters: Mapping from representative record_id -> list of cluster member record_ids.
        glossary: Optional Glossary for abbreviation expansions.
        min_supporting_clusters: Minimum distinct clusters required to consider an aspect supported.
        fuzzy_threshold: Minimum string similarity ratio for fuzzy token matching.

    Returns:
        AspectCoverageResult with per-aspect coverage, unsupported aspects, and explanatory reason.
    """
    aspect_fields = ["component", "symptom", "condition", "requested_info"]
    present_aspects = {f: getattr(aspects, f) for f in aspect_fields if getattr(aspects, f)}

    # If no aspects were extracted, treat as completely unsupported unless query is trivial
    if not present_aspects:
        return AspectCoverageResult(
            aspect_coverage={f: True for f in aspect_fields},
            unsupported_aspects=[],
            cluster_support_counts={f: 0 for f in aspect_fields},
            total_distinct_clusters=len(deduped_results),
            aspect_support_ratio=1.0,
            reason="No specific aspects identified in query."
        )

    # Pre-render expanded narrative texts for distinct cluster representatives
    cluster_texts: List[str] = []
    for rep in deduped_results:
        raw_combined = f"{rep.problem_normalized} {rep.action_normalized}"
        if glossary is not None:
            expanded = expand_text_with_glossary(raw_combined, glossary)
        else:
            expanded = raw_combined
        cluster_texts.append(expanded)

    total_clusters = len(deduped_results)
    aspect_coverage: Dict[str, bool] = {}
    cluster_support_counts: Dict[str, int] = {}
    unsupported_aspects: List[str] = []

    for f in aspect_fields:
        val = getattr(aspects, f)
        if not val:
            # Aspect not present in query: considered satisfied / not missing
            aspect_coverage[f] = True
            cluster_support_counts[f] = 0
            continue

        # Check coverage across distinct cluster texts
        supporting_clusters = 0
        for text in cluster_texts:
            if match_aspect_in_text(val, text, fuzzy_threshold=fuzzy_threshold):
                supporting_clusters += 1

        cluster_support_counts[f] = supporting_clusters
        is_supported = (supporting_clusters >= min_supporting_clusters)
        aspect_coverage[f] = is_supported

        if not is_supported:
            unsupported_aspects.append(f)

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
        unsupported_aspects=unsupported_aspects,
        cluster_support_counts=cluster_support_counts,
        total_distinct_clusters=total_clusters,
        aspect_support_ratio=round(aspect_support_ratio, 4),
        reason=reason
    )
