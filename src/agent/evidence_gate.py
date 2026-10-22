"""
Deterministic Evidence Assessment & Near-Duplicate Clustering Layer.
Evaluates retrieval evidence quality without relying exclusively on LLM self-reporting
or arbitrary rank-fusion score thresholds.
Supports four-aspect evidence coverage (Component, Symptom, Condition, Requested Information)
as the default production gate, while allowing configurable fallback.
"""

import os
import re
from typing import List, Tuple, Dict, Set, Literal, Optional
from pydantic import BaseModel, Field

from src.retrieval.models import RetrievalResult
from src.agent.aspect_extractor import QueryAspects, extract_aspects
from src.agent.aspect_coverage import check_aspect_coverage, AspectCoverageResult, AspectDetail
from src.abbreviation.models import Glossary


EvidenceStatus = Literal["sufficient", "insufficient", "needs_clarification"]

# Configuration flag for Aspect Gate (default True in production, can be toggled by environment)
ENABLE_ASPECT_GATE = os.environ.get("ENABLE_ASPECT_GATE", "true").lower() == "true"

_WORD_PATTERN = re.compile(r"\b[A-Za-z0-9#/'\-]+\b")
_CYLINDER_PATTERN = re.compile(
    r"(?:#\s*(\d{1,2})|\b(?:cyl|cylinder)s?[\s\.\-#]*(\d{1,2})\b)",
    re.IGNORECASE
)

# Generic single-word tokens that lack subsystem context
_GENERIC_VAGUE_WORDS = {
    "LEAK", "LEAKS", "LEAKING", "BROKEN", "CRACK", "CRACKED", "LOOSE", "WORN",
    "REMOVED", "REPLACED", "TIGHTENED", "REPAIRED", "PROBLEM", "DEFECT", "BAD",
    "ENGINE", "AIRCRAFT", "CHECK", "FAILURE", "DAMAGE", "INSPECT", "SEAL"
}

# Aircraft systems and keywords outside the domain of reciprocating engine maintenance logbooks
_OUT_OF_DOMAIN_TERMS = {
    "autopilot", "boeing", "airbus", "777", "737", "787", "a320", "a350",
    "thrust reverser", "thrust reversers", "cabin pressure", "evacuation slide",
    "oxygen mask", "lavatory", "fmc", "fms", "tcas", "gpws", "fly-by-wire"
}


class AspectGateConfig(BaseModel):
    """
    Configurable parameters for aspect-level evidence coverage gating.
    """
    enabled: bool = Field(
        default_factory=lambda: os.environ.get("ENABLE_ASPECT_GATE", "true").lower() == "true",
        description="Whether aspect-level evidence coverage gating is active"
    )
    min_supporting_clusters: int = Field(
        default=1,
        description="Minimum distinct duplicate clusters required to support an aspect"
    )
    fuzzy_similarity_threshold: float = Field(
        default=0.80,
        description="Fuzzy string matching ratio for noisy/misspelled terms"
    )
    critical_aspects: List[str] = Field(
        default_factory=lambda: ["component", "symptom"],
        description="Aspects that must be supported to declare sufficiency"
    )
    abstain_on_unsupported_critical: bool = Field(
        default=True,
        description="Whether to mark status as insufficient if any critical aspect lacks support"
    )


class EvidenceAssessment(BaseModel):
    """
    Structured outcome of the deterministic evidence evaluation.
    Gains per-aspect coverage, unsupported aspects, and explanatory reason.
    """
    status: EvidenceStatus = Field(..., description="'sufficient', 'insufficient', or 'needs_clarification'")
    reason: str = Field(..., description="Deterministic rationale for the assigned evidence status")
    unique_evidence_count: int = Field(..., description="Count of distinct, non-duplicate records found")
    duplicate_clusters_count: int = Field(..., description="Number of duplicate or near-duplicate clusters identified")
    top_semantic_score: float = Field(default=0.0, description="Highest dense semantic similarity score in results")
    lexical_match_ratio: float = Field(default=0.0, description="Fraction of non-stop query terms matched in results")
    # Aspect-level evidence coverage fields
    aspect_coverage: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-aspect supported/unsupported boolean mapping"
    )
    aspect_details: Dict[str, AspectDetail] = Field(
        default_factory=dict,
        description="Detailed per-aspect status and supporting evidence tracking"
    )
    unsupported_aspects: List[str] = Field(
        default_factory=list,
        description="List of aspects present in query that lack supporting evidence"
    )
    contradicted_aspects: List[str] = Field(
        default_factory=list,
        description="List of aspects that have conflicting or contradicted records"
    )
    aspects: Optional[Dict[str, Optional[str]]] = Field(
        default=None,
        description="Extracted structured query aspects"
    )
    aspect_support_ratio: float = Field(
        default=1.0,
        description="Fraction of present query aspects supported by retrieved clusters"
    )


def get_token_set(text: str) -> Set[str]:
    """Extracts lowercase alphanumeric tokens from text."""
    if not text:
        return set()
    return {t.lower() for t in _WORD_PATTERN.findall(text)}


def extract_cylinders(text: str) -> Set[str]:
    """Extracts cylinder number designations e.g. #1, #2, #3, #4, CYL 2, cylinder 12."""
    if not text:
        return set()
    cyls = set()
    for m in _CYLINDER_PATTERN.finditer(text):
        c1, c2 = m.groups()
        c = c1 or c2
        if c:
            try:
                cyls.add(str(int(c)))
            except ValueError:
                pass
    return cyls


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Calculates Jaccard similarity over word tokens."""
    s1 = get_token_set(text1)
    s2 = get_token_set(text2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def cluster_near_duplicates(
    results: List[RetrievalResult],
    similarity_threshold: float = 0.85
) -> Tuple[List[RetrievalResult], Dict[str, List[str]]]:
    """
    Clusters near-identical logbook records (e.g. repeated entries for the same leak)
    so repetitive entries do not falsely inflate independent evidence counts.

    Returns:
        deduplicated_results: List of representative RetrievalResult objects (one per cluster).
        clusters: Mapping from representative record_id -> list of clustered record_ids.
    """
    if not results:
        return [], {}

    clusters: Dict[str, List[str]] = {}
    representatives: List[RetrievalResult] = []

    for res in results:
        combined_text = f"{res.problem_normalized} {res.action_normalized}"
        res_cyls = extract_cylinders(combined_text)
        matched_cluster_id = None

        for rep in representatives:
            rep_text = f"{rep.problem_normalized} {rep.action_normalized}"
            rep_cyls = extract_cylinders(rep_text)

            # If both records explicitly specify different cylinder numbers, they are distinct maintenance events
            if res_cyls and rep_cyls and res_cyls != rep_cyls:
                continue

            if compute_token_jaccard(combined_text, rep_text) >= similarity_threshold:
                matched_cluster_id = rep.record_id
                break

        if matched_cluster_id is not None:
            clusters[matched_cluster_id].append(res.record_id)
        else:
            representatives.append(res)
            clusters[res.record_id] = [res.record_id]

    return representatives, clusters


def assess_retrieval_evidence(
    query: str,
    results: List[RetrievalResult],
    enable_aspect_gate: Optional[bool] = None,
    config: Optional[AspectGateConfig] = None,
    glossary: Optional[Glossary] = None,
    aspects: Optional[QueryAspects] = None
) -> Tuple[EvidenceAssessment, List[RetrievalResult], Dict[str, List[str]]]:
    """
    Deterministically evaluates whether retrieved evidence is sufficient, insufficient,
    or needs clarification.

    When enable_aspect_gate is True, performs fine-grained aspect extraction and
    verifies coverage of component, symptom, condition, and requested information
    across distinct duplicate clusters.
    """
    clean_query = query.strip()
    if not clean_query:
        assessment = EvidenceAssessment(
            status="needs_clarification",
            reason="Query is empty or contains only whitespace.",
            unique_evidence_count=0,
            duplicate_clusters_count=0,
            top_semantic_score=0.0,
            lexical_match_ratio=0.0
        )
        return assessment, [], {}

    query_tokens = [t for t in _WORD_PATTERN.findall(clean_query.upper()) if len(t) > 1]

    # Check for explicitly out-of-domain systems (commercial jetliners, transport systems)
    query_lower = clean_query.lower()
    for term in _OUT_OF_DOMAIN_TERMS:
        if term in query_lower:
            assessment = EvidenceAssessment(
                status="insufficient",
                reason=f"Available historical evidence shows that '{term}' is out-of-domain for this reciprocating engine logbook corpus.",
                unique_evidence_count=0,
                duplicate_clusters_count=0,
                top_semantic_score=0.0,
                lexical_match_ratio=0.0,
                unsupported_aspects=["component"]
            )
            return assessment, [], {}

    # Check for ambiguous, single-token generic queries
    if len(query_tokens) == 1 and query_tokens[0] in _GENERIC_VAGUE_WORDS:
        assessment = EvidenceAssessment(
            status="needs_clarification",
            reason=f"The query '{clean_query}' is too generic. Please specify the component, cylinder number, or symptom.",
            unique_evidence_count=0,
            duplicate_clusters_count=0,
            top_semantic_score=0.0,
            lexical_match_ratio=0.0
        )
        return assessment, [], {}

    if not results:
        assessment = EvidenceAssessment(
            status="insufficient",
            reason="No historical logbook records matched the query terms or concepts.",
            unique_evidence_count=0,
            duplicate_clusters_count=0,
            top_semantic_score=0.0,
            lexical_match_ratio=0.0,
            unsupported_aspects=["component", "symptom"]
        )
        return assessment, [], {}

    # Cluster duplicates
    deduped_results, clusters = cluster_near_duplicates(results, similarity_threshold=0.85)
    unique_count = len(deduped_results)
    duplicate_clusters_count = sum(1 for members in clusters.values() if len(members) > 1)

    # Calculate lexical overlap with query tokens
    all_retrieved_tokens: Set[str] = set()
    for r in deduped_results:
        all_retrieved_tokens.update(get_token_set(r.combined_raw.upper()))

    matched_query_tokens = sum(1 for qt in query_tokens if qt.lower() in all_retrieved_tokens)
    lexical_ratio = matched_query_tokens / len(query_tokens) if query_tokens else 0.0

    # Extract semantic score if available from metadata
    top_semantic = 0.0
    for r in results:
        sim = r.metadata.get("cosine_similarity") or (r.score if r.mode == "semantic" else 0.0)
        if sim > top_semantic:
            top_semantic = float(sim)

    # Determine whether aspect-level gate is active
    gate_cfg = config or AspectGateConfig()
    use_aspects = enable_aspect_gate if enable_aspect_gate is not None else gate_cfg.enabled

    # --- Aspect-Level Evidence Coverage Pathway ---
    if use_aspects:
        extracted = aspects if aspects is not None else extract_aspects(clean_query)
        cov_res = check_aspect_coverage(
            query=clean_query,
            aspects=extracted,
            deduped_results=deduped_results,
            clusters=clusters,
            glossary=glossary,
            min_supporting_clusters=gate_cfg.min_supporting_clusters,
            fuzzy_threshold=gate_cfg.fuzzy_similarity_threshold
        )

        aspect_dict = extracted.model_dump()
        unsupported = cov_res.unsupported_aspects
        contradicted = cov_res.contradicted_aspects

        # Ambiguous check: if query has no component and only a single vague symptom
        if not extracted.component and extracted.symptom and extracted.symptom.upper() in _GENERIC_VAGUE_WORDS:
            assessment = EvidenceAssessment(
                status="needs_clarification",
                reason=f"The query '{clean_query}' lacks component context. Please specify which component or cylinder is experiencing {extracted.symptom}.",
                unique_evidence_count=unique_count,
                duplicate_clusters_count=duplicate_clusters_count,
                top_semantic_score=round(top_semantic, 4),
                lexical_match_ratio=round(lexical_ratio, 4),
                aspect_coverage=cov_res.aspect_coverage,
                aspect_details=cov_res.aspect_details,
                unsupported_aspects=unsupported,
                contradicted_aspects=contradicted,
                aspects=aspect_dict,
                aspect_support_ratio=cov_res.aspect_support_ratio
            )
            return assessment, deduped_results, clusters

        # Contradicted aspects check (e.g. cylinder or orientation constraint violation)
        if contradicted:
            assessment = EvidenceAssessment(
                status="insufficient",
                reason=cov_res.reason,
                unique_evidence_count=unique_count,
                duplicate_clusters_count=duplicate_clusters_count,
                top_semantic_score=round(top_semantic, 4),
                lexical_match_ratio=round(lexical_ratio, 4),
                aspect_coverage=cov_res.aspect_coverage,
                aspect_details=cov_res.aspect_details,
                unsupported_aspects=unsupported,
                contradicted_aspects=contradicted,
                aspects=aspect_dict,
                aspect_support_ratio=cov_res.aspect_support_ratio
            )
            return assessment, deduped_results, clusters

        # Check critical aspects (component, symptom)
        has_unsupported_critical = any(
            crit in unsupported for crit in gate_cfg.critical_aspects if getattr(extracted, crit)
        )

        if gate_cfg.abstain_on_unsupported_critical and has_unsupported_critical:
            assessment = EvidenceAssessment(
                status="insufficient",
                reason=cov_res.reason,
                unique_evidence_count=unique_count,
                duplicate_clusters_count=duplicate_clusters_count,
                top_semantic_score=round(top_semantic, 4),
                lexical_match_ratio=round(lexical_ratio, 4),
                aspect_coverage=cov_res.aspect_coverage,
                aspect_details=cov_res.aspect_details,
                unsupported_aspects=unsupported,
                contradicted_aspects=contradicted,
                aspects=aspect_dict,
                aspect_support_ratio=cov_res.aspect_support_ratio
            )
            return assessment, deduped_results, clusters

        # All critical aspects supported: sufficient evidence
        dup_note = f" (Note: {duplicate_clusters_count} redundant record cluster(s) consolidated)" if duplicate_clusters_count > 0 else ""
        assessment = EvidenceAssessment(
            status="sufficient",
            reason=f"Found {unique_count} distinct historical maintenance record(s) matching the reported condition{dup_note}. {cov_res.reason}",
            unique_evidence_count=unique_count,
            duplicate_clusters_count=duplicate_clusters_count,
            top_semantic_score=round(top_semantic, 4),
            lexical_match_ratio=round(lexical_ratio, 4),
            aspect_coverage=cov_res.aspect_coverage,
            aspect_details=cov_res.aspect_details,
            unsupported_aspects=unsupported,
            contradicted_aspects=contradicted,
            aspects=aspect_dict,
            aspect_support_ratio=cov_res.aspect_support_ratio
        )
        return assessment, deduped_results, clusters

    # --- Baseline Gate Pathway (when ENABLE_ASPECT_GATE is explicitly disabled) ---
    # 1. Unanswerable / out-of-domain queries produce very low lexical overlap and weak semantic similarity
    if top_semantic < 0.38 and lexical_ratio == 0.0:
        assessment = EvidenceAssessment(
            status="insufficient",
            reason="Available historical records show negligible relevance to the requested topic.",
            unique_evidence_count=unique_count,
            duplicate_clusters_count=duplicate_clusters_count,
            top_semantic_score=round(top_semantic, 4),
            lexical_match_ratio=round(lexical_ratio, 4)
        )
        return assessment, deduped_results, clusters

    # 2. Check if evidence is sufficient in legacy baseline mode
    if top_semantic >= 0.35 or lexical_ratio >= 0.25 or any(r.score > 0 for r in results):
        dup_note = f" (Note: {duplicate_clusters_count} redundant record cluster(s) consolidated)" if duplicate_clusters_count > 0 else ""
        assessment = EvidenceAssessment(
            status="sufficient",
            reason=f"Found {unique_count} distinct historical maintenance record(s) matching the reported condition{dup_note}.",
            unique_evidence_count=unique_count,
            duplicate_clusters_count=duplicate_clusters_count,
            top_semantic_score=round(top_semantic, 4),
            lexical_match_ratio=round(lexical_ratio, 4)
        )
        return assessment, deduped_results, clusters

    # Default to insufficient if evidence quality is borderline
    assessment = EvidenceAssessment(
        status="insufficient",
        reason="Retrieved records lack sufficient specific evidence to support a factual answer.",
        unique_evidence_count=unique_count,
        duplicate_clusters_count=duplicate_clusters_count,
        top_semantic_score=round(top_semantic, 4),
        lexical_match_ratio=round(lexical_ratio, 4)
    )
    return assessment, deduped_results, clusters
