"""
Deterministic Evidence Assessment & Near-Duplicate Clustering Layer.
Evaluates retrieval evidence quality without relying exclusively on LLM self-reporting
or arbitrary rank-fusion score thresholds.
"""

import re
from typing import List, Tuple, Dict, Set, Literal
from pydantic import BaseModel, Field

from src.retrieval.models import RetrievalResult


EvidenceStatus = Literal["sufficient", "insufficient", "needs_clarification"]

_WORD_PATTERN = re.compile(r"\b[A-Za-z0-9#/'\-]+\b")
_CYLINDER_PATTERN = re.compile(r"#\s*(\d+)")

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


class EvidenceAssessment(BaseModel):
    """
    Structured outcome of the deterministic evidence evaluation.
    """
    status: EvidenceStatus = Field(..., description="'sufficient', 'insufficient', or 'needs_clarification'")
    reason: str = Field(..., description="Deterministic rationale for the assigned evidence status")
    unique_evidence_count: int = Field(..., description="Count of distinct, non-duplicate records found")
    duplicate_clusters_count: int = Field(..., description="Number of duplicate or near-duplicate clusters identified")
    top_semantic_score: float = Field(default=0.0, description="Highest dense semantic similarity score in results")
    lexical_match_ratio: float = Field(default=0.0, description="Fraction of non-stop query terms matched in results")


def get_token_set(text: str) -> Set[str]:
    """Extracts lowercase alphanumeric tokens from text."""
    if not text:
        return set()
    return {t.lower() for t in _WORD_PATTERN.findall(text)}


def extract_cylinders(text: str) -> Set[str]:
    """Extracts cylinder number designations e.g. #1, #2, #3, #4."""
    if not text:
        return set()
    return set(_CYLINDER_PATTERN.findall(text))


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
    results: List[RetrievalResult]
) -> Tuple[EvidenceAssessment, List[RetrievalResult], Dict[str, List[str]]]:
    """
    Deterministically evaluates whether retrieved evidence is sufficient, insufficient,
    or needs clarification.
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
                lexical_match_ratio=0.0
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
            lexical_match_ratio=0.0
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

    # Decision logic calibrated against evaluation dataset:
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

    # 2. Check if evidence is sufficient
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
