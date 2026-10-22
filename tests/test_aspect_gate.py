"""
Unit tests for TASK N3: Aspect-level evidence coverage for abstention.
Verifies all 6 mandatory test scenarios:
1. Fully covered query
2. Partially covered (component matches but symptom does not)
3. No coverage
4. Duplicate-only support
5. Ambiguous query (needs clarification)
6. Extraction failure fallback (deterministic rule/heuristic extractor)
Also verifies flag toggling (baseline preservation) and agreement evaluation.
"""

import pytest
from unittest.mock import patch

from src.retrieval.models import RetrievalResult
from src.agent.aspect_extractor import (
    QueryAspects,
    extract_aspects,
    extract_aspects_deterministic,
    evaluate_aspect_extraction_agreement
)
from src.agent.aspect_coverage import check_aspect_coverage, match_aspect_in_text
from src.agent.evidence_gate import (
    assess_retrieval_evidence,
    AspectGateConfig,
    EvidenceAssessment
)


@pytest.fixture
def sample_records():
    """Sample records representing realistic logbook entries."""
    return [
        RetrievalResult(
            record_id="100002",
            rank=1,
            score=0.045,
            mode="hybrid",
            problem_raw="ENGINE KILLED WHEN IDLE OVERRIDE WAS ENGAGED.",
            action_raw="REMOVED & REPLACED FUEL SERVO.",
            problem_normalized="ENGINE KILLED WHEN IDLE OVERRIDE WAS ENGAGED.",
            action_normalized="REMOVED & REPLACED FUEL SERVO.",
            excerpt="ENGINE KILLED WHEN IDLE OVERRIDE WAS ENGAGED. REMOVED & REPLACED FUEL SERVO."
        ),
        RetrievalResult(
            record_id="100009",
            rank=2,
            score=0.038,
            mode="hybrid",
            problem_raw="#2 INTAKE IS LEAKING.",
            action_raw="REMOVED & REPLACED #2 INTAKE GASKET.",
            problem_normalized="#2 INTAKE IS LEAKING.",
            action_normalized="REMOVED & REPLACED #2 INTAKE GASKET.",
            excerpt="#2 INTAKE IS LEAKING. REMOVED & REPLACED #2 INTAKE GASKET."
        ),
        RetrievalResult(
            record_id="100086",
            rank=3,
            score=0.037,
            mode="hybrid",
            problem_raw="#2 INTAKE IS LEAKING.",
            action_raw="REMOVED & REPLACED #2 INTAKE GASKET.",
            problem_normalized="#2 INTAKE IS LEAKING.",
            action_normalized="REMOVED & REPLACED #2 INTAKE GASKET.",
            excerpt="#2 INTAKE IS LEAKING. REMOVED & REPLACED #2 INTAKE GASKET."
        )
    ]


class TestAspectCoverageGate:
    """Test suite for aspect-level evidence coverage and deterministic abstention."""

    def test_fully_covered(self, sample_records):
        """1. Fully covered: both component and requested action/symptom match retrieved records."""
        query = "removed and replaced fuel servo"
        # Only fuel servo record
        results = [sample_records[0]]

        assessment, deduped, clusters = assess_retrieval_evidence(
            query=query,
            results=results,
            enable_aspect_gate=True
        )

        assert assessment.status == "sufficient"
        assert assessment.aspect_coverage.get("component") is True
        assert assessment.aspect_coverage.get("requested_info") is True
        assert len(assessment.unsupported_aspects) == 0
        assert "supported across" in assessment.reason
        assert "fuel servo" in assessment.reason.lower()

    def test_partially_covered_component_matches_symptom_does_not(self, sample_records):
        """2. Partially covered: component matches ('fuel servo') but symptom ('fire in flight') does not."""
        query = "fuel servo fire in flight"
        # Only fuel servo record (which mentions engine killed, not fire or in flight)
        results = [sample_records[0]]

        assessment, deduped, clusters = assess_retrieval_evidence(
            query=query,
            results=results,
            enable_aspect_gate=True
        )

        assert assessment.status == "insufficient"
        assert assessment.aspect_coverage.get("component") is True
        assert assessment.aspect_coverage.get("symptom") is False
        assert "symptom" in assessment.unsupported_aspects
        assert "Partially supported" in assessment.reason
        assert "lack evidence for symptom" in assessment.reason

    def test_no_coverage(self, sample_records):
        """3. No coverage: neither component nor action exists in retrieved records."""
        query = "landing gear hydraulic retraction actuator overhaul"
        # Retrieved records are engine fuel servo and intake gasket
        results = [sample_records[0], sample_records[1]]

        assessment, deduped, clusters = assess_retrieval_evidence(
            query=query,
            results=results,
            enable_aspect_gate=True
        )

        assert assessment.status == "insufficient"
        assert assessment.aspect_coverage.get("component") is False
        assert "component" in assessment.unsupported_aspects
        assert "Insufficient evidence" in assessment.reason
        assert "landing gear" in assessment.reason.lower()

    def test_duplicate_only_support(self, sample_records):
        """4. Duplicate-only support: 3 records retrieved, but 2 are near-duplicates belonging to the SAME cluster."""
        query = "#2 intake gasket leak"
        # Record 100009 and 100086 are identical duplicate squawks for #2 intake gasket
        results = [sample_records[1], sample_records[2]]

        assessment, deduped, clusters = assess_retrieval_evidence(
            query=query,
            results=results,
            enable_aspect_gate=True
        )

        # There are 2 raw records, but only 1 unique cluster!
        assert assessment.unique_evidence_count == 1
        assert assessment.duplicate_clusters_count == 1
        # Gate recognizes 1 distinct cluster supports it
        assert assessment.aspect_coverage.get("component") is True
        assert assessment.aspect_coverage.get("symptom") is True

    def test_ambiguous_query(self, sample_records):
        """5. Ambiguous query: single vague symptom ('leak') without component context."""
        query = "leak"
        results = [sample_records[1]]

        assessment, deduped, clusters = assess_retrieval_evidence(
            query=query,
            results=results,
            enable_aspect_gate=True
        )

        assert assessment.status == "needs_clarification"
        assert (
            "generic" in assessment.reason.lower()
            or "specify" in assessment.reason.lower()
            or "lacks component context" in assessment.reason.lower()
        )

    def test_extraction_failure_fallback(self):
        """6. Extraction failure fallback: when LLM API throws an error, deterministic extractor succeeds cleanly."""
        query = "push rod tube shroud seal oil leak at crankcase"

        # Mock Gemini call to simulate API error / network failure
        with patch("src.agent.aspect_extractor.extract_aspects_gemini", return_value=None):
            aspects = extract_aspects(query, api_key="MOCK_INVALID_KEY", use_cache=False)

        assert isinstance(aspects, QueryAspects)
        assert aspects.component == "push rod tube shroud seal"
        assert aspects.symptom == "oil leak"
        assert aspects.condition == "at crankcase"

    def test_flag_preserves_baseline_when_disabled(self):
        """Verifies that when enable_aspect_gate=False, baseline behavior is 100% preserved."""
        # Realistic DEV_18 failure case: shallow lexical match on 'landing' fools baseline into sufficient
        query = "landing gear hydraulic retraction actuator overhaul"
        shallow_match_rec = RetrievalResult(
            record_id="104603",
            rank=1,
            score=0.0276,
            mode="hybrid",
            problem_raw="HARD LANDING REPORTED BY TOWER.",
            action_raw="COMPLETED REASSEMBLY; INCLUDING FUEL LINES AT PUMP. RAN",
            problem_normalized="HARD LANDING REPORTED BY TOWER.",
            action_normalized="COMPLETED REASSEMBLY; INCLUDING FUEL LINES AT PUMP. RAN",
            excerpt="HARD LANDING REPORTED BY TOWER. COMPLETED REASSEMBLY."
        )
        results = [shallow_match_rec]

        # Baseline gate with flag disabled
        assessment_base, _, _ = assess_retrieval_evidence(
            query=query,
            results=results,
            enable_aspect_gate=False
        )

        # Aspect gate with flag enabled
        assessment_aspect, _, _ = assess_retrieval_evidence(
            query=query,
            results=results,
            enable_aspect_gate=True
        )

        # Baseline gate evaluates global score (sufficient if any score > 0), whereas aspect gate abstains!
        assert assessment_base.status == "sufficient"
        assert assessment_aspect.status == "insufficient"
        assert "component" in assessment_aspect.unsupported_aspects

    def test_aspect_agreement_evaluator(self):
        """Verifies that evaluate_aspect_extraction_agreement runs and produces valid metrics."""
        test_queries = [
            {"query_id": "T1", "query": "removed and replaced fuel servo"},
            {"query_id": "T2", "query": "oil dipstick oil stick is loose"}
        ]
        report = evaluate_aspect_extraction_agreement(test_queries)
        assert report["total_queries"] == 2
        assert "component" in report["concordance_rates"]
        assert report["strict_full_agreement_rate"] >= 0.5
