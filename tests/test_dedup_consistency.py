"""
Regression tests for Section 7: Deduplication Consistency.
Verifies that near-duplicate clusters do not inflate candidate counts,
distinct cylinder events are never improperly merged,
and cluster mappings remain inspectable and consistent.
"""

import pytest
from src.retrieval.models import RetrievalResult
from src.agent.evidence_gate import cluster_near_duplicates, assess_retrieval_evidence


class TestDeduplicationConsistency:
    """Tests for consistent deduplication and cluster representative preservation."""

    def test_duplicate_clusters_do_not_inflate_unique_count(self):
        """Three identical intake leak entries should yield unique_count=1 and 1 representative."""
        results = [
            RetrievalResult(
                record_id="100008", rank=1, score=0.04, mode="hybrid",
                problem_raw="#2 INTAKE IS LEAKING.", action_raw="REMOVED & REPLACED GASKET.",
                problem_normalized="#2 INTAKE IS LEAKING.", action_normalized="REMOVED & REPLACED GASKET.",
                excerpt="#2 INTAKE IS LEAKING."
            ),
            RetrievalResult(
                record_id="100009", rank=2, score=0.038, mode="hybrid",
                problem_raw="#2 INTAKE IS LEAKING.", action_raw="REMOVED & REPLACED GASKET.",
                problem_normalized="#2 INTAKE IS LEAKING.", action_normalized="REMOVED & REPLACED GASKET.",
                excerpt="#2 INTAKE IS LEAKING."
            ),
            RetrievalResult(
                record_id="100086", rank=3, score=0.035, mode="hybrid",
                problem_raw="#2 INTAKE IS LEAKING.", action_raw="REMOVED & REPLACED GASKET.",
                problem_normalized="#2 INTAKE IS LEAKING.", action_normalized="REMOVED & REPLACED GASKET.",
                excerpt="#2 INTAKE IS LEAKING."
            )
        ]

        reps, clusters = cluster_near_duplicates(results, similarity_threshold=0.85)
        assert len(reps) == 1
        assert reps[0].record_id == "100008"
        assert set(clusters["100008"]) == {"100008", "100009", "100086"}

    def test_different_cylinders_never_merged(self):
        """Identical wording on different cylinders (#2 vs #3) must remain distinct events."""
        results = [
            RetrievalResult(
                record_id="100008", rank=1, score=0.04, mode="hybrid",
                problem_raw="#2 INTAKE IS LEAKING.", action_raw="REMOVED & REPLACED GASKET.",
                problem_normalized="#2 INTAKE IS LEAKING.", action_normalized="REMOVED & REPLACED GASKET.",
                excerpt="#2 INTAKE IS LEAKING."
            ),
            RetrievalResult(
                record_id="100010", rank=2, score=0.038, mode="hybrid",
                problem_raw="#3 INTAKE IS LEAKING.", action_raw="REMOVED & REPLACED GASKET.",
                problem_normalized="#3 INTAKE IS LEAKING.", action_normalized="REMOVED & REPLACED GASKET.",
                excerpt="#3 INTAKE IS LEAKING."
            )
        ]

        reps, clusters = cluster_near_duplicates(results, similarity_threshold=0.85)
        assert len(reps) == 2
        assert len(clusters) == 2
