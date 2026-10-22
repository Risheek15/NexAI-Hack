"""
Unit tests for risk-coverage evaluation metric functions.
Uses deterministic, tiny hand-made inputs with mathematically verifiable ground truth.
"""

import pytest
from src.evidence.risk_coverage import (
    compute_coverage,
    compute_risk,
    compute_correct_abstention_rate,
    compute_false_refusal_rate,
    compute_aurc,
    evaluate_operating_point
)


class TestRiskCoveragePrimitives:
    """Tests for elementary mathematical metrics."""

    def test_compute_coverage_known_inputs(self):
        assert compute_coverage(0, 10) == 0.0
        assert compute_coverage(5, 10) == 0.5
        assert compute_coverage(8, 10) == 0.8
        assert compute_coverage(10, 10) == 1.0
        # Edge case: zero total
        assert compute_coverage(0, 0) == 0.0

    def test_compute_risk_known_inputs(self):
        assert compute_risk(0, 10) == 0.0
        assert compute_risk(2, 10) == 0.2
        assert compute_risk(5, 10) == 0.5
        assert compute_risk(10, 10) == 1.0
        # Edge case: zero answered queries (coverage=0) -> risk=0.0
        assert compute_risk(0, 0) == 0.0

    def test_compute_correct_abstention_and_false_refusal(self):
        # 3 correctly abstained out of 5 unanswerable
        assert compute_correct_abstention_rate(3, 5) == 0.6
        assert compute_correct_abstention_rate(5, 5) == 1.0
        assert compute_correct_abstention_rate(0, 5) == 0.0
        assert compute_correct_abstention_rate(0, 0) == 0.0

        # 2 falsely refused out of 10 answerable
        assert compute_false_refusal_rate(2, 10) == 0.2
        assert compute_false_refusal_rate(0, 10) == 0.0
        assert compute_false_refusal_rate(10, 10) == 1.0
        assert compute_false_refusal_rate(0, 0) == 0.0

    def test_compute_aurc_known_geometries(self):
        # 1. Triangle geometry: f(x) = x from (0,0) to (1,1) -> Area = 0.5 * 1 * 1 = 0.5
        covs_tri = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        risks_tri = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert pytest.approx(compute_aurc(covs_tri, risks_tri), abs=1e-5) == 0.5

        # 2. Rectangle geometry: constant risk = 0.25 from cov=0.0 to 1.0 -> Area = 0.25
        covs_rect = [0.0, 0.5, 1.0]
        risks_rect = [0.25, 0.25, 0.25]
        assert pytest.approx(compute_aurc(covs_rect, risks_rect), abs=1e-5) == 0.25

        # 3. Disordered points should be properly sorted
        covs_unsorted = [1.0, 0.0, 0.5]
        risks_unsorted = [0.25, 0.25, 0.25]
        assert pytest.approx(compute_aurc(covs_unsorted, risks_unsorted), abs=1e-5) == 0.25

        # Edge case: empty lists
        assert compute_aurc([], []) == 0.0


class TestEvaluateOperatingPoint:
    """Tests evaluating a full operating point on a hand-made 4-query dataset."""

    def test_evaluate_operating_point_4_queries(self):
        # Dataset of 4 synthetic queries:
        # Q1: Answerable, correct target found, answered -> Correct
        # Q2: Answerable, target NOT found, answered -> Wrong (Error)
        # Q3: Unanswerable, answered -> Wrong (Hallucination)
        # Q4: Unanswerable, abstained -> Correct Abstention
        test_queries = [
            {"is_answerable": True, "has_correct_target": True, "is_answered": True},
            {"is_answerable": True, "has_correct_target": False, "is_answered": True},
            {"is_answerable": False, "has_correct_target": False, "is_answered": True},
            {"is_answerable": False, "has_correct_target": False, "is_answered": False},
        ]

        res = evaluate_operating_point(test_queries)

        assert res["total_count"] == 4
        assert res["answered_count"] == 3
        assert res["wrong_answered_count"] == 2
        assert res["total_answerable"] == 2
        assert res["total_unanswerable"] == 2

        # Coverage: 3 / 4 = 0.75
        assert res["coverage"] == 0.75

        # Risk: 2 / 3 = 0.6667
        assert pytest.approx(res["risk"], abs=1e-4) == 0.6667

        # Correct Abstention Rate: 1 / 2 = 0.5
        assert res["correct_abstention_rate"] == 0.5

        # False Refusal Rate: 0 / 2 = 0.0
        assert res["false_refusal_rate"] == 0.0

    def test_evaluate_operating_point_full_abstention(self):
        # All queries abstained (coverage=0)
        test_queries = [
            {"is_answerable": True, "has_correct_target": True, "is_answered": False},
            {"is_answerable": False, "has_correct_target": False, "is_answered": False},
        ]
        res = evaluate_operating_point(test_queries)
        assert res["coverage"] == 0.0
        assert res["risk"] == 0.0
        assert res["correct_abstention_rate"] == 1.0
        assert res["false_refusal_rate"] == 1.0
