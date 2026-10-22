"""
Regression tests for Section 12: Risk-Coverage Evaluation Consistency.
Verifies mathematical primitives, edge cases (zero coverage, zero risk, full risk),
and monotonicity across decision thresholds.
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


class TestRiskCoverageEdgeCases:
    """Tests for edge cases and mathematical invariants."""

    def test_zero_coverage_edge_case(self):
        """When 0 queries are answered, coverage is 0.0 and risk is 0.0 (not div by zero error)."""
        test_queries = [
            {"is_answerable": True, "has_correct_target": True, "is_answered": False},
            {"is_answerable": False, "has_correct_target": False, "is_answered": False}
        ]
        res = evaluate_operating_point(test_queries)
        assert res["coverage"] == 0.0
        assert res["risk"] == 0.0
        assert res["answered_count"] == 0
        assert res["correct_abstention_rate"] == 1.0
        assert res["false_refusal_rate"] == 1.0

    def test_zero_risk_edge_case(self):
        """When all answered queries are correct, risk is 0.0."""
        test_queries = [
            {"is_answerable": True, "has_correct_target": True, "is_answered": True},
            {"is_answerable": True, "has_correct_target": True, "is_answered": True},
            {"is_answerable": False, "has_correct_target": False, "is_answered": False}
        ]
        res = evaluate_operating_point(test_queries)
        assert res["coverage"] == pytest.approx(2 / 3, abs=1e-4)
        assert res["risk"] == 0.0
        assert res["wrong_answered_count"] == 0
        assert res["correct_abstention_rate"] == 1.0
        assert res["false_refusal_rate"] == 0.0

    def test_full_risk_edge_case(self):
        """When all answered queries are incorrect/hallucinated, risk is 1.0."""
        test_queries = [
            {"is_answerable": False, "has_correct_target": False, "is_answered": True},
            {"is_answerable": True, "has_correct_target": False, "is_answered": True}
        ]
        res = evaluate_operating_point(test_queries)
        assert res["coverage"] == 1.0
        assert res["risk"] == 1.0
        assert res["wrong_answered_count"] == 2

    def test_aurc_monotonicity(self):
        """AURC increases with higher risk curves over the same coverage domain."""
        covs = [0.0, 0.5, 1.0]
        low_risks = [0.1, 0.1, 0.1]
        high_risks = [0.4, 0.4, 0.4]

        aurc_low = compute_aurc(covs, low_risks)
        aurc_high = compute_aurc(covs, high_risks)

        assert aurc_low < aurc_high
        assert pytest.approx(aurc_low, abs=1e-4) == 0.1
        assert pytest.approx(aurc_high, abs=1e-4) == 0.4
