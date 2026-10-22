"""
Regression tests for Section 4: Enforcing Historical Evidence versus Approved Procedures.
Verifies detection of prescriptive requests (torque specs, airworthiness, approved repair manuals)
and backend enforcement directing users to the authoritative AMM and qualified A&P/IA personnel.
"""

import pytest
from src.evidence.policy import (
    is_prescriptive_or_regulatory_request,
    audit_evidence_policy,
    PRESCRIPTIVE_SAFETY_ADVISORY
)


class TestProcedureSafetyBoundary:
    """Tests for distinguishing historical technician actions from approved maintenance procedures."""

    def test_detects_torque_spec_request(self):
        query = "What is the approved torque spec for cylinder hold down nuts?"
        is_pres, term = is_prescriptive_or_regulatory_request(query)
        assert is_pres is True
        assert "torque spec" in term.lower()

    def test_detects_airworthiness_determination_request(self):
        query = "Can you make an airworthiness determination for this oil leak?"
        is_pres, term = is_prescriptive_or_regulatory_request(query)
        assert is_pres is True
        assert "airworthiness" in term.lower()

    def test_detects_approved_procedure_request(self):
        query = "Give me the approved procedure for magneto timing overhaul."
        is_pres, term = is_prescriptive_or_regulatory_request(query)
        assert is_pres is True
        assert "approved procedure" in term.lower()

    def test_ignores_historical_queries(self):
        query = "Find historical maintenance records mentioning spark plug fouling."
        is_pres, term = is_prescriptive_or_regulatory_request(query)
        assert is_pres is False
        assert term is None

    def test_policy_audit_enforces_prescriptive_boundary(self):
        query = "What is the torque specification for the rocker box cover screws?"
        compliant_answer = (
            "Historical logbook records only document descriptive past actions taken by technicians. "
            "They do not establish approved torque values or repair procedures. "
            "Please consult the applicable Aircraft Maintenance Manual (AMM) and authorized technician.\n\n"
            "> [!NOTE]\n"
            "> Historical logbook entries describe past actions, not approved maintenance manual procedures."
        )

        res = audit_evidence_policy(
            query=query,
            answer_text=compliant_answer,
            evidence_status="sufficient",
            retrieved_records=[],
            cited_record_ids=["100004"],
            is_fallback=False
        )

        assert res.is_prescriptive_query is True
        assert res.prescriptive_boundary_applied is True
        assert res.is_compliant is True

    def test_policy_audit_flags_missing_boundary_on_prescriptive_query(self):
        query = "Give me the torque spec for the fuel pump."
        violating_answer = (
            "The torque spec is 50 inch-pounds according to previous logs."
        )

        res = audit_evidence_policy(
            query=query,
            answer_text=violating_answer,
            evidence_status="sufficient",
            retrieved_records=[],
            cited_record_ids=["100004"],
            is_fallback=False
        )

        assert res.is_prescriptive_query is True
        assert res.prescriptive_boundary_applied is False
        assert res.is_compliant is False
        assert any("prescriptive" in w for w in res.unsupported_claim_warnings)
