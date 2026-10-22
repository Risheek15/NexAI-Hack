"""
Unit and integration tests for Phase 5: Evidence Policy, Refusal/Abstention, and Grounded Answer Generation.
Verifies safety disclaimer enforcement, causation fallacy detection, conflicting repair action detection,
structured abstention formatting, clarification prompting, and full policy auditing.
"""

import pytest
from src.evidence.policy import (
    audit_evidence_policy,
    check_safety_disclaimer_present,
    detect_causation_fallacies,
    detect_conflicting_historical_actions,
    PolicyAuditResult
)
from src.evidence.refusal import (
    format_abstention_response,
    format_clarification_response,
    SAFETY_DISCLAIMER
)


class TestSafetyDisclaimerAndCausation:
    """Tests for mandatory descriptive safety disclaimers and causation fallacy guards."""

    def test_safety_disclaimer_detection(self):
        # Text with valid disclaimer phrasing
        compliant_text = (
            "Historical logbook entries describe past actions taken by technicians; "
            "they are not approved maintenance instructions."
        )
        assert check_safety_disclaimer_present(compliant_text) is True

        # Text referencing maintenance manual
        amm_text = "For authorized procedures, refer to applicable maintenance manual."
        assert check_safety_disclaimer_present(amm_text) is True

        # Non-compliant text
        non_compliant_text = "Here is the repair procedure you must perform on the aircraft engine."
        assert check_safety_disclaimer_present(non_compliant_text) is False

    def test_causation_fallacy_detection(self):
        # Claim asserting definitive causation
        bad_text = "A loose oil stick definitely caused the engine oil loss and cylinder failure."
        warnings = detect_causation_fallacies(bad_text)
        assert len(warnings) > 0
        assert "definitely caused" in warnings[0]

        # Properly qualified descriptive text
        good_text = (
            "Historical record [ID: 100044] reported a loose oil dipstick. "
            "Technicians replaced the seal assembly IAW standard procedures."
        )
        assert len(detect_causation_fallacies(good_text)) == 0


class TestConflictingHistoricalActions:
    """Tests identifying divergent corrective actions across historical records for similar squawks."""

    def test_detect_divergent_repair_paths(self):
        # In some records technicians retorqued hardware, in others they replaced the part
        records = [
            {"record_id": "100010", "problem_raw": "BAFFLE CRACKED", "action_raw": "TIGHTENED AND TORQUED BOLTS"},
            {"record_id": "100012", "problem_raw": "BAFFLE CRACKED", "action_raw": "STOP DRILLED CRACK AND FABRICATED DOUBLER"},
            {"record_id": "100015", "problem_raw": "BAFFLE DAMAGED", "action_raw": "REPLACED WITH NEW BAFFLE ASSY"}
        ]
        conflicts = detect_conflicting_historical_actions(records)
        assert len(conflicts) > 0
        assert "divergent repair paths" in conflicts[0]
        assert "tightening" in conflicts[0]
        assert "stop-drilling" in conflicts[0]
        assert "replacement" in conflicts[0]

    def test_no_conflict_when_actions_consistent(self):
        records = [
            {"record_id": "100002", "problem_raw": "SERVO DEFECT", "action_raw": "REMOVED AND REPLACED FUEL SERVO"},
            {"record_id": "100008", "problem_raw": "SERVO LEAK", "action_raw": "REPLACED FUEL SERVO"}
        ]
        conflicts = detect_conflicting_historical_actions(records)
        assert len(conflicts) == 0


class TestRefusalAndClarificationFormatting:
    """Tests structured abstention statements and clarification prompts."""

    def test_abstention_formatting(self):
        query = "autopilot yaw damper failure on Boeing 777"
        reason = "Out-of-domain query. Logbook covers reciprocating aircraft engines."
        closest = [
            {"record_id": "100001", "problem_raw": "OIL LEAK", "action_raw": "REPLACED GASKET"}
        ]

        resp = format_abstention_response(query, reason, closest)
        assert "Historical Evidence Insufficient to Answer Query" in resp
        assert query in resp
        assert reason in resp
        assert "[ID: 100001]" in resp
        assert "NOT manufacturer-approved maintenance instructions" in resp

    def test_clarification_formatting(self):
        query = "leak"
        reason = "Generic symptom without component or cylinder context."

        resp = format_clarification_response(query, reason)
        assert "Clarification Requested" in resp
        assert "Please clarify:" in resp
        assert "Which specific component" in resp
        assert "Which cylinder number" in resp
        assert "NOT manufacturer-approved maintenance instructions" in resp


class TestPolicyAuditResult:
    """Tests comprehensive audit of answer compliance against Maintenance Copilot Evidence Policy."""

    def test_fully_compliant_answer_audit(self):
        query = "fuel servo replacement"
        answer = (
            "Historical logbook record [ID: 100002] records that technicians removed and replaced "
            "the fuel servo when the engine idle override killed the engine.\n\n"
            "*(Note: Historical logbook entries describe past actions taken by technicians; "
            "they are NOT manufacturer-approved maintenance instructions. Refer to applicable maintenance manual.)*"
        )
        audit = audit_evidence_policy(
            query=query,
            answer_text=answer,
            evidence_status="sufficient",
            retrieved_records=[{"record_id": "100002", "problem_raw": "ENGINE KILLED", "action_raw": "REMOVED & REPLACED FUEL SERVO"}],
            cited_record_ids=["100002"]
        )
        assert audit.is_compliant is True
        assert audit.has_safety_disclaimer is True
        assert len(audit.potential_causation_fallacies) == 0
        assert len(audit.unsupported_claim_warnings) == 0

    def test_audit_flags_missing_disclaimer(self):
        query = "fuel servo replacement"
        answer = "Record [ID: 100002] indicates the fuel servo was replaced."
        audit = audit_evidence_policy(
            query=query,
            answer_text=answer,
            evidence_status="sufficient",
            retrieved_records=[{"record_id": "100002", "problem_raw": "X", "action_raw": "Y"}],
            cited_record_ids=["100002"]
        )
        assert audit.is_compliant is False
        assert audit.has_safety_disclaimer is False
        assert "Missing mandatory descriptive safety disclaimer" in audit.policy_summary

    def test_audit_flags_unsupported_claims_without_citations(self):
        query = "fuel servo replacement"
        answer = (
            "Technicians always replace fuel servos when idle override fails.\n\n"
            "*(Historical logbook entries describe past actions taken by technicians; not approved maintenance instructions.)*"
        )
        audit = audit_evidence_policy(
            query=query,
            answer_text=answer,
            evidence_status="sufficient",
            retrieved_records=[{"record_id": "100002", "problem_raw": "X", "action_raw": "Y"}],
            cited_record_ids=[]  # No citations!
        )
        assert audit.is_compliant is False
        assert len(audit.unsupported_claim_warnings) > 0
        assert "no record IDs were cited" in audit.unsupported_claim_warnings[0]
