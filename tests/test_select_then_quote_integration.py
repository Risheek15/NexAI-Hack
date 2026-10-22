"""
Regression tests for Section 5: Select-Then-Quote Online and Fallback Integration.
Verifies keyword argument compatibility in fallback, strict unit ID validation,
and deterministic verbatim quote rendering.
"""

import pytest
from src.agent.tools import MaintenanceTools
from src.agent.core import MaintenanceAgent
from src.agent.segmenter import segment_logbook_record, EvidenceUnit
from src.agent.select_then_quote import (
    StructuredClaim,
    validate_and_render_claims,
    build_unit_registry_from_records
)


class TestSelectThenQuoteIntegration:
    """Tests for Select-Then-Quote pipeline and fallback execution."""

    def test_fallback_select_then_quote_execution(self):
        """Verifies that running fallback in select_then_quote mode executes without TypeError."""
        tools = MaintenanceTools()
        agent = MaintenanceAgent(tools=tools, api_key="", evidence_mode="select_then_quote")

        resp = agent.ask("fuel servo replacement")
        assert resp.is_fallback is True
        assert "Historical Maintenance Logbook Evidence" in resp.answer
        assert resp.citation_validation is not None

    def test_strict_unit_id_validation_and_rejection(self):
        """Verifies rejection of malformed, nonexistent, and unretrieved unit IDs."""
        sample_records = [
            {
                "record_id": "100004",
                "problem_raw": "CYL 2 LOWER PLUG FOULED.",
                "action_raw": "CLEANED AND RE-GAPPED PLUG."
            }
        ]
        unit_registry = build_unit_registry_from_records(sample_records)
        session_ids = {"100004"}

        # Claim with 1 valid unit, 1 malformed unit, 1 unretrieved unit, 1 nonexistent unit
        claims = [
            StructuredClaim(
                claim_type="recorded_symptom",
                unit_ids=[
                    "100004#u1",         # Valid
                    "bad_unit_format",   # Malformed
                    "100099#u1",         # Unretrieved
                    "100004#u99"         # Nonexistent
                ],
                paraphrase="Symptom analysis"
            )
        ]

        resp = validate_and_render_claims(
            query="CYL 2 plug",
            claims=claims,
            session_retrieved_record_ids=session_ids,
            unit_registry=unit_registry
        )

        assert resp.is_valid is False
        assert len(resp.rejected_unit_ids) == 3
        assert any("malformed" in r for r in resp.rejected_unit_ids)
        assert any("unretrieved" in r for r in resp.rejected_unit_ids)
        assert any("nonexistent" in r for r in resp.rejected_unit_ids)
        assert "CYL 2 LOWER PLUG FOULED." in resp.rendered_markdown
