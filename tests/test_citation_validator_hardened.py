"""
Regression tests for Section 6: Strengthened Citation and Exact Quotation Validation.
Verifies rejection of nonexistent IDs, unretrieved IDs, hallucinated quotes,
and cross-record quote attributions (Record B quote cited under Record A).
"""

import pytest
from src.agent.validator import validate_citations_and_quotes


class TestHardenedCitationValidator:
    """Tests for strict citation grounding and quotation verification."""

    @pytest.fixture
    def sample_lookup(self):
        return {
            "100002": {
                "problem_raw": "ENGINE KILLED WHEN IDLE OVERRIDE WAS ENGAGED.",
                "action_raw": "REMOVED & REPLACED FUEL SERVO."
            },
            "100004": {
                "problem_raw": "FOUND CYL 2 LOWER PLUG FOULED.",
                "action_raw": "CLEANED AND RE-GAPPED SPARK PLUG."
            },
            "100009": {
                "problem_raw": "#2 INTAKE IS LEAKING.",
                "action_raw": "REMOVED & REPLACED #2 INTAKE GASKET."
            }
        }

    def test_valid_single_citation_and_exact_quote(self, sample_lookup):
        text = 'Record [ID: 100004]: discrepancy noted "FOUND CYL 2 LOWER PLUG FOULED."'
        res = validate_citations_and_quotes(
            answer_text=text,
            valid_db_ids={"100002", "100004", "100009"},
            session_retrieved_ids={"100004"},
            records_lookup=sample_lookup
        )
        assert res.is_valid is True
        assert res.cited_ids == ["100004"]
        assert len(res.invalid_quotes) == 0

    def test_valid_multiple_citations(self, sample_lookup):
        text = (
            '• Record [ID: 100004]\n'
            '  - Issue: "FOUND CYL 2 LOWER PLUG FOULED."\n'
            '• Record [ID: 100009]\n'
            '  - Issue: "#2 INTAKE IS LEAKING."'
        )
        res = validate_citations_and_quotes(
            answer_text=text,
            valid_db_ids={"100002", "100004", "100009"},
            session_retrieved_ids={"100004", "100009"},
            records_lookup=sample_lookup
        )
        assert res.is_valid is True
        assert set(res.cited_ids) == {"100004", "100009"}

    def test_rejects_nonexistent_record_id(self, sample_lookup):
        text = 'Record [ID: 999999] mentions fuel leak.'
        res = validate_citations_and_quotes(
            answer_text=text,
            valid_db_ids={"100002", "100004"},
            session_retrieved_ids={"100004"},
            records_lookup=sample_lookup
        )
        assert res.is_valid is False
        assert "999999" in res.invalid_nonexistent_ids
        assert "[INVALID_CITATION_999999]" in res.sanitized_answer

    def test_rejects_unretrieved_record_id(self, sample_lookup):
        # 100002 exists in db, but was NOT retrieved in this session
        text = 'Record [ID: 100002] mentions fuel servo.'
        res = validate_citations_and_quotes(
            answer_text=text,
            valid_db_ids={"100002", "100004"},
            session_retrieved_ids={"100004"},
            records_lookup=sample_lookup
        )
        assert res.is_valid is False
        assert "100002" in res.unretrieved_ids

    def test_rejects_cross_record_quote_attribution(self, sample_lookup):
        # Text cites Record 100004, but the quote is actually from Record 100009!
        text = '• Record [ID: 100004]\n  - Issue: "#2 INTAKE IS LEAKING."'
        res = validate_citations_and_quotes(
            answer_text=text,
            valid_db_ids={"100002", "100004", "100009"},
            session_retrieved_ids={"100004", "100009"},
            records_lookup=sample_lookup
        )
        assert res.is_valid is False
        assert len(res.cross_record_attributions) > 0
        assert any("Cross-record" in w for w in res.warning_messages)

    def test_rejects_hallucinated_quote(self, sample_lookup):
        text = 'Record [ID: 100004]: "CYLINDER HEAD TEMPERATURE EXCEEDED 500 DEGREES."'
        res = validate_citations_and_quotes(
            answer_text=text,
            valid_db_ids={"100004"},
            session_retrieved_ids={"100004"},
            records_lookup=sample_lookup
        )
        assert res.is_valid is False
        assert len(res.invalid_quotes) == 1
