"""
Regression tests for Section 8: Eliminating Fabricated Dates.
Verifies that undated records return None from get_record_date,
records with authentic dates return their recorded dates,
and no hardcoded '2026-10-22' is injected into logbook metadata.
"""

import pytest
from src.agent.tools import MaintenanceTools
from src.agent.core import format_pointwise_answer


class TestUndatedRecords:
    """Tests for authentic date representation and elimination of fabricated timestamps."""

    def test_undated_record_returns_none(self):
        """Record 100004 in the historical dataset has no date field in raw CSV metadata."""
        tools = MaintenanceTools()
        rec_date = tools.get_record_date("100004")
        assert rec_date is None

    def test_record_with_explicit_date_returns_date(self):
        tools = MaintenanceTools()
        # Mock or inject a record with explicit date in metadata
        tools.lookup["999001"] = {
            "record_id": "999001",
            "problem_raw": "TEST",
            "action_raw": "TEST",
            "metadata": {"date": "2023-05-14"}
        }
        rec_date = tools.get_record_date("999001")
        assert rec_date == "2023-05-14"

    def test_format_pointwise_labels_undated_records_honestly(self):
        tools = MaintenanceTools()
        sample_answer = (
            "• Record [ID: 100004]\n"
            "  - Issue: CYL 2 LOWER PLUG FOULED\n"
            "  - Fix: CLEANED AND RE-GAPPED PLUG [ID: 100004]"
        )
        formatted = format_pointwise_answer(sample_answer, tools)
        assert "- Date: Undated" in formatted
        assert "2026-10-22" not in formatted
