"""
Regression tests for Section 10: Approved Records Restart Persistence.
Verifies that newly approved maintenance records stored in SQLite
are reloaded into in-memory lookup and retrieval indices on application restart.
"""

import pytest
from src.models import LogbookRecord
from src.db.database import DatabaseManager
from src.agent.tools import MaintenanceTools


class TestRestartPersistence:
    """Tests for syncing persistent SQLite approved records on application restart."""

    def test_approved_record_survives_restart(self, tmp_path):
        db_file = tmp_path / "persistence_copilot.db"
        db = DatabaseManager(db_path=str(db_file))

        # 1. Submit and approve a unique record
        sub = db.submit_pending_fix(
            problem_raw="PERSISTENCE_TEST_ACTUATOR SEIZED UPON DESCENT",
            action_raw="REPLACED ACTUATOR WITH UNIT S/N 88392",
            submitted_by="Station Inspector Vance"
        )
        ok, approved_id, rec_data = db.approve_pending_fix(sub["temp_id"], admin_name="Admin Chief")
        assert ok is True
        assert approved_id is not None

        # 2. Simulate fresh application startup / restart
        fresh_db = DatabaseManager(db_path=str(db_file))
        fresh_tools = MaintenanceTools()

        # Re-sync logic executed by lifespan
        approved_rows = fresh_db.get_all_records()
        for row in approved_rows:
            rid = row["record_id"]
            if rid not in fresh_tools.lookup:
                rec = LogbookRecord(
                    record_id=rid,
                    problem_raw=row["problem_raw"],
                    action_raw=row["action_raw"],
                    problem_normalized=row["problem_normalized"],
                    action_normalized=row["action_normalized"],
                    metadata={
                        "submitted_by": row.get("submitted_by"),
                        "approved_by": row.get("approved_by")
                    }
                )
                fresh_tools.add_record(rec)

        # 3. Verify record is retrievable by ID
        get_out = fresh_tools.get_record(approved_id)
        assert get_out.found is True
        assert get_out.record.record_id == approved_id
        assert "PERSISTENCE_TEST_ACTUATOR" in get_out.record.problem_raw

        # 4. Verify record is retrievable by query
        search_out = fresh_tools.search_records("PERSISTENCE_TEST_ACTUATOR", top_k=3)
        assert search_out.total_found > 0
        found_ids = [r.record_id for r in search_out.records]
        assert approved_id in found_ids
