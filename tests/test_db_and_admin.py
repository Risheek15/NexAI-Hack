"""
Automated Test Suite for SQLite Database, User Authentication,
Technician Fix Submissions, and Airport Administrator Governance.
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.models import LogbookRecord
from src.db.database import DatabaseManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.tools import MaintenanceTools
from src.api.main import app, resources


@pytest.fixture
def temp_db():
    """Provides a fresh temporary SQLite database with default users seeded."""
    tmpdir = tempfile.mkdtemp(prefix="copilot_test_")
    db_path = Path(tmpdir) / "test_copilot.db"
    manager = DatabaseManager(db_path=str(db_path))
    yield manager
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


def test_user_authentication_and_creation(temp_db):
    """Verifies login for default accounts and admin creation of new technician."""
    # 1. Admin login
    admin = temp_db.authenticate_user("admin", "admin123")
    assert admin is not None
    assert admin["role"] == "admin"

    # 2. Tech login
    tech1 = temp_db.authenticate_user("technician1", "password1")
    assert tech1 is not None
    assert tech1["role"] == "technician"

    # 3. Bad password rejected
    bad = temp_db.authenticate_user("technician1", "wrongpassword")
    assert bad is None

    # 4. Admin creates 4th technician
    ok, msg, new_user = temp_db.create_user(
        username="tech_sarah",
        password="secretpassword",
        full_name="Sarah Connor (Turbines)",
        role="technician"
    )
    assert ok is True
    assert new_user["username"] == "tech_sarah"

    # 5. Authenticate newly created user
    auth_new = temp_db.authenticate_user("tech_sarah", "secretpassword")
    assert auth_new is not None
    assert auth_new["full_name"] == "Sarah Connor (Turbines)"

    # 6. Check list of technicians
    techs = temp_db.get_users(role_filter="technician")
    assert len(techs) >= 4


def test_db_seeding_and_lookup(temp_db):
    """Verifies that seed_from_jsonl populates records correctly and get_record works."""
    count = temp_db.seed_from_jsonl("data/processed/records.jsonl")
    assert count >= 6000

    rec = temp_db.get_record("100004")
    assert rec is not None
    assert rec["record_id"] == "100004"
    assert "choke" in rec["problem_raw"].lower() or "engine" in rec["problem_raw"].lower()

    # Dynamic stats test
    stats = temp_db.get_stats()
    assert stats["total_records"] >= 6000
    assert stats["total_technicians"] >= 3


def test_technician_submit_and_admin_approve(temp_db):
    """
    Tests the full end-to-end lifecycle:
    1. Technician submits a fix (enters pending queue).
    2. Admin views pending list.
    3. Admin approves fix -> assigned new sequential ID.
    4. Record is permanently in `records` table.
    """
    temp_db.seed_from_jsonl("data/processed/records.jsonl")

    # 1. Submit fix
    sub_res = temp_db.submit_pending_fix(
        problem_raw="ENG 2 OIL PRESS XMTR FLT DECK INDICATION INTERMITTENT",
        action_raw="REPLACED OIL PRESSURE TRANSMITTER PER AMM 79-31-01. OPS TEST NORMAL.",
        submitted_by="Technician Collins"
    )
    temp_id = sub_res["temp_id"]
    assert temp_id.startswith("REQ-")
    assert sub_res["status"] == "pending"

    # 2. Admin retrieves pending
    pending_list = temp_db.get_pending_fixes(status_filter="pending")
    matching = [p for p in pending_list if p["temp_id"] == temp_id]
    assert len(matching) == 1
    assert matching[0]["submitted_by"] == "Technician Collins"

    # 3. Admin approves
    ok, new_id, approved_rec = temp_db.approve_pending_fix(temp_id=temp_id, admin_name="Chief Inspector Davis")
    assert ok is True
    assert new_id is not None
    assert int(new_id) >= 106170
    assert approved_rec["approved_by"] == "Chief Inspector Davis"

    # 4. Verify record in official database
    stored = temp_db.get_record(new_id)
    assert stored is not None
    assert stored["problem_raw"] == "ENG 2 OIL PRESS XMTR FLT DECK INDICATION INTERMITTENT"
    assert stored["status"] == "approved"

    # 5. Pending list is now empty for this item
    remaining_pending = temp_db.get_pending_fixes(status_filter="pending")
    assert not any(p["temp_id"] == temp_id for p in remaining_pending)


def test_admin_reject_fix(temp_db):
    """Verifies that an admin can reject a pending fix with audit explanation."""
    sub_res = temp_db.submit_pending_fix(
        problem_raw="NOISE IN CABIN",
        action_raw="CHECKED OK",
        submitted_by="Technician Rogers"
    )
    temp_id = sub_res["temp_id"]

    success = temp_db.reject_pending_fix(
        temp_id=temp_id,
        admin_name="Airport Administrator",
        reason="Vague description; lacks part number and AMM reference"
    )
    assert success is True

    # Check pending list
    pending = temp_db.get_pending_fixes(status_filter="pending")
    assert not any(p["temp_id"] == temp_id for p in pending)

    all_items = temp_db.get_pending_fixes(status_filter="all")
    rejected = [p for p in all_items if p["temp_id"] == temp_id][0]
    assert rejected["status"] == "rejected"
    assert "Vague description" in rejected["admin_notes"]


def test_query_logging(temp_db):
    """Verifies that queries made by technicians are logged for Admin audit view."""
    temp_db.log_query(
        user_name="Technician Marcus",
        user_role="technician",
        query="rudder trim actuator travel limit",
        evidence_status="sufficient",
        evidence_reason="Found 3 corroborating records",
        cited_record_ids=["100010", "100015"],
        execution_time_ms=42.5
    )

    logs = temp_db.get_query_logs(limit=10)
    assert len(logs) >= 1
    latest = logs[0]
    assert latest["user_name"] == "Technician Marcus"
    assert latest["user_role"] == "technician"
    assert latest["query"] == "rudder trim actuator travel limit"
    assert "100010" in latest["cited_record_ids"]


def test_dynamic_indexing_and_agent_retrieval():
    """
    Verifies that when an approved record is added via tools.add_record,
    hybrid retrieval can immediately retrieve and rank the new record.
    """
    tools = MaintenanceTools()

    # Create synthetic unique record
    unique_term = "ZYXW_SPECIAL_ACTUATOR_VIBRATION"
    new_record = LogbookRecord(
        record_id="109999",
        problem_raw=f"Severe oscillation in {unique_term} during rollout",
        action_raw=f"Replaced {unique_term} servo valve assembly per AMM 27-21-04",
        problem_normalized=f"severe oscillation in {unique_term.lower()} during rollout",
        action_normalized=f"replaced {unique_term.lower()} servo valve assembly per amm 27-21-04"
    )

    # Register dynamically
    tools.add_record(new_record)

    # Fast O(1) lookup check
    res = tools.get_record("109999")
    assert res.found is True
    assert res.record.record_id == "109999"

    # Hybrid search retrieval check
    search_out = tools.search_records(query=unique_term, top_k=3)
    assert search_out.total_found > 0
    top_ids = [r.record_id for r in search_out.records]
    assert "109999" in top_ids


def test_api_endpoints_workflow():
    """Tests the FastAPI routes for authentication, user creation, submission, pending queue, approval, and audit."""
    client = TestClient(app)

    # 1. Login endpoint
    login_res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert login_res.status_code == 200
    assert login_res.json()["success"] is True

    # 2. Stats endpoint
    stats_res = client.get("/api/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["total_technicians"] >= 3

    # 3. Admin creates new technician account
    create_res = client.post("/api/admin/users", json={
        "username": "tech_hannah",
        "password": "mypassword1",
        "full_name": "Hannah Abbott",
        "role": "technician"
    })
    assert create_res.status_code == 200
    assert create_res.json()["success"] is True

    # 4. List users
    users_res = client.get("/api/admin/users")
    assert users_res.status_code == 200
    assert any(u["username"] == "tech_hannah" for u in users_res.json())

    # 5. Submit fix
    submit_resp = client.post("/api/records/submit", json={
        "problem_raw": "ELEVATOR FEEL COMPUTER CAUTION LIGHT ILLUMINATED",
        "action_raw": "REPLACED ELEVATOR FEEL AND CENTERING UNIT PER AMM 27-31-00.",
        "submitted_by": "Hannah Abbott"
    })
    assert submit_resp.status_code == 200
    data = submit_resp.json()
    assert data["status"] == "pending"
    temp_id = data["temp_id"]

    # 6. Admin retrieves pending queue
    queue_resp = client.get("/api/admin/pending?status_filter=pending")
    assert queue_resp.status_code == 200
    queue = queue_resp.json()
    assert any(item["temp_id"] == temp_id for item in queue)

    # 7. Admin approves fix
    approve_resp = client.post(f"/api/admin/records/{temp_id}/approve", json={
        "admin_name": "Station Chief Miller"
    })
    assert approve_resp.status_code == 200
    approve_data = approve_resp.json()
    assert approve_data["status"] == "approved"
    assigned_id = approve_data["record_id"]

    # 8. Immediate retrieval of newly approved record via GET /api/records/{id}
    rec_resp = client.get(f"/api/records/{assigned_id}")
    assert rec_resp.status_code == 200
    assert rec_resp.json()["record_id"] == assigned_id

    # 9. Query chat endpoint with technician user info
    chat_resp = client.post("/api/chat", json={
        "query": "elevator feel computer caution light",
        "user_name": "Hannah Abbott",
        "user_role": "technician"
    })
    assert chat_resp.status_code == 200

    # 10. Admin audit logs endpoint
    audit_resp = client.get("/api/admin/audit-logs?limit=20")
    assert audit_resp.status_code == 200
    audit_items = audit_resp.json()
    assert len(audit_items) > 0
    assert any(log["user_name"] == "Hannah Abbott" for log in audit_items)

    # 11. Technician views their own query logs
    tech_logs_resp = client.get("/api/user/query-logs?user_name=Hannah+Abbott")
    assert tech_logs_resp.status_code == 200
    tech_logs = tech_logs_resp.json()
    assert len(tech_logs) > 0
    assert all(log["user_name"] == "Hannah Abbott" for log in tech_logs)

    # 12. Admin deletes the technician account
    hannah_user = [u for u in users_res.json() if u["username"] == "tech_hannah"][0]
    del_resp = client.delete(f"/api/admin/users/{hannah_user['id']}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True
