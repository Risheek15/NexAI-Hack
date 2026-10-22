"""
Integration and functional tests for Phase 6: FastAPI Backend.
Uses FastAPI TestClient to test health check, chat execution, record lookup,
error handling, and CORS headers.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped test client that runs the app lifespan once."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthAndGlossaryEndpoints:
    """Tests for system health check and glossary terms."""

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["total_records"] >= 6169
        assert data["total_glossary_terms"] >= 20
        assert "model_name" in data
        assert "api_key_configured" in data

    def test_glossary_endpoint(self, client):
        resp = client.get("/api/glossary")
        assert resp.status_code == 200
        data = resp.json()
        assert "CYL" in data
        assert data["CYL"]["expansion"] == "cylinder"
        assert "L/H" in data
        assert data["L/H"]["expansion"] == "left-hand"


class TestRecordLookupEndpoint:
    """Tests for direct 6-digit record lookup."""

    def test_record_lookup_known_id(self, client):
        resp = client.get("/api/records/100004")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["record_id"] == "100004"
        assert "CYL 2 LOWER PLUG FOULED" in data["action_raw"]
        assert data["problem_raw"] is not None

    def test_record_lookup_unknown_id(self, client):
        resp = client.get("/api/records/999999")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data
        assert "does not exist" in data["detail"]


class TestChatEndpoint:
    """Tests for the agentic chat endpoint with evidence policy and citation validation."""

    def test_chat_empty_query_validation(self, client):
        # Query with empty string must fail FastAPI/Pydantic validation
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_chat_sufficient_query(self, client):
        resp = client.post("/api/chat", json={"query": "fuel servo replacement", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "fuel servo replacement"
        assert len(data["answer"]) > 0
        assert data["evidence_status"] in ["sufficient", "insufficient"]
        assert "policy_audit" in data
        assert data["policy_audit"]["has_safety_disclaimer"] is True
        assert len(data["tool_trace"]) >= 1
        assert "safety_disclaimer" in data

    def test_chat_out_of_domain_abstention(self, client):
        resp = client.post("/api/chat", json={"query": "Boeing 777 autopilot computer failure", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["evidence_status"] == "insufficient"
        assert "insufficient" in data["answer"].lower() or "cannot support" in data["answer"].lower()
        assert data["policy_audit"]["has_safety_disclaimer"] is True

    def test_chat_ambiguous_clarification(self, client):
        resp = client.post("/api/chat", json={"query": "leak", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["evidence_status"] == "needs_clarification"
        assert any(w in data["answer"].lower() for w in ["clarify", "clarification", "specify", "which component"])


class TestCORSConfiguration:
    """Tests CORS middleware headers for Next.js frontend origin."""

    def test_cors_headers_on_options(self, client):
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        resp = client.options("/api/chat", headers=headers)
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
