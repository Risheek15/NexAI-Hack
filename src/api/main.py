"""
FastAPI Backend API for Maintenance Copilot MVP.
Provides endpoints for:
1. Health checking and record inspection
2. Agentic chat interactions with deterministic citation gating
3. Authentication and user accounts (Technicians & Airport Administrators)
4. Technician maintenance fix submissions into staging SQLite queue
5. Airport Administrator governance (approving/rejecting fixes, creating technician accounts)
6. Real-time audit logs of technician queries
"""

import os
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from src.models import LogbookRecord
from src.abbreviation.models import Glossary
from src.agent.tools import MaintenanceTools
from src.agent.core import MaintenanceAgent, AgentResponse
from src.evidence.policy import audit_evidence_policy, check_safety_disclaimer_present
from src.evidence.refusal import format_abstention_response, format_clarification_response, SAFETY_DISCLAIMER
from src.db.database import DatabaseManager
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationDetail,
    ToolTraceItem,
    PolicyAuditSummary,
    HealthResponse,
    RecordResponse,
    FixSubmissionRequest,
    FixSubmissionResponse,
    PendingFixItem,
    ApproveFixRequest,
    RejectFixRequest,
    QueryLogItem,
    LoginRequest,
    LoginResponse,
    CreateUserRequest,
    UserItem,
    StatsResponse
)

# Global singleton resources initialized during app lifespan
resources: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-warms hybrid retrieval indexes, embedding vectors, the SQLite database manager,
    and the MaintenanceAgent to ensure zero cold-start latency for user requests.
    """
    print("[INFO] Starting Maintenance Copilot API server...")
    t0 = time.perf_counter()

    tools = MaintenanceTools()
    agent = MaintenanceAgent(tools=tools)
    glossary = Glossary.load_json("data/glossary.json")

    # Initialize SQLite database and seed records if not yet present
    db = DatabaseManager()
    seeded_count = db.seed_from_jsonl("data/processed/records.jsonl")
    print(f"[INFO] SQLite database initialized at data/copilot.db (Records: {seeded_count}).")

    resources["tools"] = tools
    resources["agent"] = agent
    resources["glossary"] = glossary
    resources["db"] = db

    elapsed = time.perf_counter() - t0
    print(f"[SUCCESS] Resources warmed up in {elapsed:.2f}s. Server ready.")
    yield
    print("[INFO] Shutting down Maintenance Copilot API server...")
    resources.clear()


app = FastAPI(
    title="Maintenance Copilot API",
    description="Agentic RAG over aviation maintenance logbooks with citation validation and evidence policy enforcement.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local development and Next.js frontend
origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [orig.strip() for orig in origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> DatabaseManager:
    db = resources.get("db")
    if not db:
        db = DatabaseManager()
        db.seed_from_jsonl("data/processed/records.jsonl")
        resources["db"] = db
    return db


def get_tools() -> MaintenanceTools:
    tools = resources.get("tools")
    if not tools:
        tools = MaintenanceTools()
        resources["tools"] = tools
    return tools


def get_agent() -> MaintenanceAgent:
    agent = resources.get("agent")
    if not agent:
        tools = get_tools()
        agent = MaintenanceAgent(tools=tools)
        resources["agent"] = agent
    return agent


# --- Authentication Endpoints ---

@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
@app.post("/api/auth/login", response_model=LoginResponse, tags=["Auth"])
def login(req: LoginRequest):
    """Authenticates username and password against the SQLite database."""
    db = get_db()
    user = db.authenticate_user(username=req.username, password=req.password)
    if not user:
        return LoginResponse(
            success=False,
            message="Invalid username or password.",
            user=None
        )
    return LoginResponse(
        success=True,
        message=f"Welcome back, {user['full_name']}!",
        user=user
    )


# --- System Stats Endpoint ---

@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
@app.get("/api/stats", response_model=StatsResponse, tags=["Stats"])
def get_system_stats():
    """Returns dynamic counts of records, pending fixes, registered technicians, and queries."""
    db = get_db()
    return db.get_stats()


# --- User Management (Airport Administrator) ---

@app.get("/admin/users", response_model=List[UserItem], tags=["Admin"])
@app.get("/api/admin/users", response_model=List[UserItem], tags=["Admin"])
def list_users(role: Optional[str] = None):
    """Airport Administrator views all registered technician and administrator accounts."""
    db = get_db()
    return db.get_users(role_filter=role)


@app.post("/admin/users", tags=["Admin"])
@app.post("/api/admin/users", tags=["Admin"])
def create_technician_account(req: CreateUserRequest):
    """Airport Administrator creates a new technician or administrator account."""
    db = get_db()
    success, msg, new_user = db.create_user(
        username=req.username,
        password=req.password,
        full_name=req.full_name,
        role=req.role or "technician"
    )
    if not success or not new_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    return {
        "success": True,
        "message": msg,
        "user": new_user
    }


@app.delete("/admin/users/{user_id}", tags=["Admin"])
@app.delete("/api/admin/users/{user_id}", tags=["Admin"])
def delete_user_account(user_id: int):
    """Airport Administrator deletes a technician account."""
    db = get_db()
    success, msg = db.delete_user(user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    return {"success": True, "message": msg}


# --- Health & Status ---

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns system status, dynamic record counts, and Gemini configuration readiness."""
    tools = get_tools()
    agent = get_agent()
    db = get_db()
    glossary: Glossary = resources.get("glossary") or Glossary.load_json("data/glossary.json")

    total_records = db.get_stats()["total_records"]
    total_glossary = len(glossary.entries)
    model_name = agent.model_name
    api_key_set = bool(agent.api_key and agent.api_key != "your_gemini_api_key_here")

    return HealthResponse(
        status="healthy",
        total_records=total_records,
        total_glossary_terms=total_glossary,
        model_name=model_name,
        api_key_configured=api_key_set,
        environment=os.environ.get("ENVIRONMENT", "development")
    )


# --- Agentic Chat & Search ---

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
@app.post("/api/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """
    Executes bounded agentic query loop over maintenance logbooks.
    Performs citation validation, evidence gating, policy auditing, and query logging.
    """
    agent = get_agent()
    tools = get_tools()
    db = get_db()

    user_name = request.user_name or "Technician"
    user_role = request.user_role or "technician"

    # 1. Execute agent ask loop
    agent_resp: AgentResponse = agent.ask(request.query)

    # 2. Check evidence status and apply refusal policy if needed
    final_answer = agent_resp.answer
    if agent_resp.evidence_status == "insufficient":
        lower_ans = final_answer.lower()
        if not any(w in lower_ans for w in ["insufficient", "cannot support", "no matching", "unavailable", "abstain"]):
            final_answer = format_abstention_response(
                query=request.query,
                reason=agent_resp.evidence_reason,
                closest_records=agent_resp.retrieved_records
            )
    elif agent_resp.evidence_status == "needs_clarification":
        lower_ans = final_answer.lower()
        if not any(w in lower_ans for w in ["clarify", "specify", "too generic", "which component"]):
            final_answer = format_clarification_response(
                query=request.query,
                reason=agent_resp.evidence_reason
            )

    # 3. Ensure mandatory safety disclaimer is present
    if not check_safety_disclaimer_present(final_answer):
        final_answer += SAFETY_DISCLAIMER

    # 4. Audit evidence policy
    policy_audit = audit_evidence_policy(
        query=request.query,
        answer_text=final_answer,
        evidence_status=agent_resp.evidence_status,
        retrieved_records=agent_resp.retrieved_records,
        cited_record_ids=agent_resp.cited_record_ids,
        is_fallback=agent_resp.is_fallback
    )

    # 5. Build structured citation details for the UI
    citations: List[CitationDetail] = []
    for rid in agent_resp.cited_record_ids:
        rec = tools.lookup.get(rid)
        if rec:
            p_raw = rec.get("problem_raw", "") if isinstance(rec, dict) else getattr(rec, "problem_raw", "")
            a_raw = rec.get("action_raw", "") if isinstance(rec, dict) else getattr(rec, "action_raw", "")
            citations.append(
                CitationDetail(
                    record_id=rid,
                    problem_raw=p_raw,
                    action_raw=a_raw
                )
            )

    # 6. Build structured tool trace
    tool_trace: List[ToolTraceItem] = []
    for tc in agent_resp.trace.tool_calls:
        tool_trace.append(
            ToolTraceItem(
                tool_name=tc.tool_name,
                arguments=tc.arguments,
                output_summary=tc.output_summary,
                execution_time_ms=tc.execution_time_ms,
                is_error=tc.is_error
            )
        )

    # 7. Audit log the query for Admin visibility
    try:
        db.log_query(
            user_name=user_name,
            user_role=user_role,
            query=request.query,
            evidence_status=agent_resp.evidence_status,
            evidence_reason=agent_resp.evidence_reason,
            cited_record_ids=agent_resp.cited_record_ids,
            execution_time_ms=agent_resp.trace.total_execution_time_ms
        )
    except Exception as e:
        print(f"[WARNING] Failed to record query log in database: {e}")

    return ChatResponse(
        query=request.query,
        answer=final_answer,
        evidence_status=agent_resp.evidence_status,
        evidence_reason=agent_resp.evidence_reason,
        cited_record_ids=agent_resp.cited_record_ids,
        citations=citations,
        tool_trace=tool_trace,
        total_execution_time_ms=agent_resp.trace.total_execution_time_ms,
        policy_audit=PolicyAuditSummary(
            is_compliant=policy_audit.is_compliant,
            has_safety_disclaimer=policy_audit.has_safety_disclaimer,
            conflicting_evidence_notes=policy_audit.conflicting_evidence_notes,
            potential_causation_fallacies=policy_audit.potential_causation_fallacies,
            policy_summary=policy_audit.policy_summary
        ),
        is_fallback=agent_resp.is_fallback,
        safety_disclaimer=SAFETY_DISCLAIMER.strip()
    )


# --- Technician Fix Submission Endpoint ---

@app.post("/records/submit", response_model=FixSubmissionResponse, tags=["Submissions"])
@app.post("/api/records/submit", response_model=FixSubmissionResponse, tags=["Submissions"])
def submit_fix(req: FixSubmissionRequest):
    """
    Technicians submit newly observed squawks and corrective actions.
    The fix is queued in SQLite pending_records awaiting Airport Administrator approval.
    """
    db = get_db()
    result = db.submit_pending_fix(
        problem_raw=req.problem_raw,
        action_raw=req.action_raw,
        submitted_by=req.submitted_by or "Technician"
    )
    return FixSubmissionResponse(
        status="pending",
        message="Maintenance fix request submitted. Queued for Airport Administrator review.",
        temp_id=result["temp_id"],
        item=result
    )


def update_glossary_from_record(record_id: str, problem_raw: str, action_raw: str) -> List[str]:
    """
    Scans newly approved logbook record text, updates frequency of known acronyms,
    and learns new abbreviations dynamically in real time from live maintenance logs.
    """
    from src.abbreviation.models import Glossary, GlossaryEntry
    from src.abbreviation.detector import _ABBREVIATION_PATTERN, _EXCLUDED_TOKENS

    glossary: Optional[Glossary] = resources.get("glossary")
    if not glossary:
        try:
            glossary = Glossary.load_json("data/glossary.json")
            resources["glossary"] = glossary
        except Exception:
            return []

    combined_text = f"{problem_raw} {action_raw}".upper()
    tokens = set(_ABBREVIATION_PATTERN.findall(combined_text))
    learned_terms = []

    for term in tokens:
        clean_term = term.strip().upper()
        if clean_term in _EXCLUDED_TOKENS or len(clean_term) < 2:
            continue

        if clean_term in glossary.entries:
            entry = glossary.entries[clean_term]
            entry.frequency += 1
            sample = f"[{record_id}] {problem_raw} -> {action_raw}"
            if len(entry.context_examples) < 10 and sample not in entry.context_examples:
                entry.context_examples.append(sample)
        else:
            # Dynamically learn new acronym from logbook entry
            sample = f"[{record_id}] {problem_raw} -> {action_raw}"
            glossary.entries[clean_term] = GlossaryEntry(
                term=clean_term,
                expansion=f"Aviation logbook acronym: {clean_term}",
                status="validated",
                context_examples=[sample],
                frequency=1,
                source="live-logbook-learning",
                notes=f"Learned dynamically from approved logbook record #{record_id}"
            )
            learned_terms.append(clean_term)

    # Persist updated glossary
    try:
        glossary.save_json("data/glossary.json")
    except Exception as e:
        print(f"[WARNING] Failed to save updated glossary: {e}")

    return learned_terms


# --- Airport Administrator Governance Endpoints ---

@app.get("/admin/pending", response_model=List[PendingFixItem], tags=["Admin"])
@app.get("/api/admin/pending", response_model=List[PendingFixItem], tags=["Admin"])
def get_pending_fixes(status_filter: str = Query(default="pending", pattern="^(pending|approved|rejected|all)$")):
    """Airport Administrators inspect pending or historical fix requests submitted by technicians."""
    db = get_db()
    return db.get_pending_fixes(status_filter=status_filter)


@app.post("/admin/records/{temp_id}/approve", tags=["Admin"])
@app.post("/api/admin/records/{temp_id}/approve", tags=["Admin"])
def approve_fix(temp_id: str, req: Optional[ApproveFixRequest] = None):
    """
    Airport Administrator approves a technician's maintenance fix.
    Assigns the next sequential 6-digit ID (e.g. 106170), saves to SQLite `records` table,
    and dynamically indexes the new record into memory so future queries find it immediately.
    """
    db = get_db()
    tools = get_tools()
    admin_name = req.admin_name if req and req.admin_name else "Airport Administrator"

    success, new_id, approved_data = db.approve_pending_fix(temp_id=temp_id, admin_name=admin_name)
    if not success or not approved_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pending fix request '{temp_id}' not found or already processed."
        )

    # Dynamically re-index in-memory tools and retrieval engines
    new_record = LogbookRecord(
        record_id=new_id,
        problem_raw=approved_data["problem_raw"],
        action_raw=approved_data["action_raw"],
        problem_normalized=approved_data["problem_normalized"],
        action_normalized=approved_data["action_normalized"]
    )
    tools.add_record(new_record)

    # Dynamically learn & update acronym glossary in real time
    learned_terms = update_glossary_from_record(
        record_id=new_id,
        problem_raw=approved_data["problem_raw"],
        action_raw=approved_data["action_raw"]
    )

    return {
        "status": "approved",
        "message": f"Fix successfully approved, assigned official ID #{new_id}, and indexed into the active fleet logbook.",
        "record_id": new_id,
        "record": approved_data,
        "learned_terms": learned_terms
    }


@app.post("/admin/records/{temp_id}/reject", tags=["Admin"])
@app.post("/api/admin/records/{temp_id}/reject", tags=["Admin"])
def reject_fix(temp_id: str, req: Optional[RejectFixRequest] = None):
    """Airport Administrator rejects a submitted fix with an audit reason."""
    db = get_db()
    admin_name = req.admin_name if req and req.admin_name else "Airport Administrator"
    reason = req.reason if req and req.reason else "Insufficient documentation or duplicate"

    success = db.reject_pending_fix(temp_id=temp_id, admin_name=admin_name, reason=reason)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pending fix request '{temp_id}' not found or already processed."
        )

    return {
        "status": "rejected",
        "message": f"Pending fix request '{temp_id}' was rejected.",
        "temp_id": temp_id
    }


@app.get("/admin/audit-logs", response_model=List[QueryLogItem], tags=["Admin"])
@app.get("/api/admin/audit-logs", response_model=List[QueryLogItem], tags=["Admin"])
def get_audit_logs(limit: int = Query(default=50, ge=1, le=200), user_name: Optional[str] = None):
    """Airport Administrator views real-time audit ledger of all queries made by technicians."""
    db = get_db()
    return db.get_query_logs(limit=limit, user_name=user_name)


@app.get("/user/query-logs", response_model=List[QueryLogItem], tags=["User"])
@app.get("/api/user/query-logs", response_model=List[QueryLogItem], tags=["User"])
def get_user_query_logs(user_name: str = Query(...), limit: int = Query(default=50, ge=1, le=100)):
    """Technicians view their own past query history and audit trail."""
    db = get_db()
    return db.get_query_logs(limit=limit, user_name=user_name)


# --- Logbook Inspection & Glossary Endpoints ---

@app.get("/records/{record_id}", response_model=RecordResponse, tags=["Records"])
@app.get("/api/records/{record_id}", response_model=RecordResponse, tags=["Records"])
def get_record(record_id: str):
    """Retrieves original and normalized logbook entry for a specific 6-digit ID."""
    tools = get_tools()
    out = tools.get_record(record_id)
    if not out.found or not out.record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record [ID: {record_id}] does not exist in the source store."
        )

    return RecordResponse(
        found=True,
        record_id=out.record.record_id,
        problem_raw=out.record.problem_raw,
        action_raw=out.record.action_raw,
        problem_normalized=out.record.problem_normalized,
        action_normalized=out.record.action_normalized
    )


@app.get("/glossary", tags=["Glossary"])
@app.get("/api/glossary", tags=["Glossary"])
def get_glossary():
    """Returns validated aviation terms and expansions for UI tooltips and search helpers."""
    glossary: Glossary = resources.get("glossary") or Glossary.load_json("data/glossary.json")
    return {
        term: {
            "expansion": entry.expansion,
            "status": entry.status,
            "frequency": entry.frequency,
            "sample": entry.context_examples[0] if entry.context_examples else ""
        }
        for term, entry in glossary.entries.items()
        if entry.status == "validated"
    }
