"""
FastAPI Backend API for Maintenance Copilot MVP.
Provides endpoints for health checking, chat agent interaction, and direct record inspection.
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
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationDetail,
    ToolTraceItem,
    PolicyAuditSummary,
    HealthResponse,
    RecordResponse
)

# Global singleton resources initialized during app lifespan
resources: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-warms hybrid retrieval indexes, embedding vectors, and the MaintenanceAgent
    to ensure zero cold-start latency for user requests.
    """
    print("[INFO] Starting Maintenance Copilot API server...")
    t0 = time.perf_counter()

    tools = MaintenanceTools()
    agent = MaintenanceAgent(tools=tools)

    glossary = Glossary.load_json("data/glossary.json")

    resources["tools"] = tools
    resources["agent"] = agent
    resources["glossary"] = glossary

    elapsed = time.perf_counter() - t0
    print(f"[SUCCESS] Resources warmed up in {elapsed:.2f}s. Server ready.")
    yield
    print("[INFO] Shutting down Maintenance Copilot API server...")
    resources.clear()


app = FastAPI(
    title="Maintenance Copilot API",
    description="Agentic RAG over historical aviation maintenance logbooks with citation validation and evidence policy enforcement.",
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


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns system status, record counts, and Gemini configuration readiness."""
    tools: MaintenanceTools = resources.get("tools")
    agent: MaintenanceAgent = resources.get("agent")
    glossary: Glossary = resources.get("glossary")

    total_records = len(tools.lookup) if tools else 0
    total_glossary = len(glossary.entries) if glossary else 0
    model_name = agent.model_name if agent else os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    api_key_set = bool(agent and agent.api_key and agent.api_key != "your_gemini_api_key_here")

    return HealthResponse(
        status="healthy",
        total_records=total_records,
        total_glossary_terms=total_glossary,
        model_name=model_name,
        api_key_configured=api_key_set,
        environment=os.environ.get("ENVIRONMENT", "development")
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
@app.post("/api/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """
    Executes bounded agentic query loop over maintenance logbooks.
    Performs citation validation, evidence gating, and policy auditing.
    """
    agent: MaintenanceAgent = resources.get("agent")
    tools: MaintenanceTools = resources.get("tools")

    if not agent or not tools:
        # Fallback if accessed before lifespan completed (e.g. some test clients)
        tools = MaintenanceTools()
        agent = MaintenanceAgent(tools=tools)

    # 1. Execute agent ask loop
    agent_resp: AgentResponse = agent.ask(request.query)

    # 2. Check evidence status and apply refusal policy if needed
    final_answer = agent_resp.answer
    if agent_resp.evidence_status == "insufficient":
        # Ensure refusal is clearly formatted if model attempted hallucination
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


@app.get("/records/{record_id}", response_model=RecordResponse, tags=["Records"])
@app.get("/api/records/{record_id}", response_model=RecordResponse, tags=["Records"])
def get_record(record_id: str):
    """Retrieves original and normalized logbook entry for a specific 6-digit ID."""
    tools: MaintenanceTools = resources.get("tools")
    if not tools:
        tools = MaintenanceTools()

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
    glossary: Glossary = resources.get("glossary")
    if not glossary:
        glossary = Glossary.load_json("data/glossary.json")

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
