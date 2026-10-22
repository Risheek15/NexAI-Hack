"""
Pydantic schemas and DTOs for the Maintenance Copilot REST API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Maintenance squawk or natural language discrepancy query.")
    top_k: Optional[int] = Field(default=5, ge=1, le=10, description="Max candidate records to retrieve (1-10).")


class CitationDetail(BaseModel):
    record_id: str
    problem_raw: str
    action_raw: str
    rank: Optional[int] = None


class ToolTraceItem(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str
    execution_time_ms: float
    is_error: bool = False


class PolicyAuditSummary(BaseModel):
    is_compliant: bool
    has_safety_disclaimer: bool
    conflicting_evidence_notes: List[str] = Field(default_factory=list)
    potential_causation_fallacies: List[str] = Field(default_factory=list)
    policy_summary: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    evidence_status: str  # "sufficient" | "insufficient" | "needs_clarification"
    evidence_reason: str
    cited_record_ids: List[str] = Field(default_factory=list)
    citations: List[CitationDetail] = Field(default_factory=list)
    tool_trace: List[ToolTraceItem] = Field(default_factory=list)
    total_execution_time_ms: float
    policy_audit: PolicyAuditSummary
    is_fallback: bool
    safety_disclaimer: str


class HealthResponse(BaseModel):
    status: str
    total_records: int
    total_glossary_terms: int
    model_name: str
    api_key_configured: bool
    environment: str


class RecordResponse(BaseModel):
    found: bool
    record_id: str
    problem_raw: Optional[str] = None
    action_raw: Optional[str] = None
    problem_normalized: Optional[str] = None
    action_normalized: Optional[str] = None
    error: Optional[str] = None
