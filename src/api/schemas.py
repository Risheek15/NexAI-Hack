"""
Pydantic schemas and DTOs for the Maintenance Copilot REST API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Maintenance squawk or natural language discrepancy query.")
    top_k: Optional[int] = Field(default=5, ge=1, le=10, description="Max candidate records to retrieve (1-10).")
    user_name: Optional[str] = Field(default="Technician", description="Name of the technician or airport administrator asking the query.")
    user_role: Optional[str] = Field(default="technician", description="Role: 'technician' or 'admin'.")


class CitationDetail(BaseModel):
    record_id: str
    problem_raw: str
    action_raw: str
    rank: Optional[int] = None
    date: Optional[str] = None


class ToolTraceItem(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str
    execution_time_ms: float
    is_error: bool = False


class PolicyAuditSummary(BaseModel):
    is_compliant: bool
    has_safety_disclaimer: bool
    is_prescriptive_query: bool = False
    prescriptive_boundary_applied: bool = False
    conflicting_evidence_notes: List[str] = Field(default_factory=list)
    potential_causation_fallacies: List[str] = Field(default_factory=list)
    policy_summary: str


class CitationValidationSummary(BaseModel):
    is_valid: bool = True
    cited_ids: List[str] = Field(default_factory=list)
    invalid_nonexistent_ids: List[str] = Field(default_factory=list)
    unretrieved_ids: List[str] = Field(default_factory=list)
    invalid_quotes: List[str] = Field(default_factory=list)
    cross_record_attributions: List[str] = Field(default_factory=list)
    warning_messages: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    query: str
    answer: str
    evidence_status: str  # "sufficient" | "insufficient" | "needs_clarification"
    evidence_reason: str
    cited_record_ids: List[str] = Field(default_factory=list)
    citations: List[CitationDetail] = Field(default_factory=list)
    citation_validation: Optional[CitationValidationSummary] = None
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


# --- Role Governance & Database Schemas ---

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=3, max_length=100)


class LoginResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=100)
    role: Optional[str] = Field(default="technician", description="Role: 'technician' or 'admin'")


class UserItem(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    created_at: str
    last_active: Optional[str] = None


class StatsResponse(BaseModel):
    total_records: int
    pending_fixes: int
    total_technicians: int
    total_queries: int


class FixSubmissionRequest(BaseModel):
    problem_raw: str = Field(..., min_length=3, max_length=1000, description="Observed discrepancy or squawk.")
    action_raw: str = Field(..., min_length=3, max_length=1000, description="Corrective maintenance action taken.")
    submitted_by: Optional[str] = Field(default="Technician", description="Technician's name or badge.")


class FixSubmissionResponse(BaseModel):
    status: str
    message: str
    temp_id: str
    item: Dict[str, Any]


class PendingFixItem(BaseModel):
    id: int
    temp_id: str
    problem_raw: str
    action_raw: str
    problem_normalized: str
    action_normalized: str
    submitted_by: str
    submitted_at: str
    status: str
    admin_notes: Optional[str] = ""
    approved_record_id: Optional[str] = None


class ApproveFixRequest(BaseModel):
    admin_name: Optional[str] = Field(default="Airport Administrator", description="Administrator approving the entry.")


class RejectFixRequest(BaseModel):
    admin_name: Optional[str] = Field(default="Airport Administrator", description="Administrator rejecting the entry.")
    reason: Optional[str] = Field(default="Insufficient documentation or duplicate squawk", description="Rejection rationale.")


class QueryLogItem(BaseModel):
    id: int
    user_name: str
    user_role: str
    query: str
    evidence_status: str
    evidence_reason: str
    cited_record_ids: List[str]
    execution_time_ms: float
    timestamp: str
