"""
Structured Execution Trace Logger.
Records every tool call, argument, execution time, and result for transparent UI display and debugging.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ToolCallTrace(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    output_summary: str
    output_data: Dict[str, Any]
    execution_time_ms: float
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_error: bool = False
    error_message: Optional[str] = None


class AgentTrace(BaseModel):
    query: str
    tool_calls: List[ToolCallTrace] = Field(default_factory=list)
    retrieved_record_ids: List[str] = Field(default_factory=list)
    inspected_record_ids: List[str] = Field(default_factory=list)
    total_execution_time_ms: float = 0.0
    error: Optional[str] = None

    def add_tool_call(self, trace: ToolCallTrace) -> None:
        self.tool_calls.append(trace)
        if trace.tool_name == "search_records":
            records = trace.output_data.get("records", [])
            for r in records:
                rid = r.get("record_id")
                if rid and rid not in self.retrieved_record_ids:
                    self.retrieved_record_ids.append(rid)
        elif trace.tool_name == "get_record":
            rid = trace.output_data.get("record_id")
            if rid and rid not in self.inspected_record_ids:
                self.inspected_record_ids.append(rid)
