"""Data models for retrieval results and ranked candidates."""

from typing import Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class RetrievalResult(BaseModel):
    """
    Standardized result for any retrieval mode (BM25, Semantic, or Hybrid).
    Preserves original source text, normalized text, rankings, and provenance.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., description="Source record identifier (IDENT)")
    rank: int = Field(..., ge=1, description="1-indexed retrieval rank")
    score: float = Field(..., description="Retrieval score (BM25 score, cosine similarity, or RRF score)")
    mode: Literal["bm25", "semantic", "hybrid"] = Field(..., description="The mode that produced this ranking")
    problem_raw: str = Field(..., description="Original unedited problem text for citations")
    action_raw: str = Field(..., description="Original unedited action text for citations")
    problem_normalized: str = Field(..., description="Normalized problem text")
    action_normalized: str = Field(..., description="Normalized action text")
    excerpt: str = Field(..., description="Short representative excerpt for UI inspection")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Debugging and fusion metadata (e.g. bm25_rank, semantic_rank, rrf_score)"
    )

    @property
    def combined_raw(self) -> str:
        return f"PROBLEM: {self.problem_raw} | ACTION: {self.action_raw}"
