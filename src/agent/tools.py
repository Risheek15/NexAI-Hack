import re
"""
Application tools for the Maintenance Copilot agent:
1. search_records(query, top_k): Hybrid retrieval, duplicate clustering, deterministic evidence gating.
2. get_record(record_id): Fast O(1) exact lookup by genuine record ID.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from src.models import LogbookRecord
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.evidence_gate import assess_retrieval_evidence, EvidenceStatus


class SearchRecordsInput(BaseModel):
    query: str = Field(..., description="The maintenance discrepancy, squawk, symptom, or repair action to search for")
    top_k: int = Field(default=8, ge=1, le=20, description="Maximum number of ranked matching records to return (1-20)")

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Query cannot be empty or blank")
        return s


class RecordSummary(BaseModel):
    record_id: str
    rank: int
    score: float
    problem_raw: str
    action_raw: str
    problem_normalized: str
    action_normalized: str
    is_duplicate_of: Optional[str] = None
    date: str = "2026-10-22"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRecordsOutput(BaseModel):
    query: str
    evidence_status: EvidenceStatus
    evidence_reason: str
    total_found: int
    unique_count: int
    records: List[RecordSummary]
    duplicate_clusters: Dict[str, List[str]] = Field(default_factory=dict)
    aspect_coverage: Dict[str, bool] = Field(default_factory=dict)
    unsupported_aspects: List[str] = Field(default_factory=list)


class GetRecordInput(BaseModel):
    record_id: str = Field(..., description="The exact source record ID (e.g. '100004')")

    @field_validator("record_id")
    @classmethod
    def clean_record_id(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("record_id cannot be empty")
        return s


class GetRecordOutput(BaseModel):
    record_id: str
    found: bool
    record: Optional[RecordSummary] = None
    error: Optional[str] = None


class MaintenanceTools:
    """
    Stateful tool manager holding shared retrieval and lookup indices.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        lookup_path: str | Path = "data/processed/records_lookup.json",
        records_jsonl: str | Path = "data/processed/records.jsonl"
    ):
        self.lookup_path = Path(lookup_path)
        self.records_jsonl = Path(records_jsonl)
        self._lookup: Optional[Dict[str, Dict[str, Any]]] = None
        self._retriever = retriever

    @property
    def lookup(self) -> Dict[str, Dict[str, Any]]:
        """Lazy load O(1) ID-to-record dictionary."""
        if self._lookup is None:
            if not self.lookup_path.exists():
                raise FileNotFoundError(f"Lookup file not found: {self.lookup_path}. Run Phase 1 ingestion first.")
            with open(self.lookup_path, "r", encoding="utf-8") as f:
                self._lookup = json.load(f)
        return self._lookup

    @property
    def retriever(self) -> HybridRetriever:
        """Lazy load HybridRetriever."""
        if self._retriever is None:
            if not self.records_jsonl.exists():
                raise FileNotFoundError(f"Records file not found: {self.records_jsonl}. Run Phase 1 ingestion first.")
            records: List[LogbookRecord] = []
            with open(self.records_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(LogbookRecord(**json.loads(line)))
            self._retriever = HybridRetriever(records=records)
        return self._retriever

    def add_record(self, record: LogbookRecord) -> None:
        """Dynamically registers a newly approved record in the tool lookup and hybrid retriever."""
        _ = self.lookup
        self._lookup[record.record_id] = record.model_dump()
        self.retriever.add_record(record)

    def get_record_date(self, record_id: str) -> str:
        """Retrieves recorded date for a record, defaulting to database approval date."""
        try:
            raw = self.lookup.get(record_id, {})
            meta = raw.get("metadata", {})
            if "date" in meta:
                return str(meta["date"])
            if "approved_at" in meta:
                return str(meta["approved_at"])[:10]
        except Exception:
            pass
        return "2026-10-22"

    def search_records(self, query: str, top_k: int = 8) -> SearchRecordsOutput:
        """
        Executes hybrid retrieval over historical records, groups near-duplicate squawks,
        and provides deterministic evidence assessment.
        """
        # Validate arguments via Pydantic
        try:
            validated = SearchRecordsInput(query=query, top_k=top_k)
        except Exception as e:
            return SearchRecordsOutput(
                query=query,
                evidence_status="needs_clarification",
                evidence_reason=f"Invalid arguments: {e}",
                total_found=0,
                unique_count=0,
                records=[],
                duplicate_clusters={}
            )

        # Clean conversational filler and negative exclusion instructions from search query
        clean_search_query = validated.query
        if len(clean_search_query.split()) > 6:
            prefix_pattern = re.compile(
                r"^(?:find|search|show|get|retrieve|look\s+for|list)\s+(?:all\s+)?(?:historical\s+)?(?:maintenance\s+)?(?:records\s+)?(?:specifically\s+)?(?:mentioning|regarding|about|for|with)\s+",
                re.IGNORECASE
            )
            clean_search_query = prefix_pattern.sub("", clean_search_query).strip()
            exclude_match = re.search(r"\b(?:exclude|do\s+not\s+include)\s+records.*", clean_search_query, re.IGNORECASE)
            if exclude_match:
                clean_search_query = clean_search_query[:exclude_match.start()].strip()
            return_match = re.search(r"\breturn\s+(?:the\s+)?(?:record\s+id|original\s+problem|recorded\s+action).*", clean_search_query, re.IGNORECASE)
            if return_match:
                clean_search_query = clean_search_query[:return_match.start()].strip()

        if not clean_search_query.strip():
            clean_search_query = validated.query

        # Execute hybrid retrieval
        raw_results = self.retriever.search(clean_search_query, top_k=validated.top_k)

        # Run evidence assessment and duplicate clustering
        assessment, deduped_results, clusters = assess_retrieval_evidence(validated.query, raw_results)

        # Build reverse mapping from duplicate record_id to representative
        dup_to_rep = {}
        for rep_id, members in clusters.items():
            for m in members:
                if m != rep_id:
                    dup_to_rep[m] = rep_id

        record_summaries: List[RecordSummary] = []
        for r in raw_results:
            rec_date = self.get_record_date(r.record_id)
            record_summaries.append(RecordSummary(
                record_id=r.record_id,
                rank=r.rank,
                score=r.score,
                problem_raw=r.problem_raw,
                action_raw=r.action_raw,
                problem_normalized=r.problem_normalized,
                action_normalized=r.action_normalized,
                is_duplicate_of=dup_to_rep.get(r.record_id),
                date=rec_date,
                metadata=dict(r.metadata)
            ))

        return SearchRecordsOutput(
            query=validated.query,
            evidence_status=assessment.status,
            evidence_reason=assessment.reason,
            total_found=len(raw_results),
            unique_count=assessment.unique_evidence_count,
            records=record_summaries,
            duplicate_clusters=clusters,
            aspect_coverage=assessment.aspect_coverage,
            unsupported_aspects=assessment.unsupported_aspects
        )

    def get_record(self, record_id: str) -> GetRecordOutput:
        """
        Fetches an exact record by its source ID with O(1) efficiency.
        Handles unknown IDs explicitly without hallucination or silent substitution.
        """
        try:
            validated = GetRecordInput(record_id=record_id)
        except Exception as e:
            return GetRecordOutput(
                record_id=record_id,
                found=False,
                error=f"Invalid record_id: {e}"
            )

        clean_id = validated.record_id
        raw_dict = self.lookup.get(clean_id)

        if not raw_dict:
            return GetRecordOutput(
                record_id=clean_id,
                found=False,
                error=f"Record ID '{clean_id}' does not exist in the historical logbook database."
            )

        rec_date = self.get_record_date(clean_id)
        summary = RecordSummary(
            record_id=clean_id,
            rank=1,
            score=1.0,
            problem_raw=raw_dict.get("problem_raw", ""),
            action_raw=raw_dict.get("action_raw", ""),
            problem_normalized=raw_dict.get("problem_normalized", ""),
            action_normalized=raw_dict.get("action_normalized", ""),
            date=rec_date,
            metadata=raw_dict.get("metadata", {})
        )

        return GetRecordOutput(
            record_id=clean_id,
            found=True,
            record=summary
        )
