"""Data models for aviation maintenance logbook records and ingestion manifests."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class LogbookRecord(BaseModel):
    """
    Internal canonical representation of an aviation maintenance logbook record.
    Maintains strict separation between original immutable source text and
    normalized text representations.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(
        ...,
        description="Unique source record identifier (maps strictly to dataset IDENT)"
    )
    problem_raw: str = Field(
        ...,
        description="Immutable original discrepancy text exactly as recorded in source data"
    )
    action_raw: str = Field(
        ...,
        description="Immutable original corrective action text exactly as recorded in source data"
    )
    problem_normalized: str = Field(
        ...,
        description="Cleaned, whitespace-normalized problem text with technical tokens preserved"
    )
    action_normalized: str = Field(
        ...,
        description="Cleaned, whitespace-normalized corrective action text with technical tokens preserved"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance metadata (e.g. source line index, checksum)"
    )

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("record_id cannot be empty")
        return s

    @property
    def combined_raw(self) -> str:
        """Returns the full raw narrative: problem followed by action."""
        return f"PROBLEM: {self.problem_raw} | ACTION: {self.action_raw}"

    @property
    def combined_normalized(self) -> str:
        """Returns the full normalized text for hybrid retrieval indexing."""
        return f"{self.problem_normalized} {self.action_normalized}".strip()


class IngestionSummary(BaseModel):
    """
    Reproducible manifest documenting ingestion execution, counts, and integrity.
    """
    model_config = ConfigDict(frozen=True)

    source_file: str
    output_file: str
    source_sha256: str
    total_raw_rows: int
    valid_records_ingested: int
    duplicate_ids_count: int
    malformed_rows_count: int
    timestamp_utc: str
    fields_detected: list[str]
