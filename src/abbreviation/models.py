"""Data models for abbreviation discovery, validation status, and glossary storage."""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


ValidationStatus = Literal["validated", "needs_review", "ambiguous", "unknown"]


class GlossaryEntry(BaseModel):
    """
    Represents an abbreviation candidate, its proposed expansion, validation status,
    and contextual provenance (surrounding excerpts and source record IDs).
    """
    model_config = ConfigDict(extra="forbid")

    term: str = Field(..., description="The candidate abbreviation term (e.g. CYL, L/H)")
    expansion: str = Field(..., description="The proposed or confirmed expanded text (e.g. cylinder)")
    status: ValidationStatus = Field(
        default="needs_review",
        description="Validation status: 'validated', 'needs_review', 'ambiguous', or 'unknown'"
    )
    context_examples: List[str] = Field(
        default_factory=list,
        description="Deduplicated sample excerpts with source record IDs proving contextual usage"
    )
    alternative_meanings: List[str] = Field(
        default_factory=list,
        description="Alternative interpretations if the term is polysemous or ambiguous"
    )
    frequency: int = Field(
        default=0,
        description="Raw frequency count of candidate occurrences in corpus"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Human reviewer comments or LLM rationale"
    )
    source: str = Field(
        default="auto-discovery",
        description="Origin of expansion ('manual-validation', 'gemini-proposal', 'expert-glossary')"
    )


class Glossary(BaseModel):
    """
    Collection of glossary entries stored as a human-reviewable manifest.
    Enforces conservative application: only entries with status='validated'
    can be used in normalization.
    """
    entries: Dict[str, GlossaryEntry] = Field(
        default_factory=dict,
        description="Mapping from abbreviation term to GlossaryEntry"
    )

    def add_or_update(self, entry: GlossaryEntry) -> None:
        self.entries[entry.term] = entry

    def get_validated_map(self) -> Dict[str, str]:
        """
        Returns a dictionary of ONLY validated abbreviation expansions.
        Unvalidated, ambiguous, or unknown entries are strictly excluded.
        """
        return {
            entry.term: entry.expansion
            for entry in self.entries.values()
            if entry.status == "validated" and entry.expansion.strip()
        }

    def save_json(self, file_path: str) -> None:
        import json
        with open(file_path, "w", encoding="utf-8") as f:
            dump_data = {
                term: entry.model_dump()
                for term, entry in sorted(self.entries.items())
            }
            json.dump(dump_data, f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "Glossary":
        import json
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            return cls(entries={})
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            entries = {
                term: GlossaryEntry(**entry_dict)
                for term, entry_dict in data.items()
            }
            return cls(entries=entries)
