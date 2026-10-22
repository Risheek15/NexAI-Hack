"""
Deterministic Citation & Exact Quotation Validator.
Ensures:
1. Every cited record ID exists in the database.
2. Every cited ID was actually retrieved/inspected during the current query session.
3. Every direct quote matches an exact substring of the specific cited record's raw text.
4. Strictly flags and rejects cross-record quote attributions (quoting Record B while citing Record A).
"""

import re
from typing import Set, Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field

# Matches citations like [ID: 100004], [Record 100004], [100004], Record 100004
_CITATION_RE = re.compile(
    r"(?:\[(?:ID:?|Record:?)?\s*(\d{6})\]|\b(?:Record|ID)\s+(\d{6})\b)",
    re.IGNORECASE
)

# Matches direct quotes like "..."
_QUOTATION_RE = re.compile(r'"([^"\n]{4,})"')


class CitationValidationResult(BaseModel):
    is_valid: bool
    cited_ids: List[str] = Field(default_factory=list)
    invalid_nonexistent_ids: List[str] = Field(default_factory=list)
    unretrieved_ids: List[str] = Field(default_factory=list)
    invalid_quotes: List[str] = Field(default_factory=list)
    cross_record_attributions: List[str] = Field(default_factory=list)
    warning_messages: List[str] = Field(default_factory=list)
    sanitized_answer: str = ""


def extract_cited_ids(text: str) -> List[str]:
    """Extracts all 6-digit record IDs referenced in citations."""
    ids = []
    for match in _CITATION_RE.finditer(text):
        id_val = match.group(1) or match.group(2)
        if id_val and id_val not in ids:
            ids.append(id_val)
    return ids


def extract_direct_quotes(text: str) -> List[str]:
    """Extracts quoted strings that represent claimed exact excerpts."""
    return [q.strip() for q in _QUOTATION_RE.findall(text) if len(q.strip()) >= 5]


def _normalize_ws(s: str) -> str:
    """Normalizes whitespace and casing for reliable substring matching."""
    return " ".join(s.strip().split()).upper()


def extract_attributed_quotes(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extracts direct quotes along with the closest cited record ID they are attributed to.
    Tracks record context in point-wise blocks or preceding sentence citations.
    """
    lines = text.split("\n")
    attributed = []
    current_rec_id: Optional[str] = None

    for line in lines:
        cites_in_line = extract_cited_ids(line)
        if cites_in_line:
            current_rec_id = cites_in_line[0]

        quotes_in_line = extract_direct_quotes(line)
        for q in quotes_in_line:
            attributed.append((q, current_rec_id))

    return attributed


def validate_citations_and_quotes(
    answer_text: str,
    valid_db_ids: Set[str],
    session_retrieved_ids: Set[str],
    records_lookup: Dict[str, Dict[str, Any]]
) -> CitationValidationResult:
    """
    Validates:
    1. Every cited ID exists in the database.
    2. Every cited ID was retrieved/inspected during the current session.
    3. Every direct quote matches an exact substring of the cited source text.
    4. Quoting Record B while citing Record A (cross-record attribution) is flagged as invalid.
    """
    if not answer_text:
        return CitationValidationResult(is_valid=True, sanitized_answer="")

    cited_ids = extract_cited_ids(answer_text)
    invalid_nonexistent = []
    unretrieved = []
    warnings = []

    for cid in cited_ids:
        # Check 1: Existence in database
        if cid not in valid_db_ids:
            invalid_nonexistent.append(cid)
            warnings.append(f"Cited record ID '{cid}' does not exist in the database.")
            continue

        # Check 2: Retrieved in this session
        if cid not in session_retrieved_ids:
            unretrieved.append(cid)
            warnings.append(f"Cited record ID '{cid}' was not retrieved in this session.")

    # Check 3: Exact quotation validation & attribution
    invalid_quotes = []
    cross_record_errors = []
    attributed_quotes = extract_attributed_quotes(answer_text)

    # Pre-compute normalized raw texts for lookup
    normalized_record_texts: Dict[str, str] = {}
    for rid in session_retrieved_ids | set(cited_ids):
        raw_prob = records_lookup.get(rid, {}).get("problem_raw", "")
        raw_act = records_lookup.get(rid, {}).get("action_raw", "")
        normalized_record_texts[rid] = _normalize_ws(f"{raw_prob} {raw_act}")

    for quote, attributed_id in attributed_quotes:
        clean_quote = _normalize_ws(quote)

        # If quote is attributed to a specific record ID
        if attributed_id and attributed_id in normalized_record_texts:
            rec_text = normalized_record_texts[attributed_id]
            if clean_quote in rec_text:
                continue
            else:
                # Check if it was falsely borrowed from a different record in the session
                other_matching_ids = [
                    rid for rid, rtext in normalized_record_texts.items()
                    if rid != attributed_id and clean_quote in rtext
                ]
                if other_matching_ids:
                    err_msg = (
                        f"Cross-record quote attribution: quotation \"{quote}\" belongs to Record(s) "
                        f"{', '.join(other_matching_ids)}, but was cited under Record [ID: {attributed_id}]."
                    )
                    cross_record_errors.append(quote)
                    warnings.append(err_msg)
                else:
                    invalid_quotes.append(quote)
                    warnings.append(f"Quotation \"{quote}\" is not an exact substring of cited Record [ID: {attributed_id}].")
        else:
            # No specific ID immediately enclosing: check across all cited records
            matched_in_cited = any(
                clean_quote in normalized_record_texts.get(cid, "")
                for cid in cited_ids if cid in normalized_record_texts
            )
            if not matched_in_cited:
                # Check if it exists in an un-cited retrieved record
                other_retrieved = [
                    rid for rid in session_retrieved_ids
                    if rid not in cited_ids and clean_quote in normalized_record_texts.get(rid, "")
                ]
                if other_retrieved:
                    cross_record_errors.append(quote)
                    warnings.append(
                        f"Quotation \"{quote}\" appears in retrieved Record(s) {', '.join(other_retrieved)}, but those records were not cited."
                    )
                else:
                    invalid_quotes.append(quote)
                    warnings.append(f"Claimed direct quotation \"{quote}\" is not an exact substring of any cited record.")

    is_valid = (
        len(invalid_nonexistent) == 0
        and len(unretrieved) == 0
        and len(invalid_quotes) == 0
        and len(cross_record_errors) == 0
    )

    # Sanitize answer if there are hallucinated IDs
    sanitized = answer_text
    for bad_id in invalid_nonexistent + unretrieved:
        sanitized = re.sub(rf"\[(?:ID:?|Record:?)?\s*{bad_id}\]", f"[INVALID_CITATION_{bad_id}]", sanitized)

    return CitationValidationResult(
        is_valid=is_valid,
        cited_ids=cited_ids,
        invalid_nonexistent_ids=invalid_nonexistent,
        unretrieved_ids=unretrieved,
        invalid_quotes=invalid_quotes,
        cross_record_attributions=cross_record_errors,
        warning_messages=warnings,
        sanitized_answer=sanitized
    )
