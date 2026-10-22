"""
Deterministic Citation & Exact Quotation Validator.
Ensures every cited record ID exists in the database, was actually retrieved
during the current query session, and that direct quotes are exact substrings of source text.
"""

import re
from typing import Set, Dict, Any, List, Tuple
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
    3. Every direct quote matches an exact substring of the cited records' raw text.
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

    # Check 3: Exact quotation validation
    invalid_quotes = []
    direct_quotes = extract_direct_quotes(answer_text)
    
    # Collect all raw text from legally cited records
    valid_session_cites = [cid for cid in cited_ids if cid in session_retrieved_ids]
    concatenated_raw_text = " ".join([
        f"{records_lookup.get(cid, {}).get('problem_raw', '')} {records_lookup.get(cid, {}).get('action_raw', '')}"
        for cid in valid_session_cites
    ]).upper()

    for quote in direct_quotes:
        # Punctuation / upper check
        clean_quote = quote.upper().strip()
        # If quote claims to be an exact logbook quote but isn't found anywhere in cited records
        if clean_quote not in concatenated_raw_text:
            # Check if it's found in any retrieved record in this session
            all_retrieved_raw = " ".join([
                f"{records_lookup.get(rid, {}).get('problem_raw', '')} {records_lookup.get(rid, {}).get('action_raw', '')}"
                for rid in session_retrieved_ids
            ]).upper()
            if clean_quote not in all_retrieved_raw:
                invalid_quotes.append(quote)
                warnings.append(f"Claimed direct quotation \"{quote}\" is not an exact substring of any retrieved record.")

    is_valid = (len(invalid_nonexistent) == 0 and len(unretrieved) == 0 and len(invalid_quotes) == 0)

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
        warning_messages=warnings,
        sanitized_answer=sanitized
    )
