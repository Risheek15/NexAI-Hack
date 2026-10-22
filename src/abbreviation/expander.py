"""
Conservative abbreviation expander.
Applies ONLY validated glossary entries to normalized text representations.
Leaves original raw text and unvalidated/ambiguous terms completely intact.
"""

import re
from typing import Dict, Optional
from src.models import LogbookRecord
from src.abbreviation.models import Glossary


def expand_text_with_glossary(text: str, glossary: Glossary) -> str:
    """
    Expands abbreviations in text using ONLY validated glossary entries.
    All terms with status 'needs_review', 'ambiguous', or 'unknown' are preserved as-is.
    """
    if not text:
        return ""

    validated_map = glossary.get_validated_map()
    if not validated_map:
        return text

    expanded = text
    # Sort by length descending so longer acronyms are replaced before sub-tokens
    for term, expansion in sorted(validated_map.items(), key=lambda x: len(x[0]), reverse=True):
        # Word boundary substitution: e.g. \bCYL\b -> cylinder
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        expanded = pattern.sub(expansion, expanded)

    return expanded


def expand_record(record: LogbookRecord, glossary: Glossary) -> LogbookRecord:
    """
    Creates an updated LogbookRecord where normalized representations incorporate
    validated abbreviation expansions. Original raw text remains 100% immutable.
    """
    expanded_problem = expand_text_with_glossary(record.problem_normalized, glossary)
    expanded_action = expand_text_with_glossary(record.action_normalized, glossary)

    return LogbookRecord(
        record_id=record.record_id,
        problem_raw=record.problem_raw,
        action_raw=record.action_raw,
        problem_normalized=expanded_problem,
        action_normalized=expanded_action,
        metadata=dict(record.metadata)
    )
