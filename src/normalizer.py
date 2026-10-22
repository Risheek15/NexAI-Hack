"""
Deterministic text normalization module for aviation maintenance logbooks.
Preserves original text, technical tokens, identifiers, and numerical measurements
while cleaning whitespace, unicode irregularities, and obvious formatting noise.
"""

import re
import unicodedata
from typing import Optional, Dict


# Regex pattern to match irregular whitespace (including non-breaking spaces, tabs, newlines)
_WHITESPACE_RE = re.compile(r"[\s\u00a0\u200b\u200e\u200f]+", re.UNICODE)

# Regex pattern for double/repeated quote noise often found in CSV escapes
_QUOTE_ARTIFACT_RE = re.compile(r'"{2,}')

# Regex to clean spaces around slashes in technical identifiers (e.g. "L / H" -> "L/H")
_SLASH_SPACING_RE = re.compile(r"\b([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)\b")

# Regex to clean spaces between cylinder hash tokens (e.g. "# 1" -> "#1")
_HASH_NUMBER_RE = re.compile(r"#\s*(\d+)")


def normalize_text(text: Optional[str], validated_glossary: Optional[Dict[str, str]] = None) -> str:
    """
    Deterministically normalizes aviation maintenance narrative text.

    Guarantees:
    1. Idempotency: normalize_text(normalize_text(s)) == normalize_text(s)
    2. Preservation: Important technical tokens (#1, L/H, 2700 RPM, 40/80 PSI) are preserved.
    3. Non-destructive: No uncalibrated regex deletions or unvalidated global replacements.
    4. Safe handling of None/empty inputs.

    Args:
        text: Raw input string.
        validated_glossary: Optional map of validated abbreviation expansions (Phase 2 integration).
    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode normalization (NFKC decomposes compatibility chars into canonical forms)
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Fix double quote artifacts from escaped CSV strings ("" -> ")
    cleaned = _QUOTE_ARTIFACT_RE.sub('"', cleaned)

    # 3. Collapse multiple spaces, newlines, tabs, and non-breaking spaces into a single space
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    # 4. Standardize cylinder notation spacing ("# 1" -> "#1")
    cleaned = _HASH_NUMBER_RE.sub(r"#\1", cleaned)

    # 5. Standardize slash tokens ("L / H" -> "L/H", "R / H" -> "R/H", "OPS / LEAK" -> "OPS/LEAK")
    cleaned = _SLASH_SPACING_RE.sub(r"\1/\2", cleaned)

    # 6. Apply validated glossary expansions if provided (Conservative token-boundary substitution)
    if validated_glossary:
        for abbr, expansion in validated_glossary.items():
            pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
            cleaned = pattern.sub(expansion, cleaned)

    return cleaned


def normalize_record_fields(problem: Optional[str], action: Optional[str]) -> tuple[str, str]:
    """
    Normalizes both problem and action fields deterministically.
    """
    return normalize_text(problem), normalize_text(action)
