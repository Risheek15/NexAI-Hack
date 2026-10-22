"""
Statistical and pattern-based abbreviation detector for aviation maintenance records.
Extracts candidate abbreviations without labeled training data, collects contextual
excerpts with source record IDs, and deduplicates representative examples.
"""

import re
from typing import List, Dict, Set, Tuple
from src.models import LogbookRecord

# Standard English stop words and regular English dictionary words commonly appearing in uppercase logbooks
_EXCLUDED_TOKENS: Set[str] = {
    # Pronouns, prepositions, conjunctions, auxiliary verbs
    "AND", "FOR", "THE", "WITH", "FROM", "NOT", "WAS", "HAS", "BUT", "ALL",
    "OFF", "OUT", "NEW", "TOO", "ANY", "SET", "TOP", "TWO", "ONE", "DUE",
    "DID", "HAD", "FEW", "GOT", "RAN", "PUT", "SEE", "RUN", "LOW", "DAY",
    "ITS", "ARE", "CAN", "CUT", "LET", "WAY", "PER", "TIE", "RED", "OLD",
    "AFTER", "BEFORE", "BOTH", "SIDE", "AREA", "WHEN", "OVER", "INTO", "SOME",
    "VERY", "ALSO", "THAT", "THEY", "THIS", "BEEN", "MADE", "HAVE", "WERE",
    "MORE", "SAME", "WILL", "UPON", "THEN", "ONLY", "ONCE", "WHAT", "MUCH",
    "ON", "IS", "TO", "OF", "BE", "IN", "UP", "NO", "AT", "AS", "SO", "BY",
    "IF", "OR", "AN", "DO", "GO", "IT", "HE", "WE", "US", "AM", "ME", "MY",
    "NOW", "HOW", "WHY", "EACH", "SUCH", "LIKE", "THAN", "JUST", "ABOUT",
    "WHILE", "THERE", "WHERE", "WHICH", "AGAIN", "COULD", "WOULD", "SHOULD",

    # Common logbook parts and nouns (plain English, not abbreviations)
    "HOSE", "SEAL", "LEAK", "LEAKS", "TUBE", "COVER", "BOLT", "SCREW", "CRACK",
    "PATCH", "RIVET", "LINES", "SEALS", "TUBES", "COVERS", "BOLTS", "SCREWS",
    "ROD", "NUT", "NUTS", "LINE", "WIRES", "WIRE", "HEAD", "HEADS", "HOLE",
    "PLUG", "TABS", "TIME", "DOOR", "CASE", "DROP", "PUMP", "RING", "BASE",
    "GASKET", "INTAKE", "ROCKER", "BAFFLE", "ENGINE", "VALVE", "VALVES",
    "START", "ROUGH", "FOUND", "BROKEN", "WORN", "NEEDS", "PLUGS", "STOP",
    "GOOD", "RIGHT", "LEFT", "REAR", "FRONT", "FLIGHT", "IDLE", "POWER",
    "SMOOTH", "SMALL", "LARGE", "HOLES", "TEETH", "DRAIN", "SUMP", "TANK",
    "MOTOR", "SPRING", "SHROUD", "SEAM", "STRAP", "FRAME", "CLAMP", "CLAMPS",
    "RIVETS", "PANEL", "ARM", "ARMS", "RODS", "SPEED", "RISES", "RISE",
    "CRACKS", "CRACKED", "LEAKING", "MISSING", "FAILED", "WARMED", "DYED",
    "TEST", "TESTS", "TESTED", "CHECK", "CHECKS", "CHECKED",

    # Common logbook action verbs (past tense / standard English)
    "REMOVED", "REPLACED", "INSTALLED", "TIGHTENED", "SECURED", "CLEANED",
    "DRILLED", "TORQUED", "SAFETIED", "FABRICATED", "REPAIRED", "ADJUSTED",
    "APPLIED", "RETORQUED", "REATTACHED", "REPOSITIONED", "COMPLETED",
    "CONFIRMED", "RESWAGED", "RESWEDGED", "FINISHED", "STARTED", "VERIFIED",
    "DISASSEMBLED", "REASSEMBLED", "STOPPED", "TOUCHED", "BLENDED", "TRIMMED",
    "LOOSENED", "REINDEXED", "STAKED", "LAPPED", "REAMED", "HONED", "SEALED"
}

# Regex for candidate abbreviations:
# 1. Words with slashes: L/H, R/H, A/C, O/B, I/B, M/R
# 2. Uppercase words with apostrophes: CK'D, CK'ED, INSP'D
# 3. All-caps words between 2 and 6 letters
_ABBREVIATION_PATTERN = re.compile(
    r"\b([A-Z0-9]+/[A-Z0-9]+|[A-Z]+'[A-Z]+|[A-Z]{2,6})\b"
)


class CandidateOccurrence:
    """Represents a candidate token and its contextual occurrences."""
    def __init__(self, term: str):
        self.term = term
        self.frequency = 0
        self.excerpts: List[Tuple[str, str]] = []  # [(record_id, excerpt_text), ...]

    def add_excerpt(self, record_id: str, excerpt: str) -> None:
        self.frequency += 1
        clean_excerpt = excerpt.strip()
        existing_texts = {text for _, text in self.excerpts}
        if clean_excerpt not in existing_texts and len(self.excerpts) < 10:
            self.excerpts.append((record_id, clean_excerpt))


class AbbreviationDetector:
    """
    Discovers candidate abbreviations across records using frequency and morphology rules.
    """

    def __init__(self, min_frequency: int = 3):
        self.min_frequency = min_frequency

    def extract_surrounding_excerpt(self, text: str, start_idx: int, end_idx: int, window_words: int = 8) -> str:
        """Extracts a short surrounding window around the candidate term."""
        words = text.split()
        target_token = text[start_idx:end_idx]
        for i, w in enumerate(words):
            if target_token in w:
                left = max(0, i - window_words)
                right = min(len(words), i + window_words + 1)
                return " ".join(words[left:right])
        return text[:100]

    def discover_candidates(self, records: List[LogbookRecord]) -> Dict[str, CandidateOccurrence]:
        """
        Scans all records and collects candidates with frequency and source provenance.
        """
        candidates: Dict[str, CandidateOccurrence] = {}

        for rec in records:
            combined = f"{rec.problem_raw} {rec.action_raw}"
            for match in _ABBREVIATION_PATTERN.finditer(combined):
                token = match.group(1).strip()

                # Filter out pure numbers (e.g. 1200, 2700)
                if token.isdigit():
                    continue

                # Filter out stop words, plain parts, and verbs
                if token in _EXCLUDED_TOKENS:
                    continue

                if token not in candidates:
                    candidates[token] = CandidateOccurrence(token)

                if len(candidates[token].excerpts) < 10:
                    excerpt = self.extract_surrounding_excerpt(combined, match.start(), match.end())
                    candidates[token].add_excerpt(rec.record_id, excerpt)
                else:
                    candidates[token].frequency += 1

        # Filter by minimum frequency
        filtered = {
            k: v for k, v in candidates.items()
            if v.frequency >= self.min_frequency
        }
        return filtered
