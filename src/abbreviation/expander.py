"""
Conservative abbreviation expander and formatter.
Applies ONLY validated glossary entries to normalized text representations.
Leaves original raw text and unvalidated/ambiguous terms completely intact.
Provides full-word-with-abbreviation-in-brackets formatting for maintenance fixes.
"""

import re
from typing import Dict, Optional, List
from src.models import LogbookRecord
from src.abbreviation.models import Glossary


# Comprehensive aviation acronym expansions: Acronym -> Full Expansion
AVIATION_ACRONYM_EXPANSIONS: Dict[str, str] = {
    "AMM": "Aircraft Maintenance Manual",
    "A/C": "Aircraft",
    "CYL": "Cylinder",
    "CYLS": "Cylinders",
    "ENG": "Engine",
    "ENGS": "Engines",
    "L/H": "Left-Hand",
    "R/H": "Right-Hand",
    "LH": "Left-Hand",
    "RH": "Right-Hand",
    "IAW": "In Accordance With",
    "OPS": "Operational",
    "CK": "Check",
    "CK'D": "Checked",
    "CK'ED": "Checked",
    "MAG": "Magneto",
    "MAGS": "Magnetos",
    "RPM": "Revolutions Per Minute",
    "PSI": "Pounds Per Square Inch",
    "CHT": "Cylinder Head Temperature",
    "EGT": "Exhaust Gas Temperature",
    "FWD": "Forward",
    "AFT": "Aft",
    "I/B": "Inboard",
    "O/B": "Outboard",
    "IB": "Inboard",
    "OB": "Outboard",
    "LWR": "Lower",
    "UPR": "Upper",
    "ALT": "Alternator",
    "ASSY": "Assembly",
    "INSP": "Inspection",
    "INSP'D": "Inspected",
    "PROP": "Propeller",
    "DISCREP": "Discrepancy",
    "FDM": "Flight Data Monitoring",
    "FOD": "Foreign Object Debris",
    "RTV": "Room-Temperature Vulcanizing Silicone",
    "SRM": "Structural Repair Manual",
    "MM": "Maintenance Manual",
    "VOR": "VHF Omnidirectional Range",
    "ILS": "Instrument Landing System",
    "APU": "Auxiliary Power Unit",
    "GPU": "Ground Power Unit",
    "NDI": "Non-Destructive Inspection",
    "NDT": "Non-Destructive Testing",
    "MEL": "Minimum Equipment List",
    "FCU": "Fuel Control Unit",
    "CSD": "Constant Speed Drive",
    "AD": "Airworthiness Directive",
    "SB": "Service Bulletin",
    "R&R": "Remove and Replace",
    "R/R": "Remove and Replace",
    "O/H": "Overhaul",
    "OH": "Overhaul",
    "IPC": "Illustrated Parts Catalog",
    "TSO": "Technical Standard Order",
    "EXH": "Exhaust",
    "EXT": "Exhaust",
    "W/": "With",
    "W/O": "Without",
    "W/I": "Within",
    "STBY": "Standby",
    "XCVR": "Transceiver",
    "XPNDR": "Transponder",
    "TCAS": "Traffic Alert and Collision Avoidance System",
    "GPWS": "Ground Proximity Warning System",
    "EICAS": "Engine Indicating and Crew Alerting System",
    "ECAM": "Electronic Centralized Aircraft Monitor",
    "FADEC": "Full Authority Digital Engine Control",
    "HYD": "Hydraulic",
    "PNEU": "Pneumatic",
    "ELEC": "Electrical",
    "GEN": "Generator",
    "BATT": "Battery",
    "NAV": "Navigation",
    "COMM": "Communication",
    "COM": "Communication",
    "ADF": "Automatic Direction Finder",
    "DME": "Distance Measuring Equipment",
    "XFR": "Transfer",
    "VLV": "Valve",
    "PRESS": "Pressure",
    "TEMP": "Temperature",
    "QTY": "Quantity",
    "POS": "Position",
    "IND": "Indicator",
    "SW": "Switch",
    "CIR": "Circuit",
    "CB": "Circuit Breaker",
    "SYS": "System",
    "XMSN": "Transmission",
    "GEAR": "Landing Gear",
    "LDG": "Landing",
    "BRK": "Brake",
    "TIRE": "Tire",
    "WHL": "Wheel",
    "FLT": "Flight",
    "GND": "Ground"
}


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


def expand_fix_with_acronym_brackets(text: str, glossary: Optional[Glossary] = None) -> str:
    """
    Expands abbreviations and acronyms in maintenance fix descriptions to:
    'Full Word (ABBREVIATION)'.
    Example:
      'REMOVED & REPLACED FUEL SERVO PER AMM 28-20-00 ON A/C'
      -> 'REMOVED & REPLACED FUEL SERVO PER Aircraft Maintenance Manual (AMM) 28-20-00 ON Aircraft (A/C)'

    Strictly idempotent: will not double-expand terms that are already expanded.
    """
    if not text:
        return ""

    # Merge standard mappings with any validated glossary entries
    merged_mappings = dict(AVIATION_ACRONYM_EXPANSIONS)
    if glossary is not None:
        try:
            for term, entry in glossary.entries.items():
                if entry.status.value == "validated" and entry.expansion and entry.expansion != "UNKNOWN":
                    exp = entry.expansion
                    if not exp.lower().startswith("aviation logbook acronym"):
                        # Title case single/multi words
                        title_exp = " ".join(word.capitalize() for word in exp.split())
                        if term not in merged_mappings:
                            merged_mappings[term] = title_exp
        except Exception:
            pass

    expanded = text
    # Sort by acronym length descending to expand longer abbreviations first (e.g. CYLS before CYL)
    for abbr in sorted(merged_mappings.keys(), key=lambda x: len(x), reverse=True):
        full = merged_mappings[abbr]
        target = f"{full} ({abbr})"
        escaped_abbr = re.escape(abbr)

        def repl(match: re.Match) -> str:
            start = match.start()
            end = match.end()

            # 1. Skip if already preceded by the full word
            pre = expanded[max(0, start - len(full) - 3):start]
            if full.lower() in pre.lower():
                return match.group(0)

            # 2. Skip if already enclosed inside parentheses of an expansion e.g. (AMM)
            if start > 0 and end < len(expanded):
                if expanded[start - 1] == "(" and expanded[end] == ")":
                    return match.group(0)

            # 3. Skip if already followed by (ABBR)
            post = expanded[end:min(len(expanded), end + len(abbr) + 4)]
            if post.strip().lower().startswith(f"({abbr.lower()})"):
                return match.group(0)

            return target

        if "/" in abbr or "&" in abbr:
            pat = re.compile(r"(?<![A-Za-z0-9])" + escaped_abbr + r"(?![A-Za-z0-9])", re.IGNORECASE)
        else:
            pat = re.compile(r"\b" + escaped_abbr + r"\b", re.IGNORECASE)

        expanded = pat.sub(repl, expanded)

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
