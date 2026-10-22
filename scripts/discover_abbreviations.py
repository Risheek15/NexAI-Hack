"""
Script to discover candidate abbreviations from ingested records,
collect contextual excerpts with source IDs, resolve expansions,
and generate the editable data/glossary.json and data/glossary_report.md.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import LogbookRecord
from src.abbreviation.models import Glossary, GlossaryEntry
from src.abbreviation.detector import AbbreviationDetector
from src.abbreviation.suggester import resolve_candidate_expansion


def discover_and_build_glossary(
    records_jsonl: str = "data/processed/records.jsonl",
    output_glossary_json: str = "data/glossary.json",
    output_report_md: str = "data/glossary_report.md",
    min_frequency: int = 5,
    allow_llm: bool = False
) -> Glossary:
    records_path = Path(records_jsonl)
    if not records_path.exists():
        print(f"[ERROR] Processed records file not found at: {records_path}")
        sys.exit(1)

    print(f"[INFO] Loading records from: {records_path}")
    records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(LogbookRecord(**json.loads(line)))

    print(f"[INFO] Scanning {len(records):,} records for candidate abbreviations...")
    detector = AbbreviationDetector(min_frequency=min_frequency)
    candidates = detector.discover_candidates(records)

    print(f"[INFO] Discovered {len(candidates)} candidate abbreviations with frequency >= {min_frequency}.")

    # Load existing glossary if present to preserve human review changes
    glossary = Glossary.load_json(output_glossary_json)
    new_terms_count = 0
    updated_terms_count = 0

    for term, occ in sorted(candidates.items(), key=lambda x: x[1].frequency, reverse=True):
        formatted_excerpts = [f"[{rid}] {txt}" for rid, txt in occ.excerpts]

        if term in glossary.entries:
            # Update frequency and excerpts if existing, preserve existing validated status
            entry = glossary.entries[term]
            entry.frequency = occ.frequency
            # Merge excerpts
            existing_excerpts = set(entry.context_examples)
            for ex in formatted_excerpts:
                if ex not in existing_excerpts and len(entry.context_examples) < 10:
                    entry.context_examples.append(ex)
            updated_terms_count += 1
        else:
            # Resolve new entry
            entry = resolve_candidate_expansion(term, formatted_excerpts, occ.frequency, allow_llm=allow_llm)
            glossary.add_or_update(entry)
            new_terms_count += 1

    # Save glossary JSON
    out_json = Path(output_glossary_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    glossary.save_json(str(out_json))
    print(f"[SUCCESS] Saved {len(glossary.entries)} glossary entries to: {out_json}")

    # Generate Markdown Glossary Report
    status_counts = {"validated": 0, "needs_review": 0, "ambiguous": 0, "unknown": 0}
    for e in glossary.entries.values():
        status_counts[e.status] = status_counts.get(e.status, 0) + 1

    report_lines = [
        "# Aviation Maintenance Abbreviation Glossary Report",
        "",
        "## Summary",
        f"- **Total Discovered Terms:** {len(glossary.entries)}",
        f"- **Validated Terms:** {status_counts['validated']} (eligible for conservative expansion)",
        f"- **Needs Review:** {status_counts['needs_review']} (preserved as written until validated)",
        f"- **Ambiguous Terms:** {status_counts['ambiguous']} (multiple meanings detected; preserved)",
        f"- **Unknown Terms:** {status_counts['unknown']} (insufficient evidence; preserved)",
        "",
        "> [!IMPORTANT]",
        "> Only entries with `status: 'validated'` are applied during normalization.",
        "> Human engineers can review and edit `data/glossary.json` without modifying code.",
        "",
        "## Glossary Entries Table",
        "| Term | Expansion | Status | Freq | Provenance Source | Sample Supporting Excerpt |",
        "|---|---|---|---|---|---|"
    ]

    for term, entry in sorted(glossary.entries.items(), key=lambda x: x[1].frequency, reverse=True):
        sample_ex = entry.context_examples[0] if entry.context_examples else "N/A"
        # Escape pipe characters in sample excerpt
        clean_ex = sample_ex.replace("|", "/")
        badge = f"**{entry.status.upper()}**" if entry.status == "validated" else entry.status
        report_lines.append(
            f"| `{term}` | {entry.expansion} | {badge} | {entry.frequency} | {entry.source} | `{clean_ex[:85]}...` |"
        )

    out_report = Path(output_report_md)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[SUCCESS] Saved glossary human-review report to: {out_report}")
    return glossary


if __name__ == "__main__":
    discover_and_build_glossary()
