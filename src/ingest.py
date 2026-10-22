"""
Ingestion engine for aviation maintenance logbooks.
Validates raw records, preserves original source text, produces normalized representations,
detects duplicates and malformed records, and outputs reproducible manifests.
"""

import sys
import os
import csv
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.models import LogbookRecord, IngestionSummary
from src.normalizer import normalize_record_fields


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for immutable provenance tracking."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class Ingestor:
    """
    Ingests and validates raw maintenance logbook CSV files into canonical LogbookRecord objects.
    """

    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {self.source_path}")
        self.source_sha256 = compute_file_sha256(self.source_path)

    def run_ingestion(self) -> Tuple[List[LogbookRecord], IngestionSummary]:
        """
        Executes ingestion with strict validation and provenance tracking.
        """
        seen_ids = set()
        valid_records: List[LogbookRecord] = []
        duplicate_count = 0
        malformed_count = 0
        total_rows = 0
        detected_fields: List[str] = []

        # Read CSV with utf-8-sig to safely strip any UTF-8 BOM
        with open(self.source_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("Source CSV file is empty or missing headers.")
            
            # Map headers (case-insensitive stripped)
            field_map = {col.strip().upper(): col for col in reader.fieldnames}
            detected_fields = list(reader.fieldnames)

            id_col = field_map.get("IDENT") or field_map.get("ID") or field_map.get("RECORD_ID")
            prob_col = field_map.get("PROBLEM") or field_map.get("DISCREPANCY") or field_map.get("SQUAWK")
            act_col = field_map.get("ACTION") or field_map.get("CORRECTIVE_ACTION") or field_map.get("RESOLUTION")

            if not id_col:
                raise ValueError(f"Could not identify primary ID column in headers: {detected_fields}")
            if not prob_col:
                raise ValueError(f"Could not identify problem column in headers: {detected_fields}")
            if not act_col:
                raise ValueError(f"Could not identify action column in headers: {detected_fields}")

            for row_idx, row in enumerate(reader, start=1):
                total_rows += 1
                raw_id = row.get(id_col)

                # Validate ID presence
                if not raw_id or not str(raw_id).strip():
                    malformed_count += 1
                    continue

                clean_id = str(raw_id).strip()

                # Duplicate detection
                if clean_id in seen_ids:
                    duplicate_count += 1
                    continue
                seen_ids.add(clean_id)

                raw_problem = row.get(prob_col, "")
                raw_action = row.get(act_col, "")

                if raw_problem is None:
                    raw_problem = ""
                if raw_action is None:
                    raw_action = ""

                # Perform deterministic normalization
                norm_problem, norm_action = normalize_record_fields(raw_problem, raw_action)

                record = LogbookRecord(
                    record_id=clean_id,
                    problem_raw=str(raw_problem),
                    action_raw=str(raw_action),
                    problem_normalized=norm_problem,
                    action_normalized=norm_action,
                    metadata={
                        "source_row": row_idx,
                        "source_sha256": self.source_sha256
                    }
                )
                valid_records.append(record)

        summary = IngestionSummary(
            source_file=str(self.source_path.resolve()),
            output_file="",
            source_sha256=self.source_sha256,
            total_raw_rows=total_rows,
            valid_records_ingested=len(valid_records),
            duplicate_ids_count=duplicate_count,
            malformed_rows_count=malformed_count,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            fields_detected=detected_fields
        )

        return valid_records, summary


def ingest_and_save(
    source_csv: str | Path,
    output_dir: str | Path = "data/processed"
) -> Tuple[List[LogbookRecord], IngestionSummary]:
    """
    Ingests source CSV and serializes outputs to:
    - data/processed/records.jsonl (line-delimited canonical records)
    - data/processed/records_lookup.json (id-keyed dictionary for O(1) lookup)
    - data/processed/ingestion_report.json (provenance manifest)
    """
    source_path = Path(source_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_output = out_dir / "records.jsonl"
    lookup_output = out_dir / "records_lookup.json"
    manifest_output = out_dir / "ingestion_report.json"

    ingestor = Ingestor(source_path)
    records, summary_template = ingestor.run_ingestion()

    # 1. Write JSONL
    with open(jsonl_output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    # 2. Write id-keyed lookup JSON
    lookup_map = {rec.record_id: rec.model_dump() for rec in records}
    with open(lookup_output, "w", encoding="utf-8") as f:
        json.dump(lookup_map, f, indent=2)

    # 3. Write summary report manifest
    summary = IngestionSummary(
        source_file=str(source_path.resolve()),
        output_file=str(jsonl_output.resolve()),
        source_sha256=summary_template.source_sha256,
        total_raw_rows=summary_template.total_raw_rows,
        valid_records_ingested=summary_template.valid_records_ingested,
        duplicate_ids_count=summary_template.duplicate_ids_count,
        malformed_rows_count=summary_template.malformed_rows_count,
        timestamp_utc=summary_template.timestamp_utc,
        fields_detected=summary_template.fields_detected
    )

    with open(manifest_output, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

    print(f"[INGESTION COMPLETE]")
    print(f"  Source file:      {source_path.name}")
    print(f"  Source SHA256:    {summary.source_sha256[:16]}...")
    print(f"  Total rows read:  {summary.total_raw_rows:,}")
    print(f"  Valid ingested:   {summary.valid_records_ingested:,}")
    print(f"  Duplicates:       {summary.duplicate_ids_count}")
    print(f"  Malformed rows:   {summary.malformed_rows_count}")
    print(f"  Saved JSONL to:   {jsonl_output}")
    print(f"  Saved Lookup to:  {lookup_output}")
    print(f"  Saved Report to:  {manifest_output}")

    return records, summary


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/logbooks.csv"
    dest = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    ingest_and_save(src, dest)
