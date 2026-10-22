"""
Unit and integration tests for Phase 1: Ingestion and Normalization.
Verifies data immutability, technical token preservation, BOM handling,
duplicate detection, malformed row resilience, and deterministic reproducibility.
"""

import json
import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.models import LogbookRecord, IngestionSummary
from src.normalizer import normalize_text, normalize_record_fields
from src.ingest import Ingestor, ingest_and_save


class TestNormalization:
    """Tests for deterministic text normalization."""

    def test_whitespace_and_noise_normalization(self):
        raw = "  ENGINE  \t IDLE \r\n OVERRIDE   KILLED  ENGINE.   "
        expected = "ENGINE IDLE OVERRIDE KILLED ENGINE."
        assert normalize_text(raw) == expected

    def test_idempotency(self):
        sample = "  # 1  CYLINDER   EXHAUST VALVE   IAW   MAINT MANUAL  "
        first_pass = normalize_text(sample)
        second_pass = normalize_text(first_pass)
        assert first_pass == second_pass

    def test_none_and_empty_handling(self):
        assert normalize_text(None) == ""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_technical_tokens_preservation(self):
        sample = "# 2 & 4 CYL ROCKER COVER GASKETS LEAKING @ 1200 RPM, COMPRESSION 76/80 PSI."
        normalized = normalize_text(sample)
        # Verify # 2 was normalized to #2 and key technical numbers/tokens remain
        assert "#2 & 4 CYL" in normalized
        assert "1200 RPM" in normalized
        assert "76/80 PSI" in normalized

    def test_slash_token_spacing(self):
        sample = "L / H ENGINE AND R / H ENGINE BAFFLE TIE ROD"
        normalized = normalize_text(sample)
        assert "L/H ENGINE" in normalized
        assert "R/H ENGINE" in normalized

    def test_quote_artifacts(self):
        sample = 'CHECKED CYLINDER ""MISSING"" ON RUN UP.'
        normalized = normalize_text(sample)
        assert 'CHECKED CYLINDER "MISSING" ON RUN UP.' == normalized


class TestLogbookRecordModel:
    """Tests for the canonical LogbookRecord model."""

    def test_model_immutability(self):
        record = LogbookRecord(
            record_id="100001",
            problem_raw=" RAW PROBLEM ",
            action_raw=" RAW ACTION ",
            problem_normalized="RAW PROBLEM",
            action_normalized="RAW ACTION"
        )
        # Attempting to mutate a frozen model must raise an error
        with pytest.raises(ValidationError):
            record.problem_raw = "MODIFIED"

    def test_raw_text_preservation(self):
        raw_prob = "  UNTRIMMED   SPACES \t\n AND QUOTES \"\" "
        raw_act = " REPLACED GASKET IAW "
        record = LogbookRecord(
            record_id="100002",
            problem_raw=raw_prob,
            action_raw=raw_act,
            problem_normalized=normalize_text(raw_prob),
            action_normalized=normalize_text(raw_act)
        )
        assert record.problem_raw == raw_prob
        assert record.action_raw == raw_act
        assert record.problem_normalized != raw_prob

    def test_empty_record_id_rejected(self):
        with pytest.raises(ValidationError):
            LogbookRecord(
                record_id="   ",
                problem_raw="A",
                action_raw="B",
                problem_normalized="A",
                action_normalized="B"
            )


class TestIngestionEngine:
    """Tests for ingestion error handling, BOM reading, and duplicate detection."""

    def test_utf8_bom_handling(self, tmp_path):
        bom_csv = tmp_path / "bom_test.csv"
        # Write with UTF-8 BOM (\xef\xbb\xbf)
        content = (
            b"\xef\xbb\xbfIDENT,PROBLEM,ACTION\n"
            b"200001,TEST PROBLEM 1,TEST ACTION 1\n"
            b"200002,TEST PROBLEM 2,TEST ACTION 2\n"
        )
        bom_csv.write_bytes(content)

        ingestor = Ingestor(bom_csv)
        records, summary = ingestor.run_ingestion()

        assert len(records) == 2
        assert records[0].record_id == "200001"
        assert records[0].problem_raw == "TEST PROBLEM 1"
        assert summary.valid_records_ingested == 2
        assert summary.duplicate_ids_count == 0

    def test_duplicate_id_detection(self, tmp_path):
        dup_csv = tmp_path / "dup_test.csv"
        content = (
            "IDENT,PROBLEM,ACTION\n"
            "300001,FIRST PROBLEM,FIRST ACTION\n"
            "300001,SECOND PROBLEM DUPLICATE,SECOND ACTION DUPLICATE\n"
            "300002,UNIQUE PROBLEM,UNIQUE ACTION\n"
        )
        dup_csv.write_text(content, encoding="utf-8")

        ingestor = Ingestor(dup_csv)
        records, summary = ingestor.run_ingestion()

        assert len(records) == 2
        assert summary.total_raw_rows == 3
        assert summary.valid_records_ingested == 2
        assert summary.duplicate_ids_count == 1
        # The first occurrence is preserved
        assert records[0].record_id == "300001"
        assert records[0].problem_raw == "FIRST PROBLEM"

    def test_malformed_and_missing_id_rows(self, tmp_path):
        malformed_csv = tmp_path / "malformed_test.csv"
        content = (
            "IDENT,PROBLEM,ACTION\n"
            "400001,VALID ROW,VALID ACTION\n"
            ",MISSING ID ROW,ACTION WITH NO ID\n"
            "   ,WHITESPACE ID ROW,ACTION WITH BLANK ID\n"
            "400002,VALID ROW 2,VALID ACTION 2\n"
        )
        malformed_csv.write_text(content, encoding="utf-8")

        ingestor = Ingestor(malformed_csv)
        records, summary = ingestor.run_ingestion()

        assert len(records) == 2
        assert summary.total_raw_rows == 4
        assert summary.malformed_rows_count == 2
        assert [r.record_id for r in records] == ["400001", "400002"]


class TestFullDatasetIngestion:
    """Integration test against actual raw dataset file."""

    def test_real_dataset_ingestion_and_manifest(self, tmp_path):
        real_csv = Path("data/raw/logbooks.csv")
        assert real_csv.exists(), "Raw dataset must exist at data/raw/logbooks.csv"

        out_dir = tmp_path / "processed_test"
        records, summary = ingest_and_save(real_csv, out_dir)

        # 1. Total record count must match inspection
        assert len(records) == 6169
        assert summary.valid_records_ingested == 6169
        assert summary.duplicate_ids_count == 0
        assert summary.malformed_rows_count == 0

        # 2. Sequential ID stability
        assert records[0].record_id == "100001"
        assert records[-1].record_id == "106169"

        # 3. Output files check
        jsonl_path = out_dir / "records.jsonl"
        lookup_path = out_dir / "records_lookup.json"
        report_path = out_dir / "ingestion_report.json"

        assert jsonl_path.exists()
        assert lookup_path.exists()
        assert report_path.exists()

        # 4. Verify lookup JSON is complete and fast
        with open(lookup_path, "r", encoding="utf-8") as f:
            lookup_data = json.load(f)
        assert len(lookup_data) == 6169
        assert lookup_data["100001"]["problem_raw"] == "ENGINE IDLE OVERRIDE KILLED ENGINE."
        assert lookup_data["100001"]["action_raw"] == "TRIED TO ADJUST IDLE SEVERAL TIMES, WOULDN'T ADJUST."

        # 5. Immutability guarantee
        assert lookup_data["100001"]["problem_raw"] == records[0].problem_raw
        assert lookup_data["100001"]["action_raw"] == records[0].action_raw
