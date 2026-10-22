"""
Unit and integration tests for Phase 2: Abbreviation discovery and glossary.
Tests false positives, ambiguous terms, unknown terms, repeated terms,
conservative expansion policies, and raw text preservation.
"""

import json
from pathlib import Path
import pytest

from src.models import LogbookRecord
from src.abbreviation.models import Glossary, GlossaryEntry
from src.abbreviation.detector import AbbreviationDetector
from src.abbreviation.expander import expand_text_with_glossary, expand_record
from scripts.discover_abbreviations import discover_and_build_glossary


class TestAbbreviationDetection:
    """Tests for statistical candidate filtering and excerpt collection."""

    def test_false_positive_filtering(self):
        detector = AbbreviationDetector(min_frequency=1)
        records = [
            LogbookRecord(
                record_id="100001",
                problem_raw="1200 RPM AND HOSE LEAK WAS NOT FOUND FOR ALL COVERS.",
                action_raw="REPLACED NEW SEAL WITH BOLT AND SCREW.",
                problem_normalized="",
                action_normalized=""
            )
        ]
        candidates = detector.discover_candidates(records)
        # Numbers, stop words, and common words must be filtered out
        assert "1200" not in candidates
        assert "AND" not in candidates
        assert "FOR" not in candidates
        assert "HOSE" not in candidates
        assert "LEAK" not in candidates
        assert "SEAL" not in candidates
        assert "BOLT" not in candidates
        # True abbreviation RPM should be detected
        assert "RPM" in candidates

    def test_slash_and_apostrophe_tokens_discovery(self):
        detector = AbbreviationDetector(min_frequency=1)
        records = [
            LogbookRecord(
                record_id="100002",
                problem_raw="L/H AND R/H ENG A/C O/B BAFFLE.",
                action_raw="OPS CK'ED GOOD IAW MAINTENANCE MANUAL.",
                problem_normalized="",
                action_normalized=""
            )
        ]
        candidates = detector.discover_candidates(records)
        assert "L/H" in candidates
        assert "R/H" in candidates
        assert "A/C" in candidates
        assert "O/B" in candidates
        assert "CK'ED" in candidates
        assert "IAW" in candidates

    def test_contextual_excerpts_contain_source_ids(self):
        detector = AbbreviationDetector(min_frequency=1)
        records = [
            LogbookRecord(
                record_id="100003",
                problem_raw="ENGINE IDLE OVERRIDE KILLED ENGINE.",
                action_raw="A/C WAS RUN UP, SET IDLE SPEED, MIXTURE OK.",
                problem_normalized="",
                action_normalized=""
            )
        ]
        candidates = detector.discover_candidates(records)
        ac_cand = candidates.get("A/C")
        assert ac_cand is not None
        assert len(ac_cand.excerpts) > 0
        rec_id, excerpt = ac_cand.excerpts[0]
        assert rec_id == "100003"
        assert "A/C" in excerpt


class TestConservativeExpansionPolicy:
    """Verifies that only validated glossary terms are applied in normalization."""

    @pytest.fixture
    def test_glossary(self) -> Glossary:
        glossary = Glossary()
        # Validated terms
        glossary.add_or_update(GlossaryEntry(
            term="CYL",
            expansion="cylinder",
            status="validated"
        ))
        glossary.add_or_update(GlossaryEntry(
            term="L/H",
            expansion="left-hand",
            status="validated"
        ))
        # Ambiguous term (MUST NOT be automatically expanded)
        glossary.add_or_update(GlossaryEntry(
            term="EXT",
            expansion="exhaust",
            status="ambiguous",
            alternative_meanings=["external"]
        ))
        # Needs review term (MUST NOT be automatically expanded)
        glossary.add_or_update(GlossaryEntry(
            term="PD",
            expansion="pilot defect",
            status="needs_review"
        ))
        # Unknown term (MUST NOT be automatically expanded)
        glossary.add_or_update(GlossaryEntry(
            term="UNK",
            expansion="UNKNOWN",
            status="unknown"
        ))
        return glossary

    def test_only_validated_terms_are_expanded(self, test_glossary):
        raw_text = "REPLACED #2 CYL EXT PUSH ROD ON L/H ENG PD UNK."
        expanded = expand_text_with_glossary(raw_text, test_glossary)

        # Validated terms expand
        assert "cylinder" in expanded
        assert "left-hand" in expanded
        # Unvalidated terms remain intact as originally written
        assert "EXT" in expanded
        assert "PD" in expanded
        assert "UNK" in expanded
        assert "pilot defect" not in expanded

    def test_word_boundary_isolation(self, test_glossary):
        # Ensure sub-strings are not greedily corrupted (e.g. "CYLINDER" shouldn't become "cylinderINDER")
        raw_text = "CYLINDER CYL CYLS"
        expanded = expand_text_with_glossary(raw_text, test_glossary)
        assert expanded.startswith("CYLINDER cylinder CYLS")

    def test_record_immutability_during_expansion(self, test_glossary):
        record = LogbookRecord(
            record_id="100005",
            problem_raw="#2 & 4 CYL ROCKER COVER GASKETS ARE LEAKING.",
            action_raw="REMOVED & REPLACED GASKETS ON L/H ENGINE.",
            problem_normalized="#2 & 4 CYL ROCKER COVER GASKETS ARE LEAKING.",
            action_normalized="REMOVED & REPLACED GASKETS ON L/H ENGINE."
        )

        expanded_record = expand_record(record, test_glossary)

        # Raw texts remain strictly untouched
        assert expanded_record.problem_raw == record.problem_raw
        assert expanded_record.action_raw == record.action_raw

        # Normalized texts reflect validated expansions
        assert "cylinder" in expanded_record.problem_normalized
        assert "left-hand" in expanded_record.action_normalized


class TestHumanGlossaryEditability:
    """Verifies that human edits to glossary JSON take effect without code changes."""

    def test_human_status_override(self, tmp_path):
        glossary_file = tmp_path / "glossary.json"

        # Initially mark a term as needs_review
        glossary = Glossary()
        glossary.add_or_update(GlossaryEntry(
            term="ALT",
            expansion="alternator",
            status="needs_review"
        ))
        glossary.save_json(str(glossary_file))

        # Check before human edit
        loaded_1 = Glossary.load_json(str(glossary_file))
        assert "ALT" not in loaded_1.get_validated_map()
        assert expand_text_with_glossary("ALT ATTACH BOLT", loaded_1) == "ALT ATTACH BOLT"

        # Simulate human engineer editing the JSON directly
        with open(glossary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["ALT"]["status"] = "validated"
        with open(glossary_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Check after human edit without any code modification
        loaded_2 = Glossary.load_json(str(glossary_file))
        assert "ALT" in loaded_2.get_validated_map()
        assert expand_text_with_glossary("ALT ATTACH BOLT", loaded_2) == "alternator ATTACH BOLT"


class TestFullDatasetGlossaryIntegration:
    """Integration test against actual production records."""

    def test_real_dataset_glossary_generation(self, tmp_path):
        records_jsonl = "data/processed/records.jsonl"
        assert Path(records_jsonl).exists(), "Must run Phase 1 first."

        out_glossary = tmp_path / "glossary.json"
        out_report = tmp_path / "glossary_report.md"

        glossary = discover_and_build_glossary(
            records_jsonl=records_jsonl,
            output_glossary_json=str(out_glossary),
            output_report_md=str(out_report),
            min_frequency=5
        )

        assert len(glossary.entries) > 20
        assert out_glossary.exists()
        assert out_report.exists()

        # Key aviation terms must be present
        for term in ["CYL", "L/H", "R/H", "IAW", "OPS", "RPM"]:
            assert term in glossary.entries
            entry = glossary.entries[term]
            assert entry.status == "validated"
            assert entry.frequency >= 5
            assert len(entry.context_examples) > 0
            # Excerpts must cite real record IDs
            assert any("[" in ex for ex in entry.context_examples)
