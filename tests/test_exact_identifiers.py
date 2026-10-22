"""
Regression tests for Exact Technical Identifier Handling and Four-Aspect Evidence Coverage.
Verifies all 8 required identifier constraints from AUDIT_REPORT.md Section 2:
1. CYL 2 versus CYL 12
2. #2 versus #12
3. 2700 RPM versus 270 RPM
4. L/H MAG versus R/H MAG
5. AN4-12A versus AN4-12
6. Mixed casing, punctuation, and whitespace variations
7. Queries with no identifier constraints
8. Multiple identifiers in one query
Also verifies Section 3 four-aspect coverage statuses (supported, unsupported, contradicted, not_applicable).
"""

import pytest
from src.retrieval.models import RetrievalResult
from src.agent.aspect_extractor import QueryAspects
from src.agent.aspect_coverage import (
    TechnicalConstraints,
    match_aspect_in_text,
    check_aspect_coverage,
    extract_cylinders,
    extract_rpm_values,
    extract_part_numbers,
    extract_orientations
)
from src.agent.evidence_gate import assess_retrieval_evidence, AspectGateConfig


class TestExactTechnicalIdentifiers:
    """Regression test suite for Section 2: Exact Technical Identifier Handling."""

    def test_case_1_cyl_2_versus_cyl_12(self):
        """1. CYL 2 versus CYL 12: record mentioning only CYL 12 must not support CYL 2."""
        aspect = "CYL 2"
        rec_cyl_12 = "DISCREPANCY: CYL 12 SPARK PLUG FOULED. REPLACED LOWER PLUG."
        rec_cyl_2 = "DISCREPANCY: FOUND CYL 2 LOWER PLUG FOULED. REPLACED SPARK PLUG."

        assert match_aspect_in_text(aspect, rec_cyl_12) is False
        assert match_aspect_in_text(aspect, rec_cyl_2) is True

    def test_case_2_hash_2_versus_hash_12(self):
        """2. #2 versus #12: #2 must not match #12."""
        aspect = "#2 intake gasket"
        rec_hash_12 = "REMOVED & REPLACED #12 INTAKE GASKET."
        rec_hash_2 = "REMOVED & REPLACED #2 INTAKE GASKET."

        assert match_aspect_in_text(aspect, rec_hash_12) is False
        assert match_aspect_in_text(aspect, rec_hash_2) is True

    def test_case_3_rpm_2700_versus_270(self):
        """3. 2700 RPM versus 270 RPM: exact RPM numeric tokens required."""
        aspect = "vibration at 2700 RPM"
        rec_270 = "ENGINE VIBRATING AT 270 RPM DURING LOW IDLE CHECK."
        rec_2700 = "ENGINE VIBRATING AT 2700 RPM DURING TAKEOFF RUN UP."

        assert match_aspect_in_text(aspect, rec_270) is False
        assert match_aspect_in_text(aspect, rec_2700) is True

    def test_case_4_lh_mag_versus_rh_mag(self):
        """4. L/H MAG versus R/H MAG: left-hand and right-hand orientation must not cross-match."""
        aspect = "L/H MAG blast tube"
        rec_rh = "R/H MAG BLAST TUBE FELL OFF. REPLACED BLAST TUBE."
        rec_lh = "L/H MAG BLAST TUBE WAS LOOSE. RE-SECURED BLAST TUBE."

        assert match_aspect_in_text(aspect, rec_rh) is False
        assert match_aspect_in_text(aspect, rec_lh) is True

    def test_case_5_part_an4_12a_versus_an4_12(self):
        """5. AN4-12A versus AN4-12: part dash-number suffix distinction."""
        aspect = "AN4-12A bolt"
        rec_without_a = "REPLACED DAMAGED AN4-12 BOLT WITH NEW HARDWARE."
        rec_with_a = "REPLACED DAMAGED AN4-12A BOLT WITH NEW HARDWARE."

        assert match_aspect_in_text(aspect, rec_without_a) is False
        assert match_aspect_in_text(aspect, rec_with_a) is True

    def test_case_6_mixed_casing_punctuation_and_whitespace(self):
        """6. Mixed casing, punctuation, and whitespace variations for technical identifiers."""
        variations = [
            "cyl 2",
            "CYL #2",
            "cyl. 2",
            "cylinder 2",
            "Cylinder #2",
            "#2"
        ]
        rec_text = "FOUND CYL 2 LOWER PLUG FOULED."

        for var in variations:
            assert match_aspect_in_text(var, rec_text) is True, f"Failed for variant: {var}"

    def test_case_7_unconstrained_queries(self):
        """7. Queries with no identifier constraints should match purely by semantic / lexical tokens."""
        aspect = "fuel servo"
        rec = "REMOVED AND REPLACED FUEL SERVO AFTER FLIGHT TEST."
        assert match_aspect_in_text(aspect, rec) is True

    def test_case_8_multiple_identifiers_in_one_query(self):
        """8. Multiple identifiers in one query: e.g. cylinder 2 and 2700 RPM."""
        aspect = "CYL 2 at 2700 RPM"
        rec_mismatch_rpm = "CYL 2 MISSING FIRE AT 2400 RPM."
        rec_mismatch_cyl = "CYL 4 MISSING FIRE AT 2700 RPM."
        rec_both_match = "FOUND CYL 2 SPARK PLUG MISFIRING AT 2700 RPM RUN UP."

        assert match_aspect_in_text(aspect, rec_mismatch_rpm) is False
        assert match_aspect_in_text(aspect, rec_mismatch_cyl) is False
        assert match_aspect_in_text(aspect, rec_both_match) is True


class TestFourAspectCoverageGate:
    """Regression test suite for Section 3: Four-Aspect Evidence Coverage."""

    def test_aspect_status_contradicted(self):
        """Verifies that an identifier constraint mismatch generates 'contradicted' status."""
        aspects = QueryAspects(
            component="CYL 2",
            symptom="fouled spark plug"
        )
        # Results contain only CYL 12
        results = [
            RetrievalResult(
                record_id="100012",
                rank=1,
                score=0.04,
                mode="hybrid",
                problem_raw="CYL 12 SPARK PLUG FOULED.",
                action_raw="REPLACED CYL 12 PLUG.",
                problem_normalized="CYL 12 SPARK PLUG FOULED.",
                action_normalized="REPLACED CYL 12 PLUG.",
                excerpt="CYL 12 SPARK PLUG FOULED. REPLACED CYL 12 PLUG."
            )
        ]

        cov_res = check_aspect_coverage(
            query="CYL 2 spark plug fouled",
            aspects=aspects,
            deduped_results=results,
            clusters={"100012": ["100012"]}
        )

        assert cov_res.aspect_coverage["component"] is False
        assert "component" in cov_res.contradicted_aspects
        assert cov_res.aspect_details["component"].status == "contradicted"
        assert "100012" in cov_res.aspect_details["component"].contradicting_record_ids

    def test_aspect_status_supported_and_not_applicable(self):
        """Verifies supported aspect status and not_applicable for unspecified aspects."""
        aspects = QueryAspects(
            component="CYL 2 spark plug",
            symptom="fouled",
            condition=None,
            requested_info=None
        )
        results = [
            RetrievalResult(
                record_id="100004",
                rank=1,
                score=0.05,
                mode="hybrid",
                problem_raw="FOUND CYL 2 LOWER PLUG FOULED.",
                action_raw="CLEANED AND RE-GAPPED CYL 2 PLUG.",
                problem_normalized="FOUND CYL 2 LOWER PLUG FOULED.",
                action_normalized="CLEANED AND RE-GAPPED CYL 2 PLUG.",
                excerpt="FOUND CYL 2 LOWER PLUG FOULED. CLEANED AND RE-GAPPED CYL 2 PLUG."
            )
        ]

        cov_res = check_aspect_coverage(
            query="CYL 2 spark plug fouled",
            aspects=aspects,
            deduped_results=results,
            clusters={"100004": ["100004"]}
        )

        assert cov_res.aspect_details["component"].status == "supported"
        assert cov_res.aspect_details["symptom"].status == "supported"
        assert cov_res.aspect_details["condition"].status == "not_applicable"
        assert cov_res.aspect_details["requested_info"].status == "not_applicable"
        assert "100004" in cov_res.aspect_details["component"].supporting_record_ids

    def test_gate_abstains_on_contradiction(self):
        """Verifies that evidence assessment abstains with status='insufficient' when constraints conflict."""
        results = [
            RetrievalResult(
                record_id="100012",
                rank=1,
                score=0.04,
                mode="hybrid",
                problem_raw="CYL 12 SPARK PLUG FOULED.",
                action_raw="REPLACED CYL 12 PLUG.",
                problem_normalized="CYL 12 SPARK PLUG FOULED.",
                action_normalized="REPLACED CYL 12 PLUG.",
                excerpt="CYL 12 SPARK PLUG FOULED."
            )
        ]
        assessment, _, _ = assess_retrieval_evidence(
            query="CYL 2 spark plug fouled",
            results=results,
            enable_aspect_gate=True
        )

        assert assessment.status == "insufficient"
        assert "contradict" in assessment.reason.lower() or "insufficient" in assessment.reason.lower()
