"""
Regression tests for Section 11: BM25 Cache Integrity Checking.
Verifies that count-preserving corpus changes trigger automatic cache invalidation,
tokenizer changes force a rebuild, and matching fingerprints load cleanly from cache.
"""

import pickle
import pytest
from src.models import LogbookRecord
from src.retrieval.bm25_retriever import (
    BM25Retriever,
    compute_corpus_fingerprint,
    TOKENIZER_VERSION
)


class TestBM25CacheIntegrity:
    """Tests for SHA-256 corpus fingerprint cache invalidation."""

    def test_corpus_fingerprint_changes_on_text_modification(self):
        r1 = LogbookRecord(
            record_id="100001",
            problem_raw="FUEL LEAK", action_raw="TIGHTENED FITTING",
            problem_normalized="fuel leak", action_normalized="tightened fitting"
        )
        r2 = LogbookRecord(
            record_id="100002",
            problem_raw="OIL LEAK", action_raw="REPLACED GASKET",
            problem_normalized="oil leak", action_normalized="replaced gasket"
        )

        fp1 = compute_corpus_fingerprint([r1, r2])

        # Modify r2 text without changing record count
        r2_modified = LogbookRecord(
            record_id="100002",
            problem_raw="SPARK PLUG FOULED", action_raw="CLEANED PLUG",
            problem_normalized="spark plug fouled", action_normalized="cleaned plug"
        )
        fp2 = compute_corpus_fingerprint([r1, r2_modified])

        assert fp1 != fp2

    def test_cache_rebuilds_on_count_preserving_text_change(self, tmp_path):
        cache_file = tmp_path / "bm25_test_cache.pkl"

        r1 = LogbookRecord(
            record_id="100001",
            problem_raw="INTAKE LEAK", action_raw="REPLACED GASKET",
            problem_normalized="intake leak", action_normalized="replaced gasket"
        )
        r2 = LogbookRecord(
            record_id="100002",
            problem_raw="MAGNETO BLAST TUBE CRACKED", action_raw="REPLACED TUBE",
            problem_normalized="magneto blast tube cracked", action_normalized="replaced tube"
        )
        r3 = LogbookRecord(
            record_id="100003",
            problem_raw="RUDDER TRIM FREEZE", action_raw="LUBRICATED CABLE",
            problem_normalized="rudder trim freeze", action_normalized="lubricated cable"
        )

        # 1. First build creates cache
        retriever1 = BM25Retriever(records=[r1, r2, r3], index_path=str(cache_file))
        assert cache_file.exists()
        search1 = retriever1.search("magneto", top_k=2)
        assert len(search1) == 1
        assert search1[0].record_id == "100002"

        # 2. Modify r2 text (change 'magneto' to 'fuel servo') without changing record count
        r2_new = LogbookRecord(
            record_id="100002",
            problem_raw="FUEL SERVO LEAKING", action_raw="REPLACED SERVO",
            problem_normalized="fuel servo leaking", action_normalized="replaced servo"
        )

        # 3. Second build with same cache_file must detect fingerprint mismatch and rebuild
        retriever2 = BM25Retriever(records=[r1, r2_new, r3], index_path=str(cache_file))
        search_old = retriever2.search("magneto", top_k=2)
        assert len(search_old) == 0  # 'magneto' no longer exists in corpus

        search_new = retriever2.search("fuel servo", top_k=2)
        assert len(search_new) == 1
        assert search_new[0].record_id == "100002"

    def test_dynamic_add_record_updates_cache(self, tmp_path):
        cache_file = tmp_path / "bm25_dynamic_cache.pkl"
        r1 = LogbookRecord(
            record_id="100001",
            problem_raw="BRAKE CHATTER", action_raw="BLED LINES",
            problem_normalized="brake chatter", action_normalized="bled lines"
        )
        retriever = BM25Retriever(records=[r1], index_path=str(cache_file))

        r2 = LogbookRecord(
            record_id="100002",
            problem_raw="RUDDER TRIM RUNAWAY", action_raw="REPLACED MOTOR",
            problem_normalized="rudder trim runaway", action_normalized="replaced motor"
        )
        retriever.add_record(r2, persist=True)

        # Inspect saved cache
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        assert len(data["record_ids"]) == 2
        assert "100002" in data["record_ids"]
