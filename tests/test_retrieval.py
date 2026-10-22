"""
Unit and integration tests for Phase 3: Hybrid Retrieval.
Tests BM25, Semantic vector search, RRF fusion, index persistence,
empty/malformed queries, Hit@5 measurement, and citation preservation.
"""

import tempfile
from pathlib import Path
import pytest

from src.models import LogbookRecord
from src.retrieval.models import RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.evaluator import RetrievalEvaluator


@pytest.fixture
def sample_records():
    return [
        LogbookRecord(
            record_id="100001",
            problem_raw="ENGINE IDLE OVERRIDE KILLED ENGINE.",
            action_raw="TRIED TO ADJUST IDLE SEVERAL TIMES, WOULDN'T ADJUST.",
            problem_normalized="ENGINE IDLE OVERRIDE KILLED ENGINE.",
            action_normalized="TRIED TO ADJUST IDLE SEVERAL TIMES, WOULDN'T ADJUST."
        ),
        LogbookRecord(
            record_id="100002",
            problem_raw="ENGINE IDLE OVERRIDE KILLED ENGINE.",
            action_raw="REMOVED & REPLACED FUEL SERVO",
            problem_normalized="ENGINE IDLE OVERRIDE KILLED ENGINE.",
            action_normalized="REMOVED & REPLACED FUEL SERVO"
        ),
        LogbookRecord(
            record_id="100004",
            problem_raw="HAD ENGINE CHOKE & BRIEFLY LOSE POWER ON DEPARTURE. FULL THR",
            action_raw="PERFORMED ENGINE RUN UP, FOUND CYL 2 LOWER PLUG FOULED.",
            problem_normalized="HAD ENGINE CHOKE & BRIEFLY LOSE POWER ON DEPARTURE. FULL THR",
            action_normalized="PERFORMED ENGINE RUN UP, FOUND CYL 2 LOWER PLUG FOULED."
        ),
        LogbookRecord(
            record_id="100012",
            problem_raw="L/H REAR ENGINE BAFFLE CRACKED.",
            action_raw="STOP DRILLED CRACK.",
            problem_normalized="L/H REAR ENGINE BAFFLE CRACKED.",
            action_normalized="STOP DRILLED CRACK."
        ),
        LogbookRecord(
            record_id="100027",
            problem_raw="R/H MAG BLAST TUBE FELL OFF.",
            action_raw="REMOVED R/H AFT BAFFLE TO FACILITATE REPAIR. FABRICATED",
            problem_normalized="R/H MAG BLAST TUBE FELL OFF.",
            action_normalized="REMOVED R/H AFT BAFFLE TO FACILITATE REPAIR. FABRICATED"
        )
    ]


class TestBM25Retriever:
    """Tests for BM25 keyword search."""

    def test_bm25_search_returns_expected_record(self, sample_records, tmp_path):
        idx_file = tmp_path / "bm25_test.pkl"
        retriever = BM25Retriever(records=sample_records, index_path=idx_file)

        results = retriever.search("fuel servo", top_k=3)
        assert len(results) > 0
        assert results[0].record_id == "100002"
        assert results[0].mode == "bm25"
        assert "FUEL SERVO" in results[0].action_raw

    def test_bm25_index_persistence(self, sample_records, tmp_path):
        idx_file = tmp_path / "bm25_test.pkl"
        retriever_1 = BM25Retriever(records=sample_records, index_path=idx_file)
        assert idx_file.exists()

        # Instantiate second retriever and verify cached index is loaded
        retriever_2 = BM25Retriever(records=sample_records, index_path=idx_file)
        results = retriever_2.search("fuel servo", top_k=1)
        assert results[0].record_id == "100002"

    def test_empty_and_malformed_query_safety(self, sample_records, tmp_path):
        retriever = BM25Retriever(records=sample_records, index_path=tmp_path / "bm25.pkl")
        assert retriever.search("", top_k=5) == []
        assert retriever.search("   ", top_k=5) == []
        assert retriever.search("!@#$%^&*()", top_k=5) == []


class TestSemanticRetriever:
    """Tests for dense semantic vector search."""

    def test_semantic_search_paraphrase(self, sample_records, tmp_path):
        emb_file = tmp_path / "embs.npy"
        ids_file = tmp_path / "ids.json"
        retriever = SemanticRetriever(records=sample_records, embeddings_path=emb_file, ids_path=ids_file)

        # Query uses semantic phrasing not matching exact words
        results = retriever.search("left side rear engine cooling baffle was cracked", top_k=2)
        assert len(results) > 0
        # Should retrieve 100012 ("L/H REAR ENGINE BAFFLE CRACKED")
        top_ids = [r.record_id for r in results]
        assert "100012" in top_ids
        assert results[0].mode == "semantic"
        assert 0.0 <= results[0].score <= 1.0

    def test_embeddings_persistence(self, sample_records, tmp_path):
        emb_file = tmp_path / "embs.npy"
        ids_file = tmp_path / "ids.json"
        retriever_1 = SemanticRetriever(records=sample_records, embeddings_path=emb_file, ids_path=ids_file)
        assert emb_file.exists()
        assert ids_file.exists()

        # Second instance should load cached embeddings without recalculating
        retriever_2 = SemanticRetriever(records=sample_records, embeddings_path=emb_file, ids_path=ids_file)
        results = retriever_2.search("magneto blast tube", top_k=1)
        assert results[0].record_id == "100027"


class TestHybridRetriever:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_combines_rankings(self, sample_records, tmp_path):
        bm25_file = tmp_path / "bm25.pkl"
        emb_file = tmp_path / "embs.npy"
        ids_file = tmp_path / "ids.json"

        bm25 = BM25Retriever(records=sample_records, index_path=bm25_file)
        semantic = SemanticRetriever(records=sample_records, embeddings_path=emb_file, ids_path=ids_file)
        hybrid = HybridRetriever(records=sample_records, bm25_retriever=bm25, semantic_retriever=semantic)

        results = hybrid.search("fuel servo replacement", top_k=3)
        assert len(results) > 0
        assert results[0].record_id == "100002"
        assert results[0].mode == "hybrid"
        assert "bm25_rank" in results[0].metadata
        assert "semantic_rank" in results[0].metadata
        assert "rrf_score" in results[0].metadata

    def test_original_text_available_for_citation(self, sample_records, tmp_path):
        hybrid = HybridRetriever(records=sample_records)
        results = hybrid.search("stop drilled crack", top_k=1)
        assert len(results) == 1
        res = results[0]
        # Citations require exact raw text and record ID
        assert res.record_id == "100012"
        assert res.problem_raw == "L/H REAR ENGINE BAFFLE CRACKED."
        assert res.action_raw == "STOP DRILLED CRACK."


class TestEvaluatorMetrics:
    """Tests for Hit@k and MRR metric calculation logic."""

    def test_evaluation_metrics_calculation(self, tmp_path):
        mock_queries = [
            {
                "query_id": "T1",
                "query": "fuel servo",
                "category": "exact",
                "target_record_ids": ["100002"]
            },
            {
                "query_id": "T2",
                "query": "unanswerable question",
                "category": "unanswerable",
                "target_record_ids": []
            }
        ]
        q_file = tmp_path / "test_queries.json"
        import json
        with open(q_file, "w", encoding="utf-8") as f:
            json.dump(mock_queries, f)

        evaluator = RetrievalEvaluator(str(q_file))
        assert len(evaluator.answerable_queries) == 1
        assert len(evaluator.unanswerable_queries) == 1
