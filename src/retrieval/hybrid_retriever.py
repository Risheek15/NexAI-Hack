"""
Hybrid Retrieval engine combining BM25 keyword matching and dense semantic search
using Reciprocal Rank Fusion (RRF). Preserves record IDs, original citations,
and provides rich debugging metadata for inspection.
"""

from collections import defaultdict
from typing import List, Dict, Optional

from src.models import LogbookRecord
from src.retrieval.models import RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.semantic_retriever import SemanticRetriever


class HybridRetriever:
    """
    Combines sparse lexical (BM25) and dense semantic vector search via Reciprocal Rank Fusion.
    Formula: RRF_score(d) = sum(1 / (k + rank_m(d))) for m in [BM25, Semantic].
    """

    def __init__(
        self,
        records: List[LogbookRecord],
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_retriever: Optional[SemanticRetriever] = None,
        rrf_k: int = 60
    ):
        self.records = records
        self.records_by_id: Dict[str, LogbookRecord] = {r.record_id: r for r in self.records}
        self.rrf_k = rrf_k

        self.bm25 = bm25_retriever or BM25Retriever(records=records)
        self.semantic = semantic_retriever or SemanticRetriever(records=records)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Executes BM25 and Semantic search, fuses rankings with RRF, and returns top_k results.
        """
        if not query.strip():
            return []

        # Pull larger candidate pool for fusion
        candidate_pool_size = max(50, top_k * 3)

        bm25_results = self.bm25.search(query, top_k=candidate_pool_size)
        semantic_results = self.semantic.search(query, top_k=candidate_pool_size)

        rrf_scores: Dict[str, float] = defaultdict(float)
        bm25_ranks: Dict[str, int] = {}
        semantic_ranks: Dict[str, int] = {}

        # 1. Accumulate BM25 reciprocal ranks
        for res in bm25_results:
            rrf_scores[res.record_id] += 1.0 / (self.rrf_k + res.rank)
            bm25_ranks[res.record_id] = res.rank

        # 2. Accumulate Semantic reciprocal ranks
        for res in semantic_results:
            rrf_scores[res.record_id] += 1.0 / (self.rrf_k + res.rank)
            semantic_ranks[res.record_id] = res.rank

        # 3. Sort candidates by combined RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Construct standardized RetrievalResult models
        hybrid_results: List[RetrievalResult] = []
        rank = 1
        for rec_id, rrf_score in sorted_candidates[:top_k]:
            rec = self.records_by_id[rec_id]
            hybrid_results.append(RetrievalResult(
                record_id=rec_id,
                rank=rank,
                score=round(rrf_score, 6),
                mode="hybrid",
                problem_raw=rec.problem_raw,
                action_raw=rec.action_raw,
                problem_normalized=rec.problem_normalized,
                action_normalized=rec.action_normalized,
                excerpt=rec.combined_normalized[:120],
                metadata={
                    "rrf_score": round(rrf_score, 6),
                    "bm25_rank": bm25_ranks.get(rec_id, None),
                    "semantic_rank": semantic_ranks.get(rec_id, None),
                }
            ))
            rank += 1

        return hybrid_results
