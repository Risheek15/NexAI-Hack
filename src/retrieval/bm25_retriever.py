"""
BM25 Keyword Retrieval implementation using rank-bm25.
Indexes normalized logbook records, persists index structures to disk,
and supports fast top-k keyword search with debugging rankings.
"""

import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi
from src.models import LogbookRecord
from src.retrieval.models import RetrievalResult


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9#/'\-]+")


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into technical and alphanumeric tokens.
    Requires at least one alphanumeric character per token so bare punctuation is ignored.
    """
    if not text:
        return []
    raw_tokens = _TOKEN_PATTERN.findall(text)
    return [t.lower() for t in raw_tokens if any(c.isalnum() for c in t)]


class BM25Retriever:
    """
    Inverted-index BM25 retrieval engine over normalized logbook records.
    """

    def __init__(
        self,
        records: Optional[List[LogbookRecord]] = None,
        index_path: str | Path = "data/processed/bm25_index.pkl",
        use_expanded_abbreviations: bool = True
    ):
        self.index_path = Path(index_path)
        self.use_expanded_abbreviations = use_expanded_abbreviations
        self.records: List[LogbookRecord] = records or []
        self.records_by_id: Dict[str, LogbookRecord] = {r.record_id: r for r in self.records}
        self.record_ids: List[str] = [r.record_id for r in self.records]
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

        if self.records:
            self._build_or_load_index()

    def _build_or_load_index(self) -> None:
        """Loads index from disk if valid and record count matches, else builds and saves."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    if len(data.get("record_ids", [])) == len(self.records):
                        self.record_ids = data["record_ids"]
                        self.bm25 = data["bm25"]
                        self.tokenized_corpus = data["tokenized_corpus"]
                        return
            except Exception as e:
                print(f"[WARNING] Failed to load cached BM25 index: {e}. Rebuilding...")

        # Build new index
        self.tokenized_corpus = [
            tokenize(r.combined_normalized)
            for r in self.records
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.record_ids = [r.record_id for r in self.records]

        # Persist index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "record_ids": self.record_ids,
                "bm25": self.bm25,
                "tokenized_corpus": self.tokenized_corpus
            }, f)

    def add_record(self, record: LogbookRecord) -> None:
        """Dynamically indexes a newly approved record into the BM25 model."""
        if record.record_id in self.records_by_id:
            return
        self.records.append(record)
        self.records_by_id[record.record_id] = record
        self.record_ids.append(record.record_id)
        tokens = tokenize(record.combined_normalized)
        self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Executes BM25 search and returns ranked matching records.
        """
        if not self.bm25 or not self.records:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[RetrievalResult] = []
        rank = 1
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            # Only return positive matches; if BM25 score is 0, document has 0 matching terms
            if score <= 0.0:
                break

            rec_id = self.record_ids[idx]
            rec = self.records_by_id[rec_id]

            results.append(RetrievalResult(
                record_id=rec_id,
                rank=rank,
                score=score,
                mode="bm25",
                problem_raw=rec.problem_raw,
                action_raw=rec.action_raw,
                problem_normalized=rec.problem_normalized,
                action_normalized=rec.action_normalized,
                excerpt=rec.combined_normalized[:120],
                metadata={"bm25_score": score, "bm25_rank": rank}
            ))
            rank += 1

        return results
