"""
BM25 Keyword Retrieval implementation using rank-bm25.
Indexes normalized logbook records, persists index structures to disk,
checks cryptographic corpus fingerprints and tokenizer configuration to guarantee cache integrity,
and supports fast top-k keyword search with debugging rankings.
"""

import re
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi
from src.models import LogbookRecord
from src.retrieval.models import RetrievalResult

TOKENIZER_VERSION = "2.1-morphology-hyphen-split"

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9#/'\-]+")

_MORPHOLOGICAL_VARIANTS: Dict[str, List[str]] = {
    "fouling": ["fouled", "foul"],
    "fouled": ["fouling", "foul"],
    "foul": ["fouled", "fouling"],
    "spark-plug": ["spark", "plug", "sparkplug"],
    "sparkplug": ["spark", "plug"],
    "sparkplugs": ["spark", "plug", "plugs"],
    "spark-plugs": ["spark", "plug", "plugs"],
    "plugs": ["plug"],
    "plug": ["plugs"],
    "leaking": ["leak", "leaks", "leaked"],
    "leaked": ["leak", "leaking", "leaks"],
    "cracking": ["crack", "cracks", "cracked"],
    "cracked": ["crack", "cracks", "cracking"],
    "burning": ["burn", "burned"],
    "burned": ["burn", "burning"],
    "broken": ["break"],
    "missing": ["miss"],
    "failing": ["fail", "failed"],
    "failed": ["fail", "failing"],
    "vibrating": ["vibration", "vibrate"],
    "cylinders": ["cyl", "cylinder"],
    "cylinder": ["cyl", "cylinders"],
}


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into technical and alphanumeric tokens.
    Handles hyphenated compound technical tokens (e.g. spark-plug -> spark, plug)
    and expands morphological variants (e.g. fouling -> fouled, foul) for high-recall logbook matching.
    """
    if not text:
        return []
    raw_tokens = _TOKEN_PATTERN.findall(text)
    tokens: List[str] = []
    seen = set()

    for raw in raw_tokens:
        tl = raw.lower()
        if not any(c.isalnum() for c in tl):
            continue

        if tl not in seen:
            seen.add(tl)
            tokens.append(tl)

        # Split hyphenated terms
        if "-" in tl:
            for sub in tl.split("-"):
                if sub and any(c.isalnum() for c in sub) and sub not in seen:
                    seen.add(sub)
                    tokens.append(sub)

        # Morphological synonyms & variants
        if tl in _MORPHOLOGICAL_VARIANTS:
            for v in _MORPHOLOGICAL_VARIANTS[tl]:
                if v not in seen:
                    seen.add(v)
                    tokens.append(v)

    return tokens


def compute_corpus_fingerprint(records: List[LogbookRecord], tokenizer_version: str = TOKENIZER_VERSION) -> str:
    """Computes a stable SHA-256 fingerprint across all records and the tokenizer version."""
    hasher = hashlib.sha256()
    hasher.update(tokenizer_version.encode("utf-8"))
    for r in records:
        entry = f"{r.record_id}:{r.combined_normalized}\n"
        hasher.update(entry.encode("utf-8"))
    return hasher.hexdigest()


class BM25Retriever:
    """
    Inverted-index BM25 retrieval engine over normalized logbook records
    with SHA-256 fingerprint cache validation.
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
        self.corpus_fingerprint: str = ""

        if self.records:
            self._build_or_load_index()

    def _save_index(self, fingerprint: Optional[str] = None) -> None:
        """Persists the BM25 index with corpus content fingerprint."""
        fp = fingerprint or compute_corpus_fingerprint(self.records)
        self.corpus_fingerprint = fp
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "corpus_fingerprint": fp,
                "tokenizer_version": TOKENIZER_VERSION,
                "record_ids": self.record_ids,
                "bm25": self.bm25,
                "tokenized_corpus": self.tokenized_corpus
            }, f)

    def _build_or_load_index(self) -> None:
        """
        Loads index from disk if fingerprint and tokenizer match exactly.
        Invalidates and rebuilds automatically if record contents change or count changes.
        """
        current_fp = compute_corpus_fingerprint(self.records)

        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    cached_fp = data.get("corpus_fingerprint")
                    cached_tok_ver = data.get("tokenizer_version")
                    cached_ids = data.get("record_ids", [])

                    if (
                        cached_fp == current_fp
                        and cached_tok_ver == TOKENIZER_VERSION
                        and len(cached_ids) == len(self.records)
                    ):
                        self.record_ids = cached_ids
                        self.bm25 = data["bm25"]
                        self.tokenized_corpus = data["tokenized_corpus"]
                        self.corpus_fingerprint = cached_fp
                        return
                    else:
                        print("[INFO] BM25 cache invalidated due to corpus or tokenizer modification. Rebuilding...")
            except Exception as e:
                print(f"[WARNING] Failed to load cached BM25 index: {e}. Rebuilding...")

        # Build new index
        self.tokenized_corpus = [
            tokenize(r.combined_normalized)
            for r in self.records
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.record_ids = [r.record_id for r in self.records]

        # Persist index with fingerprint
        self._save_index(current_fp)

    def add_record(self, record: LogbookRecord, persist: bool = True) -> None:
        """Dynamically indexes a newly approved record into the BM25 model and updates disk cache."""
        if record.record_id in self.records_by_id:
            return
        self.records.append(record)
        self.records_by_id[record.record_id] = record
        self.record_ids.append(record.record_id)
        tokens = tokenize(record.combined_normalized)
        self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        if persist:
            try:
                self._save_index()
            except Exception:
                pass

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
