"""
Dense semantic retrieval implementation using sentence-transformers and NumPy.
Encodes normalized logbook records, caches embeddings to disk, and executes
fast vector search via normalized cosine similarity matrix operations.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import LogbookRecord
from src.retrieval.models import RetrievalResult


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class SemanticRetriever:
    """
    Dense semantic vector search engine over normalized logbook records.
    """

    def __init__(
        self,
        records: Optional[List[LogbookRecord]] = None,
        model_name: Optional[str] = None,
        embeddings_path: str | Path = "data/processed/embeddings.npy",
        ids_path: str | Path = "data/processed/embedding_ids.json"
    ):
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.embeddings_path = Path(embeddings_path)
        self.ids_path = Path(ids_path)
        self.records: List[LogbookRecord] = records or []
        self.records_by_id: Dict[str, LogbookRecord] = {r.record_id: r for r in self.records}
        self.record_ids: List[str] = [r.record_id for r in self.records]

        self._model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None

        if self.records:
            self._build_or_load_embeddings()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_or_load_embeddings(self) -> None:
        """Loads cached embeddings if valid, otherwise computes and saves."""
        if self.embeddings_path.exists() and self.ids_path.exists():
            try:
                with open(self.ids_path, "r", encoding="utf-8") as f:
                    saved_ids = json.load(f)
                if len(saved_ids) == len(self.records) and saved_ids == self.record_ids:
                    self.embeddings = np.load(self.embeddings_path)
                    return
            except Exception as e:
                print(f"[WARNING] Failed to load cached embeddings: {e}. Rebuilding...")

        # Compute embeddings across corpus
        texts = [r.combined_normalized for r in self.records]
        # normalize_embeddings=True gives unit vectors, meaning dot-product == cosine similarity
        raw_embs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        self.embeddings = np.asarray(raw_embs, dtype=np.float32)
        self.record_ids = [r.record_id for r in self.records]

        # Cache embeddings to disk
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.record_ids, f)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Encodes query and computes cosine similarities against all documents.
        """
        if not query.strip() or self.embeddings is None or not self.records:
            return []

        query_emb = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            normalize_embeddings=True
        )[0]

        # Dot product against all normalized embeddings: shape (N,)
        similarities = np.dot(self.embeddings, query_emb)

        # Get top_k indices
        top_k = min(top_k, len(similarities))
        # argpartition for fast top_k retrieval
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        # Sort top indices descending
        sorted_top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results: List[RetrievalResult] = []
        rank = 1
        for idx in sorted_top_indices:
            score = float(similarities[idx])
            rec_id = self.record_ids[idx]
            rec = self.records_by_id[rec_id]

            results.append(RetrievalResult(
                record_id=rec_id,
                rank=rank,
                score=score,
                mode="semantic",
                problem_raw=rec.problem_raw,
                action_raw=rec.action_raw,
                problem_normalized=rec.problem_normalized,
                action_normalized=rec.action_normalized,
                excerpt=rec.combined_normalized[:120],
                metadata={"cosine_similarity": score, "semantic_rank": rank}
            ))
            rank += 1

        return results
