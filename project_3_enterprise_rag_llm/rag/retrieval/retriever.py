"""Retrieval: BM25, dense (FAISS/Qdrant), rerank; versioning."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rag.retrieval.index_store import LocalIndexStore


@dataclass
class RetrievedChunk:
    """Retrieved chunk with score."""
    chunk_id: str
    content: str
    source_id: str
    page: Optional[int]
    section: Optional[str]
    score: float
    rank: int


class RetrieverStack:
    """
    BM25 / Dense / Hybrid + optional Rerank.
    Versioning: index_version, retriever_version (optional).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        index_version: Optional[str] = None,
        retriever_version: Optional[str] = None,
        index_store_dir: Optional[str] = None,
        auto_load: bool = True,
    ):
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f) or {}
        self.index_version_str = index_version or f"index_v{datetime.utcnow().strftime('%Y_%m_%d')}"
        self.retriever_version = retriever_version or "retriever_v1"
        self.index_store = LocalIndexStore(index_store_dir or str(Path(__file__).resolve().parents[2] / "data" / "index"))
        self._bm25 = None
        self._dense = None
        self._rerank = None
        self._index = None
        self._use_qdrant = False
        self._qdrant_version = None
        self._vector_backend = (self.config.get("dense") or {}).get("index_type", "qdrant")
        if auto_load:
            try:
                self.load_index()
            except Exception:
                # It's OK to start without index; service can build one later.
                pass

    def build_index(self, chunks: List[Dict[str, Any]]) -> str:
        """Build BM25 and/or dense index from chunks. Returns index_version."""
        # BM25
        corpus = [c["content"] for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [t.split() for t in corpus]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            pass
        # Dense (optional): sentence-transformers + FAISS
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                self.config.get("dense", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            )
            embs = model.encode(corpus, show_progress_bar=False)
            self._dense = {"model": model, "embeddings": np.asarray(embs), "chunks": chunks}
        except ImportError:
            pass
        self._index = chunks
        # Persist (local + optional Qdrant)
        self.index_store.save(
            index_version=self.index_version_str,
            chunks=chunks,
            embeddings=(self._dense["embeddings"] if self._dense else None),
            embedding_model=(self.config.get("dense", {}).get("model_name") if self._dense else None),
            vector_backend=self._vector_backend,
        )
        return self.index_version_str

    def load_index(self, index_version: Optional[str] = None) -> str:
        """Load persisted index into memory."""
        loaded = self.index_store.load(index_version=index_version)
        self.index_version_str = loaded["index_version"]
        chunks = loaded["chunks"]
        self._index = chunks

        # BM25
        try:
            from rank_bm25 import BM25Okapi
            corpus = [c["content"] for c in chunks]
            tokenized = [t.split() for t in corpus]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

        # Dense: in-memory (faiss) or Qdrant
        vector_backend = loaded.get("vector_backend", "faiss")
        embeddings = loaded.get("embeddings")
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("dense", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name)
        except ImportError:
            model = None

        if vector_backend == "qdrant" and model is not None:
            self._use_qdrant = True
            self._qdrant_version = loaded["index_version"]
            emb_dim = (loaded.get("manifest") or {}).get("embedding_dim") or 384
            self._dense = {"model": model, "chunks": chunks, "embedding_dim": emb_dim}
        elif embeddings is not None and model is not None:
            self._use_qdrant = False
            self._qdrant_version = None
            self._dense = {"model": model, "embeddings": embeddings, "chunks": chunks}
        else:
            self._dense = None

        return self.index_version_str

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        mode: str = "hybrid"
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks. mode: bm25_only | dense_only | hybrid | hybrid_rerank.
        Fallback: when _index exists but _bm25/_dense unavailable, use simple keyword match.
        """
        results = []
        if mode == "bm25_only" and self._bm25:
            scores = self._bm25.get_scores(query.split())
            order = np.argsort(scores)[::-1][:top_k]
            for i, idx in enumerate(order):
                c = self._index[idx]
                results.append(RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    source_id=c["source_id"],
                    page=c.get("page"),
                    section=c.get("section"),
                    score=float(scores[idx]),
                    rank=i + 1
                ))
        elif mode == "dense_only" and self._dense:
            model = self._dense["model"]
            qemb = model.encode([query], show_progress_bar=False)
            qvec = qemb.flatten().tolist()
            if self._use_qdrant:
                from rag.retrieval.qdrant_store import search as qdrant_search
                hits = qdrant_search(self._qdrant_version, qvec, top_k)
                for i, h in enumerate(hits):
                    results.append(RetrievedChunk(
                        chunk_id=h.get("chunk_id", ""),
                        content=h.get("content", ""),
                        source_id=h.get("source_id", ""),
                        page=h.get("page"),
                        section=h.get("section"),
                        score=float(h.get("_score", 0.0)),
                        rank=i + 1,
                    ))
            else:
                sim = np.dot(self._dense["embeddings"], np.asarray(qvec).reshape(-1, 1)).flatten()
                order = np.argsort(sim)[::-1][:top_k]
                for i, idx in enumerate(order):
                    c = self._dense["chunks"][idx]
                    results.append(RetrievedChunk(
                        chunk_id=c["chunk_id"],
                        content=c["content"],
                        source_id=c["source_id"],
                        page=c.get("page"),
                        section=c.get("section"),
                        score=float(sim[idx]),
                        rank=i + 1,
                    ))
        elif mode in ("hybrid", "hybrid_rerank"):
            # Placeholder: use bm25 or dense if only one available
            if self._bm25:
                scores = self._bm25.get_scores(query.split())
                order = np.argsort(scores)[::-1][:top_k * 2]
                for i, idx in enumerate(order[:top_k]):
                    c = self._index[idx]
                    results.append(RetrievedChunk(
                        chunk_id=c["chunk_id"],
                        content=c["content"],
                        source_id=c["source_id"],
                        page=c.get("page"),
                        section=c.get("section"),
                        score=float(scores[idx]),
                        rank=i + 1
                    ))
            elif self._dense:
                model = self._dense["model"]
                qemb = model.encode([query], show_progress_bar=False)
                qvec = qemb.flatten().tolist()
                if self._use_qdrant:
                    from rag.retrieval.qdrant_store import search as qdrant_search
                    hits = qdrant_search(self._qdrant_version, qvec, top_k)
                    for i, h in enumerate(hits):
                        results.append(RetrievedChunk(
                            chunk_id=h.get("chunk_id", ""),
                            content=h.get("content", ""),
                            source_id=h.get("source_id", ""),
                            page=h.get("page"),
                            section=h.get("section"),
                            score=float(h.get("_score", 0.0)),
                            rank=i + 1,
                        ))
                else:
                    sim = np.dot(self._dense["embeddings"], np.asarray(qvec).reshape(-1, 1)).flatten()
                    order = np.argsort(sim)[::-1][:top_k]
                    for i, idx in enumerate(order):
                        c = self._dense["chunks"][idx]
                        results.append(RetrievedChunk(
                            chunk_id=c["chunk_id"],
                            content=c["content"],
                            source_id=c["source_id"],
                            page=c.get("page"),
                            section=c.get("section"),
                            score=float(sim[idx]),
                            rank=i + 1,
                        ))
            if mode == "hybrid_rerank" and self._rerank and results:
                # Rerank top_k * 2 -> top_k
                pass
        # Fallback: when _index has data but BM25/dense are not installed, use simple keyword match
        if not results and self._index and len(self._index) > 0:
            q_lower = query.lower()
            q_words = set(q_lower.split())
            scored = []
            for c in self._index:
                text = (c.get("content") or "").lower()
                score = sum(1 for w in q_words if w in text)
                if score > 0:
                    scored.append((score, c))
            scored.sort(key=lambda x: -x[0])
            for i, (score, c) in enumerate(scored[:top_k]):
                results.append(RetrievedChunk(
                    chunk_id=c.get("chunk_id", ""),
                    content=c.get("content", ""),
                    source_id=c.get("source_id", ""),
                    page=c.get("page"),
                    section=c.get("section"),
                    score=float(score),
                    rank=i + 1,
                ))
        return results

    @property
    def index_version(self) -> str:
        return self.index_version_str
