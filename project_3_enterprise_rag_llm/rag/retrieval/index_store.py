"""Index persistence: save/load chunks + embeddings; index_version manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class IndexManifest:
    index_version: str
    chunk_count: int
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    vector_backend: str = "faiss"  # faiss | qdrant


class LocalIndexStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def index_dir(self, index_version: str) -> Path:
        return self.base_dir / index_version

    def save(
        self,
        index_version: str,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray],
        embedding_model: Optional[str] = None,
        vector_backend: str = "faiss",
    ) -> str:
        d = self.index_dir(index_version)
        d.mkdir(parents=True, exist_ok=True)

        # chunks.jsonl (always)
        chunks_path = d / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        use_qdrant = vector_backend == "qdrant"
        if embeddings is not None and use_qdrant:
            from rag.retrieval.qdrant_store import upsert as qdrant_upsert
            qdrant_upsert(index_version, chunks, embeddings)
        elif embeddings is not None:
            np.save(d / "embeddings.npy", embeddings)

        emb_dim = int(embeddings.shape[1]) if embeddings is not None and len(embeddings.shape) == 2 else None
        manifest = IndexManifest(
            index_version=index_version,
            chunk_count=len(chunks),
            embedding_model=embedding_model,
            embedding_dim=emb_dim,
            vector_backend=vector_backend,
        )
        (d / "index_manifest.json").write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")
        (self.base_dir / "LATEST").write_text(index_version, encoding="utf-8")
        return str(d)

    def load(self, index_version: Optional[str] = None) -> Dict[str, Any]:
        if index_version is None:
            latest = self.base_dir / "LATEST"
            if not latest.exists():
                raise FileNotFoundError(f"No LATEST index in {self.base_dir}")
            index_version = latest.read_text(encoding="utf-8").strip()

        d = self.index_dir(index_version)
        if not d.exists():
            raise FileNotFoundError(f"Index not found: {d}")

        chunks = []
        with (d / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))

        manifest_path = d / "index_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        vector_backend = manifest.get("vector_backend", "faiss")

        embeddings = None
        if vector_backend != "qdrant":
            emb_path = d / "embeddings.npy"
            embeddings = np.load(emb_path) if emb_path.exists() else None

        return {
            "index_version": index_version,
            "chunks": chunks,
            "embeddings": embeddings,
            "manifest": manifest,
            "vector_backend": vector_backend,
            "dir": str(d),
        }

