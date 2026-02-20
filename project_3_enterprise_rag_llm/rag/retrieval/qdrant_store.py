"""
Qdrant vector store (production).
Collections = index_version. Upsert embeddings + payload (chunk_id, source_id, page, etc.).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

# Optional: use platform Config when in monorepo
try:
    from platform_sdk.common.config import Config as PlatformConfig
except Exception:
    PlatformConfig = None


def _qdrant_client():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
    except ImportError:
        raise ImportError("qdrant-client required: pip install qdrant-client")

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY")
    if PlatformConfig:
        host = PlatformConfig.QDRANT_HOST
        port = PlatformConfig.QDRANT_PORT
        api_key = PlatformConfig.QDRANT_API_KEY

    return QdrantClient(host=host, port=port, api_key=api_key)


def collection_name(index_version: str) -> str:
    """Safe collection name (alphanumeric + underscore)."""
    return "rag_" + "".join(c if c.isalnum() or c == "_" else "_" for c in index_version)


def ensure_collection(client, coll: str, dim: int):
    from qdrant_client.http import models as qmodels
    collections = [c.name for c in client.get_collections().collections]
    if coll in collections:
        return
    client.create_collection(
        coll,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )


def upsert(
    index_version: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    """Upsert chunks + embeddings into Qdrant collection = index_version."""
    client = _qdrant_client()
    from qdrant_client.http import models as qmodels

    coll = collection_name(index_version)
    dim = int(embeddings.shape[1])
    ensure_collection(client, coll, dim)

    points = []
    for i, (c, emb) in enumerate(zip(chunks, embeddings)):
        payload = {
            "chunk_id": c.get("chunk_id", ""),
            "source_id": c.get("source_id", ""),
            "page": c.get("page"),
            "section": c.get("section"),
            "content": c.get("content", "")[:64_000],  # avoid huge payloads
        }
        points.append(
            qmodels.PointStruct(id=i, vector=emb.tolist(), payload=payload)
        )
    client.upsert(coll, points=points)


def search(
    index_version: str,
    query_vector: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Search Qdrant; return list of payloads with score, ordered by relevance."""
    client = _qdrant_client()
    from qdrant_client.http import models as qmodels

    coll = collection_name(index_version)
    collections = [c.name for c in client.get_collections().collections]
    if coll not in collections:
        return []

    res = client.search(
        collection_name=coll,
        query_vector=query_vector,
        limit=top_k,
    )
    out = []
    for r in res:
        payload = dict(r.payload or {})
        payload["_score"] = r.score
        payload["_id"] = r.id
        out.append(payload)
    return out
