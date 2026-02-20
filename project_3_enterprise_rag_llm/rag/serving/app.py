"""
RAG Serving (D9)
POST /ask, /retrieve, /evaluate/run, /feedback.
Reuse RequestMeta; /ask schema defined within project.
"""

import sys
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add ds_platform (base platform) and project_3_enterprise_rag_llm root to path
_project_root = Path(__file__).parent.parent.parent   # project_3_enterprise_rag_llm
_workspace_root = _project_root.parent                 # 2.ds
_platform_path = _workspace_root / "ds_platform"
for p in (str(_platform_path), str(_project_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import HTTPException, Request
from pydantic import BaseModel

from platform_sdk.serving.app_factory import create_app
from platform_sdk.serving.metrics import add_metrics_endpoint
from platform_sdk.schemas.api_common import RequestMeta
from platform_sdk.schemas.audit import AuditRecord
from platform_sdk.db.audit_writer import write_audit_async
from platform_sdk.serving.async_queue import AsyncJobManager

from rag.serving.rag_metrics import observe_ask

# Lazy imports for RAG components
_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    from rag.retrieval.retriever import RetrieverStack
    from rag.generation.router import Router
    from rag.generation.pipeline import RAGPipeline
    from rag.generation.llm_client import OpenAICompatChatClient
    config_dir = Path(__file__).parent.parent / "configs"
    retriever = RetrieverStack(
        config_path=str(config_dir / "retrieval.yaml"),
        index_store_dir=str(Path(__file__).resolve().parents[2] / "data" / "index"),
        auto_load=True,
    )
    router = Router()
    import os
    base_url = os.getenv("RAG_LLM_BASE_URL", "").strip()
    llm = OpenAICompatChatClient() if base_url else None
    _PIPELINE = RAGPipeline(retriever=retriever, router=router, llm_client=llm)
    return _PIPELINE

def _load_eval_set(eval_set_path: str):
    from rag.evaluation.io import load_eval_set_jsonl
    return load_eval_set_jsonl(eval_set_path)

job_manager = AsyncJobManager()


app = create_app(
    title="Enterprise RAG Service",
    version="1.0.0",
    domain="rag"
)
add_metrics_endpoint(app)


# --- Schemas (D9, reuse RequestMeta) ---

class AskRequest(BaseModel):
    query: str
    top_k: int = 6
    mode: str = "hybrid"
    meta: Optional[RequestMeta] = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunk_ids: List[str]
    index_version: str
    prompt_version: str
    latency_ms: int
    no_evidence: bool


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 6
    mode: str = "hybrid"


class RetrieveResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    index_version: str


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int  # 1-5
    reason: Optional[str] = None
    notes: Optional[str] = None


class EvaluateRunRequest(BaseModel):
    eval_set_path: Optional[str] = None
    modes: List[str] = ["bm25_only", "dense_only", "hybrid"]
    gate_mode: str = "hybrid"  # Which mode to use as final gate metric (default hybrid)
    callback_url: Optional[str] = None


# --- Handlers ---

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    """POST /ask: query → answer + citations. Audit index_version / prompt_version / retrieved_chunk_ids."""
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    trace_id = getattr(request.state, "trace_id", None)
    try:
        pipeline = _get_pipeline()
        resp = pipeline.ask(req.query, top_k=req.top_k, mode=req.mode)
        await write_audit_async(
            AuditRecord(
                request_id=request_id,
                domain="rag",
                model_name="rag_answerer",
                model_version=resp.prompt_version,
                latency_ms=resp.latency_ms,
                predictions={
                    "answer": resp.answer,
                    "citations": [{"chunk_id": c.chunk_id, "source_id": c.source_id, "page": c.page} for c in resp.citations],
                    "retrieved_chunk_ids": resp.retrieved_chunk_ids,
                    "index_version": resp.index_version,
                    "prompt_version": resp.prompt_version,
                },
                decision={"no_evidence": resp.no_evidence},
                trace_id=trace_id,
            )
        )
        pt = getattr(resp, "prompt_tokens", 0) or 0
        ct = getattr(resp, "completion_tokens", 0) or 0
        observe_ask("rag", resp.latency_ms / 1000.0, pt, ct, resp.no_evidence)
        return AskResponse(
            request_id=request_id,
            answer=resp.answer,
            citations=[{"chunk_id": c.chunk_id, "source_id": c.source_id, "page": c.page, "excerpt": c.excerpt} for c in resp.citations],
            retrieved_chunk_ids=resp.retrieved_chunk_ids,
            index_version=resp.index_version,
            prompt_version=resp.prompt_version,
            latency_ms=resp.latency_ms,
            no_evidence=resp.no_evidence,
        )
    except Exception as e:
        logging.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """POST /retrieve: top-k chunks (debugging)."""
    try:
        pipeline = _get_pipeline()
        retriever = pipeline.retriever
        chunks = retriever.retrieve(req.query, top_k=req.top_k, mode=req.mode)
        return RetrieveResponse(
            chunks=[
                {"chunk_id": c.chunk_id, "content": c.content, "source_id": c.source_id, "page": c.page, "score": c.score, "rank": c.rank}
                for c in chunks
            ],
            index_version=getattr(retriever, "index_version_str", "unknown"),
        )
    except Exception as e:
        logging.exception("retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/run")
async def evaluate_run(req: EvaluateRunRequest):
    """POST /evaluate/run: Trigger evaluation (async). Returns job_id."""
    try:
        if not req.eval_set_path:
            return {"status": "ok", "message": "eval_set_path required (jsonl)", "metrics": {}}
        payload = {"eval_set_path": req.eval_set_path, "modes": req.modes, "gate_mode": req.gate_mode}
        job_id = job_manager.enqueue_job(domain="rag", task_name="rag.evaluate_run", payload=payload, callback_url=req.callback_url)
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        logging.exception("evaluate/run failed")
        raise HTTPException(status_code=500, detail=str(e))


class IndexBuildRequest(BaseModel):
    chunks_path: str  # One of jsonl/parquet (current implementation supports jsonl)


@app.post("/index/build")
async def index_build(req: IndexBuildRequest):
    """
    Build index (minimal loop for D13 demo).
    Expects chunks_path to be jsonl, each line contains: chunk_id, content, source_id, page, section, etc.
    """
    try:
        import json
        from pathlib import Path as _Path
        p = _Path(req.chunks_path)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"chunks_path not found: {req.chunks_path}")
        chunks = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        pipeline = _get_pipeline()
        ver = pipeline.retriever.build_index(chunks)
        return {"status": "ok", "index_version": ver, "chunk_count": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("index/build failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Query async job status (A4.1)."""
    try:
        return job_manager.get_job_status(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """POST /feedback: Write to rag.feedback."""
    try:
        from platform_sdk.db.pg import get_db
        from sqlalchemy import text
        db = get_db()
        session = db.get_session()
        try:
            session.execute(
                text("""
                    INSERT INTO rag.feedback (request_id, rating, reason, notes)
                    VALUES (:request_id, :rating, :reason, :notes)
                """),
                {"request_id": req.request_id, "rating": req.rating, "reason": req.reason, "notes": req.notes}
            )
            session.commit()
            return {"status": "ok", "request_id": req.request_id}
        finally:
            session.close()
    except Exception as e:
        logging.exception("feedback failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
