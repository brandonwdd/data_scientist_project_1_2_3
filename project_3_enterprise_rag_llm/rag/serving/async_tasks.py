"""
Async tasks for RAG (Production)

- rag.evaluate_run: run eval + gate, persist result in platform.async_jobs.result
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure platform_sdk and project_3_enterprise_rag_llm are importable
_base = Path(__file__).parent.parent.parent.parent
platform_path = _base / "ds_platform" / "platform_sdk"
project3_root = _base / "project_3_enterprise_rag_llm"
for p in (str(platform_path), str(project3_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from platform_sdk.serving.async_queue import create_async_task, celery_app
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


@create_async_task("rag.evaluate_run", "rag")
def evaluate_run_task(job_id: str, domain: str, payload: Dict[str, Any]):
    """
    Payload:
    {
      "eval_set_path": "...jsonl",
      "modes": [...],
      "gate_mode": "hybrid"
    }
    """
    from rag.retrieval.retriever import RetrieverStack
    from rag.generation.router import Router
    from rag.generation.pipeline import RAGPipeline
    from rag.generation.llm_client import OpenAICompatChatClient
    from rag.evaluation.io import load_eval_set_jsonl
    from rag.evaluation.metrics import run_eval, run_ragas_metrics
    from rag.evaluation.eval_gate import evaluate_gate, load_gate_config
    from pathlib import Path as _Path

    config_dir = _Path(__file__).parent.parent / "configs"
    retriever = RetrieverStack(
        config_path=str(config_dir / "retrieval.yaml"),
        index_store_dir=str(_Path(__file__).resolve().parents[2] / "data" / "index"),
        auto_load=True,
    )
    llm = OpenAICompatChatClient() if os.getenv("RAG_LLM_BASE_URL", "").strip() else None
    pipeline = RAGPipeline(retriever=retriever, router=Router(), llm_client=llm)

    eval_set = load_eval_set_jsonl(payload["eval_set_path"])
    modes = payload.get("modes") or ["hybrid"]
    gate_mode = payload.get("gate_mode") or "hybrid"
    report = run_eval(pipeline, eval_set, modes=modes)

    ab = report.get("ablation", {}).get(gate_mode, {})
    metrics = {
        "evidence_recall_at_k": float(ab.get("evidence_recall_at_k_mean", 0.0)),
        "citation_accuracy": float(ab.get("citation_accuracy_mean", 0.0)),
        "citation_coverage": float(ab.get("citation_coverage_mean", 0.0)),
        "latency_p95_ms": float(ab.get("latency_p95_ms", 0.0)),
    }
    ragas = run_ragas_metrics(pipeline, eval_set, mode=gate_mode)
    metrics.update(ragas)
    gate = evaluate_gate(metrics, load_gate_config())
    return {"gate_mode": gate_mode, "metrics": metrics, "gate": gate, "report": report}

