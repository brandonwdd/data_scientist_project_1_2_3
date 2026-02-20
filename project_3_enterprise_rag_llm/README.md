# Project 3: Enterprise RAG System

Enterprise Knowledge Base RAG + Evaluation Gate + Cost Routing

## D1. Executive Summary

- **Pipeline**: ingestion → chunk/index → retrieve/rerank → generate → citations
- **Evaluation system** (ragas/evaluate) as core: deployment gate
- **Groundedness / Citation accuracy**, cost/latency optimization + routing strategy
- **Feedback loop**; optional LoRA (router/reranker)

## D2. Use Cases & KPI

- **Use cases**: Internal support, sales enablement, policy/process Q&A
- **Metrics**: faithfulness, answer relevancy, context precision/recall, citation coverage & accuracy, latency & token cost

## Project Structure

```
project_3_enterprise_rag_llm/
├── rag/
│   ├── configs/
│   │   ├── promotion_gate.yaml   # Eval gate (D8), Citation Accuracy (D8.1)
│   │   └── retrieval.yaml       # BM25 / Dense / Rerank
│   ├── ingestion/               # D4
│   │   ├── parsers.py           # pymupdf / unstructured
│   │   └── chunker.py           # By heading hierarchy, overlap, metadata
│   ├── retrieval/               # D5
│   │   └── retriever.py         # BM25, Dense, Hybrid, Rerank; versioning
│   ├── generation/              # D6
│   │   ├── pipeline.py          # Prompt, "I don't know" when no evidence, citations
│   │   └── router.py            # FAQ/short → cheap; complex → full RAG
│   ├── evaluation/              # D7, D8
│   │   ├── metrics.py           # ragas, citation accuracy (D7.1), ablation
│   │   └── eval_gate.py         # Eval gate (D8), D8.1
│   ├── feedback/                # D10
│   │   └── hard_set.py          # Low scores → hard_set.jsonl
│   ├── serving/                 # D9
│   │   └── app.py               # /ask, /retrieve, /evaluate/run, /feedback
│   ├── artifacts.py             # D12 MLflow artifacts
│   └── demo/
│       └── demo_5min.py         # D13
└── README.md
```

## D4. Ingestion & Chunking

- **Parsing**: pymupdf / unstructured
- **Chunking**: By heading hierarchy, overlap, metadata (source_id, page, section, created_at, hash)
- **Output**: chunks parquet (S3)

## D5. Retrieval & Versioning

- **BM25**: rank-bm25
- **Dense**: sentence-transformers + FAISS / Qdrant
- **Rerank**: cross-encoder (optional)
- **Versioning**: index_version, prompt_version, retriever_version

## D6. Generation & Guardrails

- Prompt enforces citation format; returns `"I don't know"` when no evidence
- **Router**: FAQ/short → cheap; complex → full RAG (+ rerank)

## D7. Evaluation Harness

- **Eval set**: 200–500 items; query / gold answer / gold evidence (chunk_id + page)
- **Metrics**: faithfulness, answer relevancy, context recall/precision, latency, cost
- **D7.1**: Evidence Recall@k, Citation Accuracy
- **Ablation**: BM25-only / dense-only / hybrid+rerank

## D8. Eval Gate

- `configs/promotion_gate.yaml`: faithfulness ≥ 0.80, answer_relevancy ≥ 0.85, context_recall ≥ 0.75, context_precision ≥ 0.55, citation_coverage ≥ 0.90, latency p95 ≤ 1.2s, must_return_idk_when_no_evidence, hallucination_flag_rate ≤ 0.05
- **D8.1**: evidence_recall_at_k_min ≥ 0.75, citation_accuracy_min ≥ 0.70

## D9. Serving & APIs

- **POST /ask**: query → answer + citations (audit index_version, prompt_version, retrieved_chunk_ids)
- **POST /retrieve**: top-k chunks (debugging)
- **POST /evaluate/run**: trigger evaluation (async)
- **POST /feedback**: write to `rag.feedback`

## D10. Feedback Loop

- Weekly job: sample low scores/low faithfulness → `hard_set.jsonl`
- Hard set drives: chunking adjustments, reranker data augmentation, router fine-tuning

## D12. MLflow Artifacts

- eval_metrics.json, ragas_report.json, latency_report.json, cost_report.json
- citation_coverage_report.json, ablation_table.csv, retrieval_curves.png
- prompt_version.txt, index_manifest.json, model_card.md, known_failure_cases.md

## D13. 5-Min Demo

```bash
cd project_3_enterprise_rag_llm
# 1. Start RAG service
uvicorn rag.serving.app:app --host 0.0.0.0 --port 8002

# 2. Run demo (another terminal)
python rag/demo/demo_5min.py
```

- Ingest 10 PDFs → build index → run eval gate → /ask → /feedback → dashboard (cost/latency/citation)

## Quick Start

```bash
cd project_3_enterprise_rag_llm
pip install -r requirements.txt
pip install -e ../ds_platform/platform_sdk
uvicorn rag.serving.app:app --host 0.0.0.0 --port 8002
# Demo: python rag/demo/demo_5min.py
```

---

## Production Deployment (Recommended)

### 1. Start Infrastructure + Qdrant

```bash
cd ds_platform/infra
docker compose up -d
```

Includes Postgres, Redis, MLflow, Prometheus, **Qdrant** (vector store). Project defaults to `index_type: qdrant` in `retrieval.yaml`.

### 2. Environment Variables

Copy `project_3_enterprise_rag_llm/env.example` to `.env` and modify as needed:

- `RAG_LLM_BASE_URL`: OpenAI-compatible API (OpenAI / vLLM, etc.)
- `RAG_LLM_API_KEY`, `RAG_LLM_MODEL`
- `QDRANT_HOST`, `QDRANT_PORT` (default localhost:6333)
- `POSTGRES_*`, `REDIS_*`, `CELERY_*` consistent with compose

### 3. Install Dependencies

```bash
cd project_3_enterprise_rag_llm
pip install -r requirements.txt
pip install -e ../ds_platform/platform_sdk
```

### 4. Ingestion → Build Index

```bash
# PDF → chunks.jsonl, optionally build index
python -m rag.cli.ingest --pdf-dir data/pdfs --out data/chunks.jsonl --build-index --rag-url http://localhost:8002
```

Or generate `chunks.jsonl` first, then `POST /index/build` with body: `{"chunks_path": "/abs/path/to/chunks.jsonl"}`. Index persists to `data/index/` + Qdrant, automatically loads `LATEST` on service startup.

### 5. Start RAG Service

```bash
cd project_3_enterprise_rag_llm
uvicorn rag.serving.app:app --host 0.0.0.0 --port 8002
```

### 6. Celery Worker (Async Evaluation)

```bash
cd project_3_enterprise_rag_llm
celery -A rag.serving.async_tasks:celery_app worker -l info -Q celery
```

Ensure `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` point to Redis. After `POST /evaluate/run` enqueues, worker executes `rag.evaluate_run`, writes results to `platform.async_jobs`; `GET /jobs/{job_id}` queries status.

### 7. Evaluation & Gate

- `POST /evaluate/run`: body `{"eval_set_path": "/path/to/eval.jsonl", "modes": ["bm25_only","dense_only","hybrid"], "gate_mode": "hybrid"}` → returns `job_id`. `eval_set_path` should use **absolute path** (worker runtime cwd may differ).
- Eval uses **ragas** to calculate faithfulness / answer_relevancy / context recall & precision, plus D7.1 citation metrics, then runs D8 Eval Gate

### 8. Monitoring

- **Prometheus**: `/metrics` exposes `rag_ask_latency_seconds`, `rag_tokens_total`, `rag_ask_total`, etc.; can configure scrape `rag-service:8002`
- **Grafana**: Connect to Prometheus, display latency, QPS, token consumption, citation-related panels

### 9. Summary

| Component | Description |
|------|------|
| Vector Store | Qdrant (included in compose), `index_type: qdrant` |
| Index | Local `data/index/` + Qdrant persistence, auto-loads on startup |
| LLM | OpenAI-compatible API, enabled when `RAG_LLM_BASE_URL` is configured |
| Evaluation | ragas + D7.1 + D8 gate, async worker execution |
| Observability | Prometheus + platform middleware, RAG-specific metrics |

---

## Data & Isolation (D3)

- **S3**: `s3://lake/rag/*` (chunks, metadata, evalset)
- **Postgres**: `rag.*` + `platform.*`
- **Vector Store**: Qdrant or FAISS
