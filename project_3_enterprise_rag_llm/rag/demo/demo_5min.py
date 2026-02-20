"""
5-Min Demo (D13)
ingest 10 PDFs → build index → run eval gate (show metrics) → /ask returns citations
→ /feedback writes to DB → dashboard shows cost/latency/citation coverage
"""

import os
import sys
import time
import requests
from pathlib import Path

# project_3_enterprise_rag_llm as root (so that "rag" package is importable)
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main():
    print("=" * 60)
    print("Enterprise RAG — 5-Min Demo")
    print("=" * 60)

    print("\n[1/5] Ingest + build index...")
    ingest_and_build()

    print("\n[2/5] Run eval gate...")
    run_eval_gate()

    print("\n[3/5] /ask → citations...")
    ask_demo()

    print("\n[4/5] /feedback writing to DB...")
    feedback_demo()

    print("\n[5/5] Dashboard (cost/latency/citation)...")
    dashboard_note()

    print("\n" + "=" * 60)
    print("Demo done.")
    print("=" * 60)


def ingest_and_build():
    """Ingest 10 PDFs, build index."""
    try:
        from rag.ingestion.parsers import PdfParser
        from rag.ingestion.chunker import Chunker
        from rag.retrieval.retriever import RetrieverStack
    except Exception as e:
        print(f"  Skip (deps): {e}")
        return
    # PDF directory fixed at rag/data (project_3_enterprise_rag_llm/rag/data)
    pdf_dir = Path(__file__).parent.parent / "data"
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print("  No PDFs in rag/data; add files and re-run.")
        return
    pdfs = list(pdf_dir.glob("*.pdf"))[:10]
    if not pdfs:
        print("  No PDFs found.")
        return
    parser = PdfParser()
    chunker = Chunker(chunk_size=512, overlap=64)
    all_chunks = []
    for p in pdfs:
        blocks = parser.parse(str(p))
        chunks = chunker.chunk_blocks(blocks)
        all_chunks.extend(chunker.to_records(chunks))
    retriever = RetrieverStack()
    retriever.build_index(all_chunks)
    print(f"  Indexed {len(all_chunks)} chunks from {len(pdfs)} PDFs.")


def run_eval_gate():
    """Run eval, show gate metrics."""
    try:
        r = requests.post("http://localhost:8002/evaluate/run", json={"modes": ["bm25_only", "hybrid"]}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            print("  Gate / eval:", d.get("gate", {}), d.get("metrics", {}))
        else:
            print("  Eval run failed:", r.status_code, r.text[:200])
    except Exception as e:
        print("  Eval skip:", e)


def ask_demo():
    """POST /ask, print answer + citations."""
    try:
        r = requests.post(
            "http://localhost:8002/ask",
            json={"query": "What is the refund policy?", "top_k": 6, "mode": "hybrid"},
            timeout=10,
        )
        if r.status_code == 200:
            d = r.json()
            print("  answer:", (d.get("answer") or "")[:200])
            print("  citations:", d.get("citations", [])[:3])
            print("  index_version:", d.get("index_version"))
        else:
            print("  /ask failed:", r.status_code, r.text[:200])
    except Exception as e:
        print("  /ask skip:", e)


def feedback_demo():
    """POST /feedback → rag.feedback."""
    try:
        r = requests.post(
            "http://localhost:8002/feedback",
            json={"request_id": "demo-req-1", "rating": 4, "reason": "demo", "notes": "5min"},
            timeout=5,
        )
        if r.status_code == 200:
            print("  feedback written:", r.json())
        else:
            print("  feedback failed:", r.status_code, r.text[:200])
    except Exception as e:
        print("  feedback skip:", e)


def dashboard_note():
    print("  → /metrics (Prometheus): cost, latency, citation coverage.")
    print("  → Grafana dashboard: point to RAG metrics.")


if __name__ == "__main__":
    main()
