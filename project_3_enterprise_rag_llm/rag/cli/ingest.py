"""
Production ingest: PDFs → chunks jsonl → optional /index/build.
PDF directory is fixed at project_3_enterprise_rag_llm/rag/data; place your PDFs there.

Usage:
  python -m rag.cli.ingest --out data/chunks.jsonl
  python -m rag.cli.ingest --out data/chunks.jsonl --build-index --rag-url http://localhost:8002
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# project_3_enterprise_rag_llm root
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from rag.ingestion.parsers import PdfParser
from rag.ingestion.chunker import Chunker


def main():
    # Default PDF directory: project_3_enterprise_rag_llm/rag/data
    default_pdf_dir = _root / "rag" / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(default_pdf_dir), help="Directory containing PDFs (default: rag/data)")
    ap.add_argument("--out", required=True, help="Output chunks.jsonl path")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--build-index", action="store_true", help="POST /index/build after writing chunks")
    ap.add_argument("--rag-url", default="http://localhost:8002", help="RAG service URL for /index/build")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.exists():
        print(f"Missing: {pdf_dir}")
        return 1

    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {pdf_dir}")
        return 1

    parser = PdfParser()
    chunker = Chunker(chunk_size=args.chunk_size, overlap=args.overlap)
    all_records = []
    for p in pdfs:
        blocks = parser.parse(str(p))
        chunks = chunker.chunk_blocks(blocks)
        all_records.extend(chunker.to_records(chunks))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_records)} chunks to {out_path}")

    if args.build_index:
        import requests
        # /index/build expects chunks_path; use absolute path
        abs_out = str(out_path.resolve())
        r = requests.post(
            f"{args.rag_url.rstrip('/')}/index/build",
            json={"chunks_path": abs_out},
            timeout=60,
        )
        if r.status_code != 200:
            print(f"POST /index/build failed: {r.status_code} {r.text}")
            return 1
        print("Index build OK:", r.json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
