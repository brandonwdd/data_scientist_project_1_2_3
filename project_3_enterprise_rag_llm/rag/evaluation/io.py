"""Eval set IO (D7). JSONL format: query, gold_answer, gold_evidence[{chunk_id, page}]."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rag.evaluation.metrics import EvalSample


def load_eval_set_jsonl(path: str) -> List[EvalSample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"eval_set_path not found: {path}")

    out: List[EvalSample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                EvalSample(
                    query=obj["query"],
                    gold_answer=obj.get("gold_answer", ""),
                    gold_evidence=obj.get("gold_evidence", []) or [],
                )
            )
    return out

