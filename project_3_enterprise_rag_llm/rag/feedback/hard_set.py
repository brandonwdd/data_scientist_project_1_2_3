"""
Feedback Loop (D10)
Weekly job: sample low score/low faithfulness requests → hard_set.jsonl.
Hard set drives: chunking adjustments / reranker data augmentation / router fine-tuning.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def build_hard_set(
    feedback_records: List[Dict[str, Any]],
    output_path: str,
    min_count: int = 20,
    max_count: int = 500
) -> str:
    """
    Sample low-rating / low-faithfulness requests into hard_set.jsonl.
    Returns path to written file.
    """
    # Filter: low rating (<=2) or low faithfulness
    low = [
        r for r in feedback_records
        if r.get("rating", 5) <= 2 or r.get("faithfulness", 1.0) < 0.7
    ]
    low = low[:max_count]
    if len(low) < min_count:
        return ""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in low:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(out)
