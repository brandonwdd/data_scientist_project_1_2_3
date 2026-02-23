"""MLflow artifacts: eval, ragas, latency, cost, citation, ablation, model_card."""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List


def write_artifacts(
    output_dir: str,
    eval_metrics: Dict[str, float],
    ragas_report: Dict[str, Any],
    latency_report: Dict[str, Any],
    cost_report: Dict[str, Any],
    citation_report: Dict[str, Any],
    ablation_table: List[Dict[str, Any]],
    prompt_version: str,
    index_manifest: Dict[str, Any],
    known_failures: List[Dict[str, Any]],
) -> None:
    """Write all D12 artifacts to output_dir."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    (p / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    (p / "ragas_report.json").write_text(json.dumps(ragas_report, indent=2), encoding="utf-8")
    (p / "latency_report.json").write_text(json.dumps(latency_report, indent=2), encoding="utf-8")
    (p / "cost_report.json").write_text(json.dumps(cost_report, indent=2), encoding="utf-8")
    (p / "citation_coverage_report.json").write_text(json.dumps(citation_report, indent=2), encoding="utf-8")
    (p / "prompt_version.txt").write_text(prompt_version, encoding="utf-8")
    (p / "index_manifest.json").write_text(json.dumps(index_manifest, indent=2), encoding="utf-8")
    (p / "known_failure_cases.md").write_text(_known_failures_md(known_failures), encoding="utf-8")

    if ablation_table:
        with open(p / "ablation_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ablation_table[0].keys())
            w.writeheader()
            w.writerows(ablation_table)

    model_card = _model_card(eval_metrics, prompt_version, index_manifest)
    (p / "model_card.md").write_text(model_card, encoding="utf-8")


def _model_card(metrics: Dict, prompt_version: str, index_manifest: Dict) -> str:
    return f"""# Model Card: RAG System

## Metadata
- prompt_version: {prompt_version}
- index_manifest: {json.dumps(index_manifest, indent=2)}

## Eval Metrics
- faithfulness: {metrics.get('faithfulness', 'N/A')}
- answer_relevancy: {metrics.get('answer_relevancy', 'N/A')}
- context_recall: {metrics.get('context_recall', 'N/A')}
- context_precision: {metrics.get('context_precision', 'N/A')}
- citation_coverage: {metrics.get('citation_coverage', 'N/A')}
- evidence_recall_at_k: {metrics.get('evidence_recall_at_k', 'N/A')}
- citation_accuracy: {metrics.get('citation_accuracy', 'N/A')}

## Intended Use
Enterprise RAG: ingestion → chunk/index → retrieve/rerank → generate → citations.
"""


def _known_failures_md(failures: List[Dict]) -> str:
    lines = ["# Known Failure Cases\n"]
    for i, f in enumerate(failures[:20], 1):
        lines.append(f"## {i}. {f.get('query', '')[:80]}")
        lines.append(f"- **Reason**: {f.get('reason', '')}")
        lines.append(f"- **Fix**: {f.get('fix', '')}\n")
    return "\n".join(lines)
