"""
Evaluation Harness (D7)
Eval set: query / gold answer / gold evidence (chunk_id + page).
Metrics: faithfulness, answer_relevancy, context recall/precision, latency, cost.
D7.1: Evidence Recall@k, Citation Accuracy.
Ablation: BM25-only / dense-only / hybrid+rerank.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EvalSample:
    query: str
    gold_answer: str
    gold_evidence: List[Dict[str, Any]]  # [{"chunk_id": ..., "page": ...}]


def compute_citation_accuracy(
    retrieved_chunk_ids: List[str],
    gold_evidence: List[Dict[str, Any]],
    answer_citations: List[str],
    k: int = 6
) -> Dict[str, float]:
    """
    D7.1 Citation Accuracy.
    - Evidence Recall@k: gold evidence chunks in retrieved top-k.
    - Citation Accuracy: answer citations vs gold evidence (chunk_id/page overlap).
    """
    # Prefer chunk_id; if gold only has page, fallback to page string (keep metric reproducible)
    gold_ids = {str(e.get("chunk_id") or e.get("page") or "") for e in gold_evidence}
    gold_ids = {x for x in gold_ids if x}
    top_k = set(retrieved_chunk_ids[:k])

    recall_at_k = 0.0
    if gold_ids:
        overlap = len(gold_ids & top_k) / len(gold_ids)
        recall_at_k = min(1.0, overlap)

    # Citation accuracy: cited chunks that are in gold
    cited = set(answer_citations)
    accuracy = 0.0
    if cited:
        hit = len(cited & gold_ids) / len(cited)
        accuracy = min(1.0, hit)

    return {
        "evidence_recall_at_k": recall_at_k,
        "citation_accuracy": accuracy,
    }


def compute_citation_coverage(answer: str, citations: List[str], idk_answer: str = "I don't know.") -> float:
    """
    D8: citation_coverage ensures "has citation".
    Metric (reproducible, lightweight): For non-IDK answers, if at least 1 citation (chunk_id) is included, coverage=1, otherwise 0.
    """
    if not answer or answer.strip() == idk_answer:
        return 1.0  # Direct IDK when no evidence is considered compliant
    return 1.0 if citations else 0.0


def compute_guardrail_rates(answer: str, had_retrieval: bool, idk_answer: str = "I don't know.") -> Dict[str, float]:
    """
    D8 Guardrails:
    - must_return_idk_when_no_evidence: Must return IDK when no evidence is retrieved
    - hallucination_flag_rate: Simplified reproducible rule placeholder (for gate wiring)
    """
    idk_violation = 1.0 if (not had_retrieval and answer.strip() != idk_answer) else 0.0
    # Lightweight hallucination rule (reproducible, placeholder): Non-IDK and no citations in format [x:pN]
    import re
    has_ref = bool(re.findall(r"\[[\w\-]+:p\d+\]", answer or ""))
    hallucination_flag = 1.0 if (had_retrieval and answer.strip() != idk_answer and not has_ref) else 0.0
    return {
        "idk_violation_rate": idk_violation,
        "hallucination_flag_rate": hallucination_flag,
    }


def run_eval(
    pipeline,
    eval_set: List[EvalSample],
    modes: List[str] = ["bm25_only", "dense_only", "hybrid"]
) -> Dict[str, Any]:
    """
    Run evaluation over eval_set.
    Returns metrics + ablation table (D7).
    """
    results: List[Dict[str, Any]] = []
    for sample in eval_set:
        row = {"query": sample.query}
        for mode in modes:
            resp = pipeline.ask(sample.query, mode=mode)
            acc = compute_citation_accuracy(
                resp.retrieved_chunk_ids,
                sample.gold_evidence,
                [c.chunk_id for c in resp.citations],
                k=6
            )
            row[f"{mode}_latency_ms"] = resp.latency_ms
            row[f"{mode}_evidence_recall_at_k"] = acc["evidence_recall_at_k"]
            row[f"{mode}_citation_accuracy"] = acc["citation_accuracy"]
            row[f"{mode}_no_evidence"] = resp.no_evidence
            row[f"{mode}_citation_coverage"] = compute_citation_coverage(
                resp.answer,
                [c.chunk_id for c in resp.citations],
                idk_answer="I don't know."
            )
        results.append(row)

    # Aggregate
    out = {"samples": results, "ablation": {}}
    for mode in modes:
        latencies = [r[f"{mode}_latency_ms"] for r in results]
        rec = [r[f"{mode}_evidence_recall_at_k"] for r in results]
        acc = [r[f"{mode}_citation_accuracy"] for r in results]
        cov = [r[f"{mode}_citation_coverage"] for r in results]
        out["ablation"][mode] = {
            "latency_p95_ms": float(__import__("numpy").np.percentile(latencies, 95)) if latencies else 0,
            "evidence_recall_at_k_mean": sum(rec) / len(rec) if rec else 0,
            "citation_accuracy_mean": sum(acc) / len(acc) if acc else 0,
            "citation_coverage_mean": sum(cov) / len(cov) if cov else 0,
        }
    return out


def run_ragas_metrics(
    pipeline,
    eval_set: List[EvalSample],
    mode: str = "hybrid",
) -> Dict[str, float]:
    """
    D7: faithfulness, answer_relevancy, context recall/precision via ragas.
    Builds HF Dataset from pipeline outputs + eval_set, then ragas.evaluate().
    """
    out = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
    }
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except ImportError:
        return out

    questions, answers, contexts_list, ground_truths = [], [], [], []
    for s in eval_set:
        resp = pipeline.ask(s.query, mode=mode)
        # contexts: list[list[str]] per ragas (full chunk content)
        ctxs = getattr(resp, "retrieved_contexts", None) or []
        contexts_list.append(ctxs if ctxs else [""])
        answers.append(resp.answer or "")
        ground_truths.append(s.gold_answer or "")
        questions.append(s.query)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    try:
        df = result.to_pandas() if hasattr(result, "to_pandas") else None
        if df is not None:
            for k in out:
                if k in df.columns:
                    out[k] = float(df[k].mean())
    except Exception:
        pass
    return out
