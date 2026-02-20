"""
RAG-specific Prometheus metrics (D2/D7/D8).
latency, token cost, citation coverage.
"""

from __future__ import annotations

_metrics = None


def _get():
    global _metrics
    if _metrics is not None:
        return _metrics
    try:
        from prometheus_client import Histogram, Counter
        _metrics = {
            "rag_ask_latency_seconds": Histogram(
                "rag_ask_latency_seconds",
                "RAG /ask latency",
                ["domain"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            ),
            "rag_tokens_total": Counter(
                "rag_tokens_total",
                "RAG LLM tokens",
                ["domain", "type"],  # type: prompt | completion
            ),
            "rag_ask_total": Counter(
                "rag_ask_total",
                "RAG /ask requests",
                ["domain", "no_evidence"],
            ),
        }
        return _metrics
    except ImportError:
        return {}


def observe_ask(domain: str, latency_seconds: float, prompt_tokens: int, completion_tokens: int, no_evidence: bool):
    m = _get()
    if not m:
        return
    m["rag_ask_latency_seconds"].labels(domain=domain).observe(latency_seconds)
    m["rag_tokens_total"].labels(domain=domain, type="prompt").inc(prompt_tokens)
    m["rag_tokens_total"].labels(domain=domain, type="completion").inc(completion_tokens)
    m["rag_ask_total"].labels(domain=domain, no_evidence=str(no_evidence).lower()).inc()
