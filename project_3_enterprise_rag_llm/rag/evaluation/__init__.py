"""
Evaluation Harness (D7)
ragas + evaluate, citation accuracy (D7.1), ablation.
"""

from rag.evaluation.metrics import run_eval, compute_citation_accuracy

__all__ = ["run_eval", "compute_citation_accuracy"]
