"""
Generation & Guardrails (D6)
Prompt, "I don't know" when no evidence, Router (cheap / full RAG).
"""

from rag.generation.pipeline import RAGPipeline
from rag.generation.router import Router

__all__ = ["RAGPipeline", "Router"]
