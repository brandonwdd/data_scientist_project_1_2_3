"""Router: FAQ/short → cheap path; complex → full RAG + rerank."""

from typing import Literal


def is_short_or_faq(query: str, max_len: int = 50) -> bool:
    """Heuristic: short or FAQ-like → cheap path."""
    q = query.strip()
    if len(q) <= max_len:
        return True
    # FAQ-style: "What is X?", "How to Y?"
    faq_start = ("what is", "how to", "how do", "can i", "where is", "when does")
    lower = q.lower()
    return any(lower.startswith(s) for s in faq_start)


def route(query: str) -> Literal["cheap", "full"]:
    """
    Route query to cheap (FAQ/short) or full RAG path.
    Returns "cheap" or "full".
    """
    return "cheap" if is_short_or_faq(query) else "full"


class Router:
    """Router: cheap vs full RAG."""

    def __init__(self, max_short_len: int = 50):
        self.max_short_len = max_short_len

    def route(self, query: str) -> str:
        return "cheap" if is_short_or_faq(query, self.max_short_len) else "full"
