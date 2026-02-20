"""
Generation & Guardrails (D6)
Prompt enforces citation format; returns "I don't know" when no evidence.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Citation:
    chunk_id: str
    source_id: str
    page: Optional[int]
    excerpt: str


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    retrieved_chunk_ids: List[str]
    retrieved_contexts: List[str]  # full chunk content for ragas
    index_version: str
    prompt_version: str
    latency_ms: int
    no_evidence: bool  # True iff no evidence → "I don't know"
    prompt_tokens: int = 0
    completion_tokens: int = 0


CITATION_FMT = "[{source_id}:p{page}]"  # Enforced citation format
IDK_ANSWER = "I don't know."  # Returned when no evidence


def format_citation(c: Citation) -> str:
    return f"[{c.source_id}:p{c.page or 0}]"


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Build context string from retrieved chunks for prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        sid = c.get("source_id", "?")
        page = c.get("page") or 0
        parts.append(f"[{i}] [{sid}:p{page}]\n{c.get('content', '')}")
    return "\n\n".join(parts)


def build_prompt(query: str, context: str, prompt_version: str = "prompt_v1") -> str:
    """Prompt enforces citation format."""
    return f"""Use the following context to answer the question. Cite each claim with [source_id:pPage]. If the context does not contain relevant information, respond with exactly: {IDK_ANSWER}

Context:
{context}

Question: {query}

Answer:"""


def parse_citations_from_answer(answer: str) -> List[str]:
    """Extract [source_id:ppage] refs from model output."""
    import re
    return re.findall(r"\[[\w\-]+:p\d+\]", answer)


class RAGPipeline:
    """
    Retrieve → Generate → Citations.
    Guardrail: no evidence → "I don't know".
    """

    def __init__(
        self,
        retriever,
        router,
        llm_client=None,
        prompt_version: str = "prompt_v1"
    ):
        self.retriever = retriever
        self.router = router
        self.llm_client = llm_client
        self.prompt_version = prompt_version

    def ask(
        self,
        query: str,
        top_k: int = 6,
        mode: str = "hybrid"
    ) -> RAGResponse:
        """
        Run RAG: retrieve → generate → citations.
        D6 Router: FAQ/short queries → cheap path (smaller top_k + bm25_only), complex queries → full RAG (default hybrid).
        """
        start = datetime.utcnow()
        routed = getattr(self.router, "route", None)
        if callable(routed):
            path = self.router.route(query)
            if path == "cheap":
                top_k = min(top_k, 3)
                # If caller doesn't explicitly specify other mode, cheap path defaults to bm25_only
                if mode == "hybrid":
                    mode = "bm25_only"

        chunks = self.retriever.retrieve(query, top_k=top_k, mode=mode)
        retrieved_ids = [c.chunk_id for c in chunks]

        if not chunks:
            latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            return RAGResponse(
                answer=IDK_ANSWER,
                citations=[],
                retrieved_chunk_ids=[],
                retrieved_contexts=[],
                index_version=getattr(self.retriever, "index_version_str", "unknown"),
                prompt_version=self.prompt_version,
                latency_ms=latency_ms,
                no_evidence=True,
                prompt_tokens=0,
                completion_tokens=0,
            )

        context = build_context(
            [{"source_id": c.source_id, "page": c.page, "content": c.content} for c in chunks]
        )
        prompt = build_prompt(query, context, self.prompt_version)

        # If LLM is available, use it to generate; otherwise or on failure, build answer from retrieved chunks
        answer = IDK_ANSWER
        prompt_tokens, completion_tokens = 0, 0
        if self.llm_client is not None:
            try:
                out = self.llm_client.generate(prompt)
                if isinstance(out, tuple):
                    answer, usage = out[0] or IDK_ANSWER, out[1]
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                else:
                    answer = out or IDK_ANSWER
            except Exception:
                answer = IDK_ANSWER
        # Retrieved chunks but answer is still IDK (no LLM or LLM failed): build answer from retrieved content
        if answer.strip() == IDK_ANSWER and chunks:
            parts = []
            for i, c in enumerate(chunks[:5], 1):
                sid = c.source_id or "?"
                page = c.page if c.page is not None else 0
                excerpt = (c.content[:500] + "..." if len(c.content or "") > 500 else (c.content or ""))
                parts.append(f"[{sid}:p{page}] {excerpt}")
            answer = "Based on the retrieved context:\n\n" + "\n\n---\n\n".join(parts)

        refs = parse_citations_from_answer(answer)
        citations = []
        for c in chunks:
            cit = Citation(chunk_id=c.chunk_id, source_id=c.source_id, page=c.page, excerpt=(c.content[:200] if c.content else ""))
            tag = format_citation(cit)
            if refs and any(tag.replace(" ", "") == r.replace(" ", "") for r in refs):
                citations.append(cit)
            elif not refs:
                citations.append(cit)

        latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
        retrieved_contexts = [c.content or "" for c in chunks]
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunk_ids=retrieved_ids,
            retrieved_contexts=retrieved_contexts,
            index_version=getattr(self.retriever, "index_version_str", "unknown"),
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            no_evidence=answer.strip() == IDK_ANSWER,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
