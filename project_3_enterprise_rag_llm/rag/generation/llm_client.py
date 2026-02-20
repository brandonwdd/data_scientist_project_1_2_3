"""
Production LLM client (OpenAI-compatible HTTP)

Works with:
- OpenAI API (if you point base_url to https://api.openai.com/v1)
- vLLM OpenAI-compatible server
- Any OpenAI-compatible gateway
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import os
import requests


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAICompatChatClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 120,
    ):
        self.base_url = (base_url or os.getenv("RAG_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
        self.timeout_s = int(os.getenv("RAG_LLM_TIMEOUT", str(timeout_s)))  # local Ollama 8B can be slow; default 120s

    def generate(self, prompt: str) -> Tuple[str, LLMUsage]:
        """
        Call OpenAI-compatible API. Returns (answer_text, usage).
        If base_url is not set, raise to force explicit configuration in production.
        """
        if not self.base_url:
            raise RuntimeError("RAG_LLM_BASE_URL is required for production LLM")

        url = f"{self.base_url}/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Follow citation rules."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.getenv("RAG_LLM_TEMPERATURE", "0.2")),
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        u = LLMUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        )
        text = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return (text, u)

