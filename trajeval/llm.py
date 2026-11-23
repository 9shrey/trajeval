"""Minimal async client for OpenAI-compatible chat endpoints (nanoserve, vLLM, hosted APIs)."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    message: dict           # {"role": "assistant", "content": ..., "tool_calls": [...]}
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 600.0,
                 max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        key = api_key or os.environ.get("TRAJEVAL_API_KEY") or "none"
        self._client = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {key}"},
                                         limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
        self.max_retries = max_retries

    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None, temperature: float = 0.0,
                   max_tokens: int = 512, seed: int | None = None, top_p: float = 1.0,
                   top_k: int = 0) -> LLMResponse:
        # Sampling params are always sent explicitly, so servers cannot silently apply different
        # defaults (vLLM applies the model's generation_config unless told otherwise).
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "top_p": top_p, "top_k": top_k, "repetition_penalty": 1.0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if seed is not None:
            body["seed"] = seed
        last: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions", json=body)
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:300]}", request=r.request, response=r)
                if r.status_code >= 400:
                    raise LLMRequestError(f"{r.status_code}: {r.text[:500]}")
                data = r.json()
                latency = (time.perf_counter() - t0) * 1000
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return LLMResponse(choice["message"], choice.get("finish_reason"),
                                   usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), latency)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                await asyncio.sleep(2 ** attempt)
        raise LLMRequestError(f"request failed after {self.max_retries} attempts: {last}")

    async def aclose(self):
        await self._client.aclose()


class LLMRequestError(Exception):
    """A request the server rejected (e.g. context length exceeded) or that kept failing."""
