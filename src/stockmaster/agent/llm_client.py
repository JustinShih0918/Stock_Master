"""Chat-model access for the agent: OpenAI-compatible providers (NVIDIA Integrate, Gemini),
retries, a record/replay cassette for reproducible runs, and a scripted model for tests.

API keys come from ``Settings`` only and are never placed in requests we log.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import openai
from tenacity import (Retrying, retry_if_exception_type, stop_after_attempt,
                      wait_exponential_jitter)

from stockmaster.settings import Settings


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as produced by the model


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall]
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)


class ChatModel(Protocol):
    provider: str
    model_id: str
    supports_tools: bool

    def complete(self, messages: list[dict], tools: list[dict] | None, *,
                 temperature: float, max_tokens: int) -> ChatResult: ...


class ToolsUnsupported(RuntimeError):
    pass


class ProviderUnavailable(RuntimeError):
    pass


PROVIDERS = {
    "nvidia": ("nvidia_base_url", "nvidia_api_key"),
    "gemini": ("gemini_base_url", "gemini_api_key"),
}
_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
              openai.InternalServerError)


def _parse_result(resp, latency_ms: int) -> ChatResult:
    choice = resp.choices[0]
    msg = choice.message
    calls = [ToolCall(tc.id or f"call_{i}", tc.function.name, tc.function.arguments or "{}")
             for i, tc in enumerate(msg.tool_calls or [])]
    extra = getattr(msg, "model_extra", None) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    usage = resp.usage
    return ChatResult(content=msg.content, tool_calls=calls, reasoning=reasoning,
                      finish_reason=choice.finish_reason,
                      prompt_tokens=getattr(usage, "prompt_tokens", None),
                      completion_tokens=getattr(usage, "completion_tokens", None),
                      latency_ms=latency_ms, raw=resp.model_dump(mode="json"))


class OpenAICompatModel:
    def __init__(self, provider: str, model_id: str, settings: Settings | None = None,
                 timeout: float = 300.0, supports_tools: bool = True):
        settings = settings or Settings()  # fresh: picks up keys added to .env after startup
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        url_attr, key_attr = PROVIDERS[provider]
        key = getattr(settings, key_attr)
        if key is None or not key.get_secret_value():
            raise ProviderUnavailable(f"{key_attr.upper()} is not set")
        self.provider, self.model_id, self.supports_tools = provider, model_id, supports_tools
        self._client = openai.OpenAI(base_url=getattr(settings, url_attr),
                                     api_key=key.get_secret_value(), timeout=timeout, max_retries=0)

    def complete(self, messages: list[dict], tools: list[dict] | None, *, temperature: float,
                 max_tokens: int) -> ChatResult:
        kwargs = dict(model=self.model_id, messages=messages, temperature=temperature,
                      max_tokens=max_tokens)
        if tools and self.supports_tools:
            kwargs.update(tools=tools, tool_choice="auto")
        t0 = time.monotonic()
        try:
            # NVIDIA's free endpoints return bursts of transient 500s; ride them out (~4-5 min).
            for attempt in Retrying(retry=retry_if_exception_type(_RETRYABLE),
                                    wait=wait_exponential_jitter(initial=5, max=90),
                                    stop=stop_after_attempt(7), reraise=True):
                with attempt:
                    resp = self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if tools and "tool" in str(e).lower():
                raise ToolsUnsupported(str(e)) from e
            raise
        return _parse_result(resp, int((time.monotonic() - t0) * 1000))


class Cassette:
    """Record/replay of model responses keyed by the request, for golden-day replays."""

    def __init__(self, path: Path, mode: str = "replay"):
        assert mode in {"record", "replay"}
        self.path, self.mode = path, mode
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                self.entries[e["key"]] = e["result"]

    @staticmethod
    def key(model_id: str, messages: list[dict], tools: list[dict] | None) -> str:
        blob = json.dumps({"m": model_id, "msgs": messages, "tools": tools}, sort_keys=True,
                          ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> ChatResult | None:
        e = self.entries.get(key)
        if e is None:
            return None
        e = dict(e)
        e["tool_calls"] = [ToolCall(**tc) for tc in e["tool_calls"]]
        return ChatResult(**e)

    def put(self, key: str, result: ChatResult) -> None:
        d = result.__dict__.copy()
        d["tool_calls"] = [tc.__dict__ for tc in result.tool_calls]
        d["raw"] = {}
        self.entries[key] = d
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "result": d}, ensure_ascii=False) + "\n")


class CassetteModel:
    """Wraps a model: replay from the cassette when possible, otherwise call and (record) it."""

    def __init__(self, inner: ChatModel | None, cassette: Cassette, provider: str, model_id: str):
        self.inner, self.cassette = inner, cassette
        self.provider, self.model_id = provider, model_id
        self.supports_tools = True if inner is None else inner.supports_tools

    def complete(self, messages, tools, *, temperature, max_tokens) -> ChatResult:
        k = Cassette.key(self.model_id, messages, tools)
        hit = self.cassette.get(k)
        if hit is not None:
            return hit
        if self.cassette.mode == "replay" or self.inner is None:
            raise ProviderUnavailable(f"cassette miss for {self.model_id}")
        res = self.inner.complete(messages, tools, temperature=temperature, max_tokens=max_tokens)
        self.cassette.put(k, res)
        return res


class ScriptedModel:
    """Deterministic fake model for tests: returns the scripted results in order."""

    def __init__(self, script: Sequence[ChatResult], provider: str = "scripted",
                 model_id: str = "scripted-v1", supports_tools: bool = True):
        self.script = list(script)
        self.provider, self.model_id, self.supports_tools = provider, model_id, supports_tools
        self.requests: list[list[dict]] = []

    def complete(self, messages, tools, *, temperature, max_tokens) -> ChatResult:
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        if not self.script:
            raise ProviderUnavailable("script exhausted")
        return self.script.pop(0)


def parse_chain(spec: str | None = None, settings: Settings | None = None,
                fallbacks: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """'nvidia:model-a,gemini:model-b' -> [(provider, model_id), ...].

    When LLM_CHAIN is empty: NVIDIA_MODEL, then the configured fallback models
    (strategy.yaml llm.fallback_models), then GEMINI_MODEL. Duplicates are removed."""
    settings = settings or Settings()
    spec = spec if spec is not None else settings.llm_chain
    out: list[tuple[str, str]] = []
    for part in [p.strip() for p in (spec or "").split(",") if p.strip()]:
        provider, _, model = part.partition(":")
        out.append((provider.strip(), model.strip()))
    if not out:
        if settings.nvidia_model:
            out.append(("nvidia", settings.nvidia_model))
        if fallbacks is None:
            from stockmaster.config import load_strategy
            fallbacks = load_strategy().llm.fallback_models
        for part in fallbacks:
            provider, _, model = part.partition(":")
            out.append((provider, model))
        if settings.gemini_model:
            out.append(("gemini", settings.gemini_model))
    seen, uniq = set(), []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def build_chain(spec: str | None = None, settings: Settings | None = None
                ) -> tuple[list[ChatModel], list[str]]:
    """Instantiate available models; returns (models, skipped reasons)."""
    settings = settings or Settings()
    models, skipped = [], []
    for provider, model_id in parse_chain(spec, settings):
        try:
            models.append(OpenAICompatModel(provider, model_id, settings))
        except (ProviderUnavailable, ValueError) as e:
            skipped.append(f"{provider}:{model_id}: {e}")
    return models, skipped
