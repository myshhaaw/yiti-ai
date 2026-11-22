"""Cached, rate-limited access to the chat and embedding API.

Every LLM and embedding call in the project goes through Provider (or FakeProvider in tests).
Responses are cached on disk in .cache/, keyed by sha256 of (provider, model, request payload),
so re-running the same request never costs quota.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import ssl
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openai

from .config import ROOT

JSON_INSTRUCTION = "Reply with a single JSON object only: no markdown fences and no other text."

# Google resets free-tier daily quotas at midnight Pacific time. A fixed UTC-8 offset is used so no
# timezone database is needed; during daylight saving it resets our counter an hour late (conservative).
QUOTA_TZ = timezone(timedelta(hours=-8))


class ProviderError(RuntimeError):
    pass


class QuotaError(ProviderError):
    pass


class JSONParseError(ProviderError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def extract_json(text: str) -> dict:
    """Return the first JSON object in text, tolerating ```json fences and surrounding prose."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise JSONParseError("no JSON object found in model reply", text)


class RateLimiter:
    """Sliding 60-second window: blocks until another request is allowed."""

    def __init__(self, per_minute: int, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.per_minute, self.sleep, self.clock = per_minute, sleep, clock
        self.times: deque[float] = deque()

    def wait(self) -> None:
        now = self.clock()
        while self.times and now - self.times[0] >= 60:
            self.times.popleft()
        if len(self.times) >= self.per_minute:
            self.sleep(60 - (now - self.times[0]))
            self.times.popleft()
        self.times.append(self.clock())


class DailyUsage:
    """Successful network requests per quota day, persisted so separate runs share one budget."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def today() -> str:
        return datetime.now(QUOTA_TZ).date().isoformat()

    def count(self, kind: str) -> int:
        return self._load().get(self.today(), {}).get(kind, 0)

    def add(self, kind: str) -> None:
        data = self._load()
        day = data.setdefault(self.today(), {})
        day[kind] = day.get(kind, 0) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def provider_error_category(exc: BaseException) -> str:
    """Return safe diagnostics only; never include provider messages or request content."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None) or getattr(exc, "code", None)
    if status == 401:
        return "authentication_rejected"
    if status == 403:
        return "access_or_project_rejected"
    if status == 404:
        return "model_or_endpoint_not_found"
    if status == 429 or isinstance(exc, QuotaError):
        return "rate_limit_or_quota"
    if status in (400, 422):
        return "request_or_model_compatibility"
    # OpenAI's transport errors wrap httpx/socket exceptions. Inspect only exception
    # classes and errno values: error strings can contain URLs, proxy details, or headers.
    chain: list[BaseException] = []
    seen: set[int] = set()
    pending = [exc]
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        chain.append(item)
        for child in (item.__cause__, item.__context__):
            if isinstance(child, BaseException):
                pending.append(child)
    class_names = {type(item).__name__.lower() for item in chain}
    if any(isinstance(item, (ssl.SSLError, ssl.CertificateError)) for item in chain) or any(
        "ssl" in name or "tls" in name or "certificate" in name for name in class_names
    ):
        return "tls_certificate_failure"
    if any("proxy" in name for name in class_names):
        return "proxy_connection_failure"
    if any(isinstance(item, socket.gaierror) for item in chain) or class_names & {
        "name_resolution_error", "gaierror"
    }:
        return "dns_resolution_failure"
    if any(name in class_names for name in ("connecttimeout", "connecttimeouterror")):
        return "connection_timeout"
    if any(name in class_names for name in ("readtimeout", "readtimeouterror")):
        return "read_timeout"
    if any(name in class_names for name in ("writetimeout", "pooltimeout")):
        return "network_timeout"
    if any(isinstance(item, PermissionError) or getattr(item, "winerror", None) == 10013
           for item in chain):
        return "network_permission_denied"
    errnos = {getattr(item, "errno", None) for item in chain}
    if any(errno in errnos for errno in (10061, 111)):  # Windows/Linux connection refused
        return "tcp_connection_refused"
    if any(errno in errnos for errno in (10054, 104)):  # Windows/Linux connection reset
        return "connection_reset"
    if (status is not None and status >= 500) or class_names & {
        "servererror", "internalservererror", "serviceunavailable"
    }:
        return "provider_unavailable"
    if isinstance(exc, openai.APITimeoutError):
        return "network_timeout"
    if isinstance(exc, openai.APIConnectionError):
        return "network_connection"
    if isinstance(exc, ProviderError):
        return "provider_configuration_or_response"
    return "unexpected_generation_error"


def provider_error_types(exc: BaseException) -> tuple[str, ...]:
    """Safe exception class names for local logs; deliberately excludes messages and request data."""
    names: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(names) < 5:
        seen.add(id(current))
        name = type(current).__name__
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,59}", name):
            names.append(name)
        current = current.__cause__ or current.__context__
    return tuple(names)


def resolve_provider_cfg(cfg: dict, env: Any = None) -> dict:
    """The provider block to use: LLM_PROVIDER from .env overrides config.yaml provider.name.

    config.yaml keeps the default provider in `provider` and the others in `providers.<name>`. A cfg with no
    `providers` section (tests, v1 runs) is used as is. For Azure the model names are deployment names, and
    AZURE_CHAT_DEPLOYMENT / AZURE_EMBED_DEPLOYMENT in .env override the ones in config.yaml.
    """
    env = os.environ if env is None else env
    default, others = cfg["provider"], cfg.get("providers") or {}
    wanted = (env.get("LLM_PROVIDER") or "").strip().lower() or default["name"]
    if not others or wanted == default["name"]:
        p = default
    elif wanted in others:
        p = {"name": wanted, **others[wanted]}
    else:
        raise ProviderError(f"LLM_PROVIDER={wanted!r} is not configured; choose one of "
                            f"{', '.join([default['name'], *others])}")
    if p["name"] == "azure":
        p = dict(p)
        for key, var in (("chat_model", "AZURE_CHAT_DEPLOYMENT"), ("embed_model", "AZURE_EMBED_DEPLOYMENT")):
            if env.get(var):
                p[key] = env[var]
    return p


class Provider:
    """Gemini and Azure chat/embedding clients with disk cache and request throttling.

    Gemini uses Google's first-party SDK; Azure continues to use the OpenAI SDK.
    Provider selection is controlled by resolve_provider_cfg().
    """

    def __init__(self, cfg: dict, cache_dir: Path | None = None, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep, env: Any = None):
        provider_env = os.environ if env is None else env
        p = resolve_provider_cfg(cfg, env=provider_env)
        self.name = p["name"]
        # Real Gemini traffic uses Google's first-party SDK. OpenAI-shaped injected
        # fakes remain supported for existing fixtures only.
        self._native_gemini = self.name == "gemini" and (client is None or hasattr(client, "models"))
        # Keep answers cached by the previous OpenAI-compatible transport separate
        # from responses produced by Google's native SDK.
        self._cache_provider_name = "gemini_google_genai" if self._native_gemini else self.name
        self.embed_dim = p.get("embed_dim")  # when set, every embedding must have this length
        self.chat_model = p["chat_model"]
        self.embed_model = p["embed_model"]
        self.temperature = p.get("temperature", 0)
        self.reasoning_effort = p.get("reasoning_effort")
        self.embed_batch_size = p.get("embed_batch_size", 64)
        self.daily_caps = {"chat": p.get("chat_requests_per_day"), "embed": p.get("embed_requests_per_day")}
        self.limiters = {"chat": RateLimiter(p["chat_requests_per_minute"], sleep),
                         "embed": RateLimiter(p["embed_requests_per_minute"], sleep)}
        self.sleep = sleep
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / ".cache"
        self.usage = DailyUsage(self.cache_dir / "usage.json")

        # Downgraded at runtime if the endpoint rejects them; the state is reported by metadata().
        self.json_mode = True
        self.effort_active = self.reasoning_effort is not None

        if client is None:
            key = provider_env.get(p["api_key_env"])
            if not key:
                raise ProviderError(f"{p['api_key_env']} is not set; add it to .env")
            if self.name == "azure":
                endpoint = provider_env.get(p["endpoint_env"])
                if not endpoint:
                    raise ProviderError(f"{p['endpoint_env']} is not set; add it to .env")
                client = openai.AzureOpenAI(azure_endpoint=endpoint, api_key=key,
                                            api_version=p["api_version"], max_retries=0, timeout=45.0)
            elif self.name == "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        timeout=45_000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
            else:
                client = openai.OpenAI(base_url=p["base_url"], api_key=key, max_retries=0, timeout=45.0)
        self.client = client

        self.calls_made = 0
        self.cache_hits = 0

    # ---- cache ----

    def _key(self, payload: dict) -> str:
        blob = json.dumps({"provider": self._cache_provider_name, "payload": payload},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> dict | None:
        try:
            return json.loads(self._cache_path(key).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def _cache_put(self, key: str, value: dict) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # ---- network ----

    def _request(self, kind: str, fn: Callable[[], Any]) -> Any:
        """Make one provider request. Never repeat a request that may incur API usage."""
        cap = self.daily_caps[kind]
        if cap is not None and self.usage.count(kind) >= cap:
            raise QuotaError(f"daily {kind} request cap of {cap} reached")
        self.limiters[kind].wait()
        self.calls_made += 1
        result = fn()
        self.usage.add(kind)
        return result

    def chat_json(self, messages: list[dict]) -> dict:
        """Send a chat request asking for JSON and return the parsed object (raises JSONParseError)."""
        payload = {"model": self.chat_model, "messages": messages, "temperature": self.temperature,
                   "response_format": {"type": "json_object"}, "reasoning_effort": self.reasoning_effort}
        key = self._key(payload)
        hit = self._cache_get(key)
        if hit is not None:
            self.cache_hits += 1
            return extract_json(hit["content"])
        content, sent = self._chat_network(messages)
        self._cache_put(key, {"content": content, "sent": sent})
        return extract_json(content)

    def chat_text(self, messages: list[dict]) -> str:
        """Send a chat request and return the reply as plain text (used for grounded answers and drafts).
        Cached like chat_json; the cache key has no response_format, so the two never collide."""
        payload = {"model": self.chat_model, "messages": messages, "temperature": self.temperature,
                   "reasoning_effort": self.reasoning_effort}
        key = self._key(payload)
        hit = self._cache_get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit["content"]
        content, sent = self._chat_network(messages, want_json=False)
        self._cache_put(key, {"content": content, "sent": sent})
        return content

    def _chat_network(self, messages: list[dict], want_json: bool = True) -> tuple[str, dict]:
        if self._native_gemini:
            from google.genai import types
            system_parts = [m["content"] for m in messages if m.get("role") == "system"]
            contents = [
                {"role": "model" if m.get("role") == "assistant" else "user",
                 "parts": [{"text": str(m.get("content", ""))}]}
                for m in messages if m.get("role") in {"assistant", "user"}
            ]
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction="\n\n".join(system_parts) or None,
                response_mime_type="application/json" if want_json else None,
                max_output_tokens=2048,
            )
            resp = self._request("chat", lambda: self.client.models.generate_content(
                model=self.chat_model, contents=contents, config=config))
            content = getattr(resp, "text", None) or ""
            if not content.strip():
                raise ProviderError("Gemini returned no text content")
            sent = {"model": self.chat_model, "temperature": self.temperature,
                    "response_mime_type": "application/json" if want_json else None,
                    "thinking_level": None}
            return content, sent
        kwargs: dict[str, Any] = {"model": self.chat_model, "temperature": self.temperature}
        if not want_json:
            kwargs["messages"] = messages
        elif self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"] = messages
        else:
            kwargs["messages"] = [{"role": "system", "content": JSON_INSTRUCTION}] + messages
        if self.effort_active:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self._request("chat", lambda: self.client.chat.completions.create(**kwargs))
        sent = {k: v for k, v in kwargs.items() if k != "messages"}
        return resp.choices[0].message.content or "", sent

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts; cached per text, uncached ones sent in batches of embed_batch_size."""
        out: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            hit = self._cache_get(self._key({"model": self.embed_model, "input": text}))
            if hit is not None:
                self.cache_hits += 1
                self._check_dim(hit["embedding"])
                out[i] = hit["embedding"]
            else:
                missing.setdefault(text, []).append(i)
        pending = list(missing)
        for start in range(0, len(pending), self.embed_batch_size):
            batch = pending[start:start + self.embed_batch_size]
            if self._native_gemini:
                resp = self._request("embed", lambda: self.client.models.embed_content(
                    model=self.embed_model, contents=batch))
                data = getattr(resp, "embeddings", [])
            else:
                resp = self._request("embed", lambda: self.client.embeddings.create(model=self.embed_model, input=batch))
                data = resp.data
            if len(data) != len(batch):
                raise ProviderError(f"sent {len(batch)} texts to embed but got {len(data)} embeddings back")
            # Gemini omits index 0 (protobuf drops zero values), so a missing index means 0.
            if self._native_gemini:
                vectors = [list(item.values) for item in data]
            else:
                vectors = [list(item.embedding) for item in sorted(data, key=lambda d: d.index or 0)]
            for text, vec in zip(batch, vectors):
                self._check_dim(vec)
                self._cache_put(self._key({"model": self.embed_model, "input": text}), {"embedding": vec})
                for i in missing[text]:
                    out[i] = vec
        return out  # type: ignore[return-value]

    def _check_dim(self, vec: list[float]) -> None:
        if self.embed_dim is not None and len(vec) != self.embed_dim:
            raise ProviderError(f"{self.name} embedding model {self.embed_model!r} returned {len(vec)} dimensions "
                                f"but config expects {self.embed_dim}; fix embed_dim or the deployment")

    def check_embed_dim(self) -> int:
        """Embed one fixed text (cached after the first run) and fail loudly if its length is not embed_dim."""
        [vec] = self.embed(["embedding dimension check"])
        self._check_dim(vec)
        return len(vec)

    def metadata(self) -> dict:
        meta = {"provider": self.name, "chat_model": self.chat_model, "embed_model": self.embed_model,
                "temperature": self.temperature,
                "reasoning_effort_requested": self.reasoning_effort,
                "reasoning_effort_sent": self.reasoning_effort if self.effort_active else None,
                "json_mode": "response_format" if self.json_mode else "prompt_instruction"}
        if self.embed_dim is not None:
            meta["embed_dim"] = self.embed_dim
        return meta

    def stats(self) -> dict:
        return {"calls_made": self.calls_made, "cache_hits": self.cache_hits}


def make_provider(cfg: dict, **kwargs) -> Provider:
    """Provider for application startup (server, v2): builds it, then verifies the embedding size when
    config sets embed_dim. v1 and the evaluation build Provider directly, so they make no extra call."""
    provider = Provider(cfg, **kwargs)
    if provider.embed_dim is not None:
        provider.check_embed_dim()
    return provider


class FakeProvider:
    """Deterministic stand-in for tests: no network and no disk.

    chat replies come from `chat_replies` (consumed in order) or `chat_fn(messages)`; each may be a dict
    or a raw string. Plain-text replies (chat_text) come from `text_replies` or `text_fn(messages)`.
    Embeddings are hashed bag-of-words vectors, so texts sharing words are similar. Requests are recorded in
    `chat_requests` / `text_requests` / `embed_requests`. With `embed_dim` set, embed() checks the vector
    length like the real Provider (a dim that differs from embed_dim raises ProviderError).
    """

    name = "fake"
    chat_model = "fake-chat"
    embed_model = "fake-embed"

    def __init__(self, chat_replies: list[dict | str] | None = None,
                 chat_fn: Callable[[list[dict]], dict | str] | None = None, dim: int = 64,
                 text_replies: list[str] | None = None, text_fn: Callable[[list[dict]], str] | None = None,
                 embed_dim: int | None = None):
        self.chat_replies = list(chat_replies or [])
        self.chat_fn = chat_fn
        self.text_replies = list(text_replies or [])
        self.text_fn = text_fn
        self.dim = dim
        self.embed_dim = embed_dim
        self.chat_requests: list[list[dict]] = []
        self.text_requests: list[list[dict]] = []
        self.embed_requests: list[list[str]] = []
        self.calls_made = 0
        self.cache_hits = 0

    def chat_json(self, messages: list[dict]) -> dict:
        self.calls_made += 1
        self.chat_requests.append(messages)
        reply = self.chat_fn(messages) if self.chat_fn else self.chat_replies.pop(0)
        return reply if isinstance(reply, dict) else extract_json(reply)

    def chat_text(self, messages: list[dict]) -> str:
        self.calls_made += 1
        self.text_requests.append(messages)
        return self.text_fn(messages) if self.text_fn else self.text_replies.pop(0)

    def check_embed_dim(self) -> int:
        [vec] = self.embed(["embedding dimension check"])
        return len(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls_made += 1
        self.embed_requests.append(list(texts))
        if self.embed_dim is not None and self.dim != self.embed_dim:
            raise ProviderError(f"fake embedding model returned {self.dim} dimensions "
                                f"but config expects {self.embed_dim}")
        vecs = []
        for text in texts:
            v = np.zeros(self.dim)
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
            vecs.append(v.tolist() if v.any() else [1.0] + [0.0] * (self.dim - 1))
        return vecs

    def metadata(self) -> dict:
        return {"provider": self.name, "chat_model": self.chat_model, "embed_model": self.embed_model}

    def stats(self) -> dict:
        return {"calls_made": self.calls_made, "cache_hits": self.cache_hits}
