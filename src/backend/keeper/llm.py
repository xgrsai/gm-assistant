"""Model profiles and a factory that builds each chat model once (no client per request)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, SecretStr


class ModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["google_genai", "ollama", "vllm"]
    model: str
    temperature: float | None = None   # ignored for Gemini >= 3.6 (provider-fixed sampling)
    max_tokens: int = 1024             # Gemini: thinking tokens count toward output, keep generous
    thinking_level: str | None = None  # Gemini 3+: minimal | low | medium | high
    context_window: int = 32768        # used to budget history
    num_ctx: int = 16384               # Ollama only
    finetuned: bool = False
    base: str | None = None            # FT profile -> its base profile
    quant: str | None = None           # logged: bf16 | Q4_K_M | ...


class ModelFactory:
    def __init__(self, profiles: dict[str, ModelProfile], *, google_api_key: SecretStr | None = None,
                 ollama_url: str = "http://localhost:11434", vllm_url: str | None = None,
                 timeout: float = 120.0, google_rpm: int = 30):
        self.profiles = profiles
        self._key = google_api_key
        self._ollama_url = ollama_url
        self._vllm_url = vllm_url
        self._timeout = timeout
        self._cache: dict[tuple[str, int | None], BaseChatModel] = {}
        # One limiter per provider: quotas are per project/model, not per model instance.
        # Bucket of 2 = one whole turn (adjudicate + narrate) passes without waiting after the player pauses.
        google = InMemoryRateLimiter(requests_per_second=google_rpm / 60, check_every_n_seconds=0.1,
                                     max_bucket_size=2)
        google.available_tokens = google.max_bucket_size  # starts empty by default: first turn would stall
        self._limiters = {"google_genai": google}

    @classmethod
    def from_yaml(cls, path: Path, **kw) -> ModelFactory:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({name: ModelProfile(**p) for name, p in raw.items()}, **kw)

    def profile(self, name: str) -> ModelProfile:
        try:
            return self.profiles[name]
        except KeyError:
            raise ValueError(f"unknown model profile {name!r}") from None

    def chat(self, name: str, *, seed: int | None = None) -> BaseChatModel:
        key = (name, seed)
        if key not in self._cache:
            self._cache[key] = self._build(self.profile(name), seed)
        return self._cache[key]

    def structured(self, name: str, schema: type[BaseModel], *, seed: int | None = None) -> Runnable:
        return (self.chat(name, seed=seed)
                .with_structured_output(schema, method="json_schema")
                .with_config(tags=["nostream"]))  # never leaks into the token stream

    def _build(self, p: ModelProfile, seed: int | None) -> BaseChatModel:
        match p.provider:
            case "google_genai":
                from langchain_google_genai import ChatGoogleGenerativeAI

                kw: dict = {}
                if self._key:  # pass only when set; otherwise the SDK reads GOOGLE_API_KEY itself
                    kw["api_key"] = self._key.get_secret_value()
                if p.thinking_level:
                    kw["thinking_level"] = p.thinking_level
                if seed is not None:
                    kw["seed"] = seed
                if p.temperature is not None:
                    kw["temperature"] = p.temperature
                return ChatGoogleGenerativeAI(model=p.model, max_tokens=p.max_tokens, timeout=self._timeout,
                                              max_retries=3, rate_limiter=self._limiters["google_genai"], **kw)
            case "ollama":
                from langchain_ollama import ChatOllama

                kw = {"seed": seed} if seed is not None else {}
                if p.temperature is not None:
                    kw["temperature"] = p.temperature
                return ChatOllama(model=p.model, base_url=self._ollama_url, num_ctx=p.num_ctx,
                                  num_predict=p.max_tokens, reasoning=False,
                                  client_kwargs={"timeout": self._timeout}, **kw)
            case "vllm":  # OpenAI-compatible; a LoRA adapter is selected by model name
                from langchain_openai import ChatOpenAI

                if not self._vllm_url:
                    raise ValueError("VLLM_BASE_URL is not set")
                kw = {"seed": seed} if seed is not None else {}
                if p.temperature is not None:
                    kw["temperature"] = p.temperature
                return ChatOpenAI(model=p.model, base_url=self._vllm_url, api_key="EMPTY",
                                  max_tokens=p.max_tokens, timeout=self._timeout, max_retries=2, **kw)
        raise ValueError(f"unknown provider {p.provider}")
