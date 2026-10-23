"""Test doubles: scripted structured outputs, a streaming fake chat model, and bag-of-words embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from keeper.llm import ModelProfile


class Script(Iterator):
    """Iterator of narrations; an Exception instance in the list is raised instead of returned."""

    def __init__(self, items: list):
        self.items = list(items)

    def __next__(self):
        if not self.items:
            raise StopIteration
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return AIMessage(content=item)


class RecordingChat(GenericFakeChatModel):
    """Fake chat model that appends every input message list to `seen` (shared with the factory)."""

    seen: Any = Field(default_factory=list)  # Any: pydantic would copy a list-typed field

    # GenericFakeChatModel streams through _generate, so recording here covers invoke and stream once each.
    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class FakeFactory:
    def __init__(self, narrations: list | None = None, adjudications: list | None = None,
                 queries: list | None = None):
        self.narrations = Script(narrations or [])
        self.adjudications = list(adjudications or [])
        self.queries = list(queries or [])
        self.structured_calls: list = []
        self.chat_inputs: list = []  # message lists the narrator model received, one per call
        self.responders: dict = {}  # schema name -> callable(messages) -> dict, takes precedence over the lists

    def profile(self, name: str) -> ModelProfile:
        return ModelProfile(provider="google_genai", model=f"fake-{name}", max_tokens=1024, context_window=32768)

    def chat(self, name: str, *, seed: int | None = None) -> GenericFakeChatModel:
        return RecordingChat(messages=self.narrations, seen=self.chat_inputs)

    def structured(self, name: str, schema, *, seed: int | None = None):
        def call(messages):
            self.structured_calls.append((schema.__name__, messages))
            if schema.__name__ in self.responders:
                item = self.responders[schema.__name__](messages)
            elif schema.__name__ == "SearchQuery":
                item = self.queries.pop(0) if self.queries else "general rules"
                item = {"query_en": item} if isinstance(item, str) else item
            else:
                item = self.adjudications.pop(0) if self.adjudications else {"kind": "in_fiction", "needs_roll": False}
            if isinstance(item, BaseException):
                raise item
            return schema.model_validate(item)

        return RunnableLambda(call)


class BagOfWordsEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words vectors: shared words -> higher cosine similarity."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
