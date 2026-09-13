"""Test doubles: a model factory whose structured calls return scripted adjudications and whose chat model streams."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

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


class FakeFactory:
    def __init__(self, narrations: list | None = None, adjudications: list[dict] | None = None):
        self.narrations = Script(narrations or [])
        self.adjudications = list(adjudications or [])
        self.structured_calls: list = []

    def profile(self, name: str) -> ModelProfile:
        return ModelProfile(provider="google_genai", model=f"fake-{name}", max_tokens=1024, context_window=32768)

    def chat(self, name: str, *, seed: int | None = None) -> GenericFakeChatModel:
        return GenericFakeChatModel(messages=self.narrations)

    def structured(self, name: str, schema, *, seed: int | None = None):
        def call(messages):
            self.structured_calls.append(messages)
            item = self.adjudications.pop(0) if self.adjudications else {"kind": "in_fiction", "needs_roll": False}
            if isinstance(item, BaseException):
                raise item
            return schema.model_validate(item)

        return RunnableLambda(call)
