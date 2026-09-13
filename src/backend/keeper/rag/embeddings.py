"""Embedding profiles (config/embedders.yaml) and task-prefix handling."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict


class EmbedderProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["ollama"]
    model: str
    dim: int
    query_prefix: str = ""
    document_prefix: str = ""


class PrefixedEmbeddings(Embeddings):
    """Adds task prefixes by composition.

    Do not subclass OllamaEmbeddings to do this: its embed_query/aembed_query call
    self.embed_documents/self.aembed_documents, so a prefixing subclass double-prefixes queries.
    """

    def __init__(self, inner: Embeddings, query_prefix: str = "", document_prefix: str = ""):
        self.inner = inner
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_documents([self.query_prefix + text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.inner.aembed_documents([self.document_prefix + t for t in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.inner.aembed_documents([self.query_prefix + text]))[0]


def load_profiles(path: Path) -> dict[str, EmbedderProfile]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: EmbedderProfile(**p) for name, p in raw.items()}


def build_embeddings(profile: EmbedderProfile, *, ollama_url: str) -> PrefixedEmbeddings:
    match profile.provider:
        case "ollama":
            from langchain_ollama import OllamaEmbeddings

            inner = OllamaEmbeddings(model=profile.model, base_url=ollama_url)
    return PrefixedEmbeddings(inner, profile.query_prefix, profile.document_prefix)
