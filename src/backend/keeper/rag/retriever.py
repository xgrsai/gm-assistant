"""Semantic search in Qdrant with per-source-type quotas and exact (deterministic) cosine search."""

from __future__ import annotations

import re

from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, models

from ..spec import RetrievalSpec

EXCERPT_CHARS = 300
HEADING_MARKS = re.compile(r"^#+\s*", re.M)
KEYWORD_FIELDS = ("metadata.game_system", "metadata.source_type", "metadata.visibility")


def match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


class Retriever:
    def __init__(self, client: AsyncQdrantClient, embedders: dict[str, Embeddings], dims: dict[str, int]):
        self.client = client
        self.embedders = embedders
        self.dims = dims

    async def search(self, query: str, spec: RetrievalSpec, *, game_system: str,
                     visibility: str = "public") -> list[dict]:
        vector = await self.embedders[spec.embedder].aembed_query(query)
        found = []
        for source_type, limit in spec.quotas.items():
            if limit <= 0:
                continue
            res = await self.client.query_points(
                collection_name=spec.collection, query=vector, limit=limit, with_payload=True,
                search_params=models.SearchParams(exact=spec.exact), score_threshold=spec.min_score,
                query_filter=models.Filter(must=[match("metadata.game_system", game_system),
                                                 match("metadata.source_type", source_type),
                                                 match("metadata.visibility", visibility)]))
            found.extend(res.points)
        hits = []
        for i, p in enumerate(found, start=1):
            meta = p.payload["metadata"]
            text = p.payload["page_content"]
            hits.append({
                "marker": i, "chunk_id": str(p.id), "score": round(float(p.score), 4),
                "source": meta.get("source_title"), "section": meta.get("section_path"),
                "source_type": meta.get("source_type"), "page": meta.get("page"),
                "text": text, "excerpt": HEADING_MARKS.sub("", text)[:EXCERPT_CHARS],
            })
        return hits

    async def verify(self, spec: RetrievalSpec) -> None:
        """Fail fast at startup with an actionable message instead of on the first player turn."""
        if spec.embedder not in self.embedders:
            raise RuntimeError(f"unknown embedder profile {spec.embedder!r}")
        if not await self.client.collection_exists(spec.collection):
            raise RuntimeError(f"Qdrant collection {spec.collection!r} does not exist; run the ingestion CLI "
                               f"(uv run python -m keeper.rag.ingest ../../data/open/coc7e)")
        info = await self.client.get_collection(spec.collection)
        size = info.config.params.vectors.size
        if size != self.dims[spec.embedder]:
            raise RuntimeError(f"collection {spec.collection!r} has vector size {size}, "
                               f"embedder {spec.embedder!r} produces {self.dims[spec.embedder]}")
        await self.embedders[spec.embedder].aembed_query("ping")


async def ensure_collection(client: AsyncQdrantClient, name: str, dim: int, *, recreate: bool = False) -> None:
    if recreate and await client.collection_exists(name):
        await client.delete_collection(name)
    if not await client.collection_exists(name):
        await client.create_collection(name, vectors_config=models.VectorParams(size=dim,
                                                                                distance=models.Distance.COSINE))
    for field_name in KEYWORD_FIELDS:
        await client.create_payload_index(name, field_name=field_name, field_schema=models.PayloadSchemaType.KEYWORD)
