"""Inspect retrieval from the command line.

cd src/backend
uv run python -m keeper.rag.search "how do bonus dice work"
uv run python -m keeper.rag.search "sanity loss for seeing a corpse" --arm play-gemini-rag
"""

from __future__ import annotations

import argparse
import asyncio

from qdrant_client import AsyncQdrantClient

from server.settings import get_settings

from ..experiments import ExperimentRegistry
from .embeddings import build_embeddings, load_profiles
from .retriever import Retriever


async def run(query: str, arm: str | None) -> None:
    s = get_settings()
    spec = ExperimentRegistry.from_dir(s.experiments_dir, s.default_arm).resolve(arm)
    profiles = load_profiles(s.config_dir / "embedders.yaml")
    client = AsyncQdrantClient(url=s.qdrant_url)
    try:
        retriever = Retriever(
            client,
            {n: build_embeddings(p, ollama_url=s.ollama_base_url) for n, p in profiles.items()},
            {n: p.dim for n, p in profiles.items()},
        )
        for h in await retriever.search(query, spec.retrieval, game_system="coc7e"):
            print(f"[R{h['marker']}] {h['score']:.3f} {h['source_type']:<6} {h['section']}")
            print("     " + h["excerpt"].replace("\n", " ")[:200])
    finally:
        await client.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--arm", default=None)
    a = p.parse_args()
    asyncio.run(run(a.query, a.arm))


if __name__ == "__main__":
    main()
