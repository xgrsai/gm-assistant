"""Ingest a Markdown corpus into Qdrant.

    cd src/backend
    uv run python -m keeper.rag.ingest ../../data/open/coc7e            # default: nomic15, 1000/200
    uv run python -m keeper.rag.ingest ../../data/open/coc7e ../../data/private/coc7e --recreate

The collection name encodes game system, embedder and chunking (e.g. coc7e__nomic15__c1000o200), so a different
chunking or embedder never overwrites an existing collection. A chunk dump for inspection and for building gold
retrieval sets is written to artifacts/corpus/<collection>/chunks.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from qdrant_client import AsyncQdrantClient, models

from server.settings import REPO_ROOT, get_settings

from .corpus import load_corpus
from .embeddings import build_embeddings, load_profiles
from .retriever import ensure_collection

BATCH = 32


def collection_name(game_system: str, embedder: str, chunk_size: int, overlap: int) -> str:
    return f"{game_system}__{embedder}__c{chunk_size}o{overlap}"


async def ingest(
    paths: list[Path], *, embedder: str, chunk_size: int, overlap: int, collection: str | None, recreate: bool
) -> str:
    s = get_settings()
    profile = load_profiles(s.config_dir / "embedders.yaml")[embedder]
    chunks = load_corpus(paths, chunk_size=chunk_size, overlap=overlap, root=REPO_ROOT)
    if not chunks:
        raise SystemExit("no chunks found")
    systems = {c.metadata["game_system"] for c in chunks}
    if len(systems) != 1 and not collection:
        raise SystemExit(f"several game systems {sorted(systems)}: pass --collection explicitly")
    name = collection or collection_name(systems.pop(), embedder, chunk_size, overlap)

    emb = build_embeddings(profile, ollama_url=s.ollama_base_url)
    client = AsyncQdrantClient(url=s.qdrant_url)
    try:
        await ensure_collection(client, name, profile.dim, recreate=recreate)
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vectors = await emb.aembed_documents([c.text for c in batch])
            await client.upsert(
                name,
                wait=True,
                points=[
                    models.PointStruct(id=c.id, vector=v, payload=c.payload())
                    for c, v in zip(batch, vectors, strict=True)
                ],
            )
            print(f"  upserted {min(i + BATCH, len(chunks))}/{len(chunks)}")
        count = (await client.count(name, exact=True)).count
    finally:
        await client.close()

    out = s.artifacts_dir / "corpus" / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
    by_type = Counter(c.metadata["source_type"] for c in chunks)
    print(f"collection {name}: {len(chunks)} chunks ingested ({dict(by_type)}), {count} points total")
    print(f"chunk dump: {out / 'chunks.jsonl'}")
    return name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="Markdown files or directories")
    p.add_argument("--embedder", default="nomic15")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--collection", default=None)
    p.add_argument("--recreate", action="store_true", help="drop the collection first")
    a = p.parse_args()
    asyncio.run(
        ingest(
            a.paths,
            embedder=a.embedder,
            chunk_size=a.chunk_size,
            overlap=a.overlap,
            collection=a.collection,
            recreate=a.recreate,
        )
    )


if __name__ == "__main__":
    main()
