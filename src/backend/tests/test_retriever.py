from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from keeper.rag.corpus import chunk_markdown, load_corpus
from keeper.rag.retriever import Retriever, ensure_collection
from keeper.spec import RetrievalSpec

from .fakes import BagOfWordsEmbeddings

REPO = Path(__file__).resolve().parents[3]
SPEC = RetrievalSpec(collection="test", embedder="bow", quotas={"rules": 3, "lore": 1})


async def ingest(client, chunks, emb, name="test"):
    await ensure_collection(client, name, emb.dim)
    vectors = emb.embed_documents([c.text for c in chunks])
    await client.upsert(name, points=[models.PointStruct(id=c.id, vector=v, payload=c.payload())
                                      for c, v in zip(chunks, vectors, strict=True)])


@pytest.fixture
async def retriever():
    client = AsyncQdrantClient(location=":memory:")
    emb = BagOfWordsEmbeddings()
    chunks = load_corpus([REPO / "data" / "open" / "coc7e"], chunk_size=1000, overlap=200, root=REPO)
    secret = chunk_markdown("---\nsource_title: Secret\nsource_type: rules\ngame_system: coc7e\nvisibility: keeper\n"
                            "---\n# Secret\nbonus penalty dice tens die secret keeper-only passage", chunk_size=1000,
                            overlap=0)
    other = chunk_markdown("---\nsource_title: D&D\nsource_type: rules\ngame_system: dnd5e\n---\n# Advantage\n"
                           "bonus penalty dice tens die advantage", chunk_size=1000, overlap=0)
    await ingest(client, chunks + secret + other, emb)
    yield Retriever(client, {"bow": emb}, {"bow": emb.dim})
    await client.close()


async def test_quotas_order_markers_and_payload(retriever):
    hits = await retriever.search("bonus dice penalty dice tens die", SPEC, game_system="coc7e")
    assert [h["marker"] for h in hits] == [1, 2, 3, 4]
    assert [h["source_type"] for h in hits] == ["rules", "rules", "rules", "lore"]
    assert "Bonus and Penalty Dice" in hits[0]["section"]
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]
    assert hits[0]["text"].startswith("### Bonus and Penalty Dice")
    assert hits[0]["excerpt"].startswith("Bonus and Penalty Dice") and len(hits[0]["excerpt"]) <= 300


async def test_filters_game_system_and_visibility(retriever):
    hits = await retriever.search("bonus penalty dice tens die advantage secret", SPEC, game_system="coc7e")
    assert all(h["source"] not in {"Secret", "D&D"} for h in hits)
    dnd = await retriever.search("advantage", SPEC, game_system="dnd5e")
    assert {h["source"] for h in dnd} == {"D&D"}


async def test_min_score_drops_weak_hits(retriever):
    spec = SPEC.model_copy(update={"min_score": 0.99})
    assert await retriever.search("bonus dice", spec, game_system="coc7e") == []


async def test_verify(retriever):
    await retriever.verify(SPEC)
    with pytest.raises(RuntimeError, match="does not exist"):
        await retriever.verify(SPEC.model_copy(update={"collection": "missing"}))
    with pytest.raises(RuntimeError, match="unknown embedder"):
        await retriever.verify(SPEC.model_copy(update={"embedder": "nope"}))
    await ensure_collection(retriever.client, "small", 8)
    with pytest.raises(RuntimeError, match="vector size"):
        await retriever.verify(SPEC.model_copy(update={"collection": "small"}))
