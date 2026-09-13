"""RAG arm end to end: analyze -> retrieve (in-memory Qdrant over the real corpus) -> adjudicate -> narrate."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.prompts import PromptSet
from keeper.rag.corpus import load_corpus
from keeper.rag.retriever import Retriever
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from keeper.spec import ExperimentSpec, RetrievalSpec
from server.main import create_app
from server.settings import REPO_ROOT, Settings

from .fakes import BagOfWordsEmbeddings, FakeFactory
from .test_api import NO_ROLL, SPOT, parse_sse, turn
from .test_retriever import ingest

RETRIEVAL = RetrievalSpec(collection="kb", embedder="bow", quotas={"rules": 3, "lore": 1})


@pytest.fixture
def rag_client(tmp_path):
    clients = []

    def _make(fake: FakeFactory, *, ingest_corpus: bool = True) -> TestClient:
        emb = BagOfWordsEmbeddings()
        qdrant = AsyncQdrantClient(location=":memory:")
        if ingest_corpus:
            chunks = load_corpus([REPO_ROOT / "data" / "open" / "coc7e"], chunk_size=1000, overlap=200, root=REPO_ROOT)
            asyncio.run(ingest(qdrant, chunks, emb, name="kb"))
        settings = Settings(_env_file=None, checkpoint_path=tmp_path / "cp.db", app_db_path=tmp_path / "app.db",
                            artifacts_dir=tmp_path / "artifacts", data_dir=REPO_ROOT / "data")
        arms = {
            "rag": ExperimentSpec(arm="rag", generator="gen", aux="aux", seed=3, use_rag=True, retrieval=RETRIEVAL,
                                  prompt_version="keeper-v2", adjudicate_prompt_version="adjudicate-v2"),
            "plain": ExperimentSpec(arm="plain", generator="gen", aux="aux", seed=3, prompt_version="keeper-v2",
                                    adjudicate_prompt_version="adjudicate-v2"),
        }
        deps = Deps(llms=fake, prompts=PromptSet(), scenarios=ScenarioRepo(settings.data_dir / "scenarios"),
                    experiments=ExperimentRegistry(arms, "rag"), runlog=RunLog(settings.artifacts_dir / "runs"),
                    retriever=Retriever(qdrant, {"bow": emb}, {"bow": emb.dim}))
        client = TestClient(create_app(settings, deps))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def create(client, arm=None):
    body = {"title": "RAG", "scenario_id": "demo-haunt", "language": "uk"}
    if arm:
        body["arm"] = arm
    r = client.post("/api/campaigns", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_meta_arms_and_unknown_arm(rag_client):
    c = rag_client(FakeFactory())
    meta = c.get("/api/meta").json()
    assert meta["default_arm"] == "rag"
    assert {a["id"]: a["use_rag"] for a in meta["arms"]} == {"rag": True, "plain": False}
    r = c.post("/api/campaigns", json={"title": "x", "arm": "nope"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_arm"
    assert create(c)["use_rag"] is True and create(c, "plain")["use_rag"] is False


def test_rag_turn_retrieves_and_validates_citations(rag_client, tmp_path):
    fake = FakeFactory(
        queries=["Spot Hidden skill roll noticing hidden details bonus penalty dice"],
        adjudications=[SPOT],
        narrations=["Ви помічаєте подряпини.\n\n🎲 Пильність (Spot Hidden): успіх [R1]. Див. також [R9]."],
    )
    c = rag_client(fake)
    cid = create(c)["id"]
    events = parse_sse(turn(c, cid, "Я роздивляюся підлогу під килимом").text)
    stages = [d["node"] for e, d in events if e == "progress"]
    assert stages == ["analyze", "retrieve", "adjudicate", "mechanics", "narrate"]

    # the analyzer saw the player's Ukrainian text as data; the adjudicator and narrator got the knowledge base
    name, analyze_msgs = fake.structured_calls[0]
    assert name == "SearchQuery" and "<player_turn>" in analyze_msgs[1].content
    name, adjudicate_msgs = fake.structured_calls[1]
    assert name == "Adjudication"
    assert "RULES CONTEXT" in adjudicate_msgs[0].content and "[R1]" in adjudicate_msgs[0].content

    msg = events[-1][1]["message"]
    assert [x["marker"] for x in msg["citations"]] == [1] and msg["invalid_markers"] == [9]
    cite = msg["citations"][0]
    assert cite["source_type"] == "rules" and cite["chunk_id"] and cite["excerpt"] and "text" not in cite

    reloaded = c.get(f"/api/campaigns/{cid}/messages").json()["items"][-1]
    assert reloaded["citations"] == msg["citations"] and reloaded["invalid_markers"] == [9]

    record = json.loads(next((tmp_path / "artifacts" / "runs").glob("*.jsonl")).read_text(encoding="utf-8")
                        .splitlines()[-1])
    assert record["search_query"].startswith("Spot Hidden") and record["rules_hits"]
    assert record["citations"]["invalid_markers"] == [9]


def test_plain_arm_skips_retrieval(rag_client):
    fake = FakeFactory(adjudications=[NO_ROLL], narrations=["Тиша [R1]."])
    c = rag_client(fake)
    cid = create(c, "plain")["id"]
    events = parse_sse(turn(c, cid, "Я чекаю").text)
    assert [d["node"] for e, d in events if e == "progress"] == ["adjudicate", "narrate"]
    assert [n for n, _ in fake.structured_calls] == ["Adjudication"]
    msg = events[-1][1]["message"]
    assert msg["citations"] == [] and msg["invalid_markers"] == []


def test_analyze_parse_failure_falls_back_to_player_text(rag_client):
    from langchain_core.exceptions import OutputParserException

    fake = FakeFactory(queries=[OutputParserException("bad")], adjudications=[NO_ROLL], narrations=["Добре."])
    c = rag_client(fake)
    cid = create(c)["id"]
    assert parse_sse(turn(c, cid, "bonus dice?").text)[-1][0] == "done"


def test_startup_fails_without_collection(rag_client):
    with pytest.raises(RuntimeError, match="does not exist"):
        rag_client(FakeFactory(), ingest_corpus=False)
