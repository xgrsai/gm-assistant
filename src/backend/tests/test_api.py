"""End-to-end API tests: real graph, real SQLite checkpointer, scripted fake models."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.prompts import PromptSet
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from keeper.spec import ExperimentSpec
from server.main import create_app
from server.settings import REPO_ROOT, Settings

from .fakes import FakeFactory

SPOT = {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill", "target": "Spot Hidden"}
NO_ROLL = {"kind": "in_fiction", "needs_roll": False}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        name, data = "message", []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def _make(fake: FakeFactory, seed: int | None = 5, **spec) -> TestClient:
        settings = Settings(_env_file=None, checkpoint_path=tmp_path / "cp.db", app_db_path=tmp_path / "app.db",
                            artifacts_dir=tmp_path / "artifacts", data_dir=REPO_ROOT / "data")
        arms = {"test": ExperimentSpec(arm="test", generator="gen", aux="aux", seed=seed, **spec)}
        deps = Deps(llms=fake, prompts=PromptSet(), scenarios=ScenarioRepo(settings.data_dir / "scenarios"),
                    experiments=ExperimentRegistry(arms, "test"), runlog=RunLog(settings.artifacts_dir / "runs"))
        client = TestClient(create_app(settings, deps))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def create(client: TestClient, language: str = "uk") -> dict:
    r = client.post("/api/campaigns", json={"title": "Тест", "scenario_id": "demo-haunt", "language": language})
    assert r.status_code == 201, r.text
    return r.json()


def turn(client: TestClient, cid: str, text: str, turn_id: str | None = None):
    return client.post(f"/api/campaigns/{cid}/turns", json={"client_turn_id": turn_id or str(uuid.uuid4()),
                                                            "text": text})


def names(events):
    return [e for e, _ in events]


def test_health_meta_scenarios(make_client):
    c = make_client(FakeFactory())
    assert c.get("/api/health").json() == {"status": "ok"}
    assert c.get("/api/meta").json()["default_arm"] == "test"
    items = c.get("/api/scenarios").json()["items"]
    assert items[0]["id"] == "demo-haunt" and items[0]["title"]["uk"] == "Тиша на Пекарській"


def test_campaign_lifecycle(make_client):
    c = make_client(FakeFactory())
    camp = create(c)
    assert camp["scenario_title"] == "Тиша на Пекарській" and camp["arm"] == "test"
    assert [x["id"] for x in c.get("/api/campaigns").json()["items"]] == [camp["id"]]
    msgs = c.get(f"/api/campaigns/{camp['id']}/messages").json()
    assert len(msgs["items"]) == 1 and msgs["items"][0]["role"] == "keeper" and not msgs["incomplete"]
    assert msgs["investigator"]["name"] == "Олена Коваль" and msgs["investigator"]["san"] == 60
    assert c.get(f"/api/campaigns/{camp['id']}").json()["investigator"]["hp"] == 11
    assert c.delete(f"/api/campaigns/{camp['id']}").status_code == 204
    assert c.get(f"/api/campaigns/{camp['id']}/messages").status_code == 404
    assert c.get("/api/campaigns").json()["items"] == []


def test_turn_with_roll_streams_events(make_client, tmp_path):
    fake = FakeFactory(narrations=["Ви помічаєте свіжі подряпини на дошках."], adjudications=[SPOT])
    c = make_client(fake)
    cid = create(c)["id"]
    r = turn(c, cid, "Я піднімаю килим і роздивляюся підлогу.")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    seq = names(events)
    assert seq[0] == "progress" and seq[-1] == "done"
    assert seq.index("roll") < seq.index("token")
    assert [d["node"] for e, d in events if e == "progress"] == ["adjudicate", "mechanics", "narrate"]
    roll = next(d for e, d in events if e == "roll")
    assert roll["target"] == "Spot Hidden" and roll["value"] == 70 and roll["turn_no"] == 1
    streamed = "".join(d["text"] for e, d in events if e == "token")
    done = events[-1][1]
    assert done["message"]["text"] == streamed == "Ви помічаєте свіжі подряпини на дошках."
    assert done["message"]["rolls"] == [roll]
    msgs = c.get(f"/api/campaigns/{cid}/messages").json()["items"]
    assert [m["role"] for m in msgs] == ["keeper", "player", "keeper"] and msgs[2]["rolls"] == [roll]
    # the adjudicator saw the player's text wrapped as data
    assert "<player_turn>" in fake.structured_calls[0][1][1].content
    # one research record per turn
    lines = (tmp_path / "artifacts" / "runs").glob("*.jsonl")
    record = json.loads(next(lines).read_text(encoding="utf-8").splitlines()[-1])
    assert record["status"] == "done" and record["rolls"] == [roll] and record["spec_hash"]
    assert "model_card" not in record["models"]["generator"]   # no card: the keys the run log always had


def test_no_roll_turn(make_client):
    c = make_client(FakeFactory(narrations=["Тиша."], adjudications=[NO_ROLL]))
    cid = create(c)["id"]
    events = parse_sse(turn(c, cid, "Я чекаю.").text)
    assert "roll" not in names(events) and events[-1][0] == "done"
    assert events[-1][1]["message"]["rolls"] == []


def test_validation_and_duplicates(make_client):
    c = make_client(FakeFactory(narrations=["Раз."], adjudications=[NO_ROLL]))
    cid = create(c)["id"]
    assert turn(c, cid, "   ").status_code == 422
    assert c.post(f"/api/campaigns/{cid}/turns", json={"client_turn_id": "nope", "text": "x"}).status_code == 422
    assert turn(c, "missing", "x").status_code == 404
    tid = str(uuid.uuid4())
    assert parse_sse(turn(c, cid, "Раз", tid).text)[-1][0] == "done"
    r = turn(c, cid, "Раз", tid)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "duplicate_turn"


def test_unknown_skill_is_reported_to_narrator(make_client):
    fake = FakeFactory(narrations=["Уточніть, будь ласка."],
                       adjudications=[{**SPOT, "target": "Astrophysics"}])
    c = make_client(fake)
    cid = create(c)["id"]
    events = parse_sse(turn(c, cid, "Я рахую зорі").text)
    assert "roll" not in names(events) and events[-1][0] == "done"


def test_failed_narration_resumes_without_rerolling(make_client):
    fake = FakeFactory(narrations=[ConnectionError("boom"), "Нарешті ви бачите."], adjudications=[SPOT])
    c = make_client(fake)
    cid = create(c)["id"]
    events = parse_sse(turn(c, cid, "Оглядаю кімнату").text)
    first_roll = next(d for e, d in events if e == "roll")
    assert events[-1] == ("error", {"code": "upstream_unavailable", "retryable": True,
                                    "message": "The model provider is unavailable."})
    assert c.get(f"/api/campaigns/{cid}/messages").json()["incomplete"] is True
    r = turn(c, cid, "Ще одна дія")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "turn_incomplete"

    retry = parse_sse(c.post(f"/api/campaigns/{cid}/turns/retry").text)
    assert "roll" not in names(retry) and retry[-1][0] == "done"
    assert retry[-1][1]["message"]["rolls"] == [first_roll]  # the dice were not re-rolled
    assert len(fake.structured_calls) == 1                    # adjudication was not repeated
    msgs = c.get(f"/api/campaigns/{cid}/messages").json()
    assert not msgs["incomplete"] and [m["role"] for m in msgs["items"]] == ["keeper", "player", "keeper"]
    r = c.post(f"/api/campaigns/{cid}/turns/retry")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "nothing_to_retry"


def test_discard_incomplete_turn(make_client):
    fake = FakeFactory(narrations=[ConnectionError("boom"), "Після скасування."], adjudications=[NO_ROLL, NO_ROLL])
    c = make_client(fake)
    cid = create(c)["id"]
    assert parse_sse(turn(c, cid, "Невдалий хід").text)[-1][0] == "error"
    assert c.post(f"/api/campaigns/{cid}/turns/discard").json() == {"ok": True}
    msgs = c.get(f"/api/campaigns/{cid}/messages").json()
    assert not msgs["incomplete"] and [m["role"] for m in msgs["items"]] == ["keeper"]
    events = parse_sse(turn(c, cid, "Новий хід").text)
    assert events[-1][0] == "done" and events[-1][1]["message"]["text"] == "Після скасування."


def test_sanity_event_updates_sheet(make_client):
    adj = {"kind": "in_fiction", "needs_roll": True, "roll_type": "sanity", "sanity_event": "ghouls_feeding"}
    c = make_client(FakeFactory(narrations=["Жах."], adjudications=[adj]))
    cid = create(c)["id"]
    events = parse_sse(turn(c, cid, "Я дивлюся в темряву склепу").text)
    san = next(d for e, d in events if e == "roll" and d["kind"] == "sanity")
    assert san["notation"] == "1/1D8"
    assert events[-1][1]["investigator"]["san"] == 60 - san["san_loss"]


def test_new_game_day_resets_the_sanity_budget(make_client):
    lose_6 = {"kind": "in_fiction", "needs_roll": True, "roll_type": "sanity", "sanity_loss": "6/6"}
    fake = FakeFactory(narrations=["Жах.", "Тиша.", "Знову жах."],
                       adjudications=[lose_6, {"kind": "in_fiction", "needs_roll": False, "new_day": True},
                                      lose_6])
    c = make_client(fake, adjudicate_prompt_version="adjudicate-v3")
    cid = create(c)["id"]
    parse_sse(turn(c, cid, "Я дивлюся в темряву склепу").text)
    events = parse_sse(turn(c, cid, "Я йду спати, а вранці повертаюся").text)
    assert "roll" not in names(events)  # the new day goes through mechanics without dice
    assert [d["node"] for e, d in events if e == "progress"] == ["adjudicate", "mechanics", "narrate"]
    events = parse_sse(turn(c, cid, "Я знову спускаюся в склеп").text)
    san = next(d for e, d in events if e == "roll" and d["kind"] == "sanity")
    # 6 + 6 = 12 would reach 60 // 5 within one day; after the night only 6 of 54 // 5 = 10 count
    assert san["san_loss"] == 6 and not san["indefinite_insanity"]
    assert {n for n, _ in fake.structured_calls} == {"AdjudicationV3"}


def test_out_of_character_talk_never_starts_a_new_day(make_client):
    fake = FakeFactory(narrations=["Відповідь."],
                       adjudications=[{"kind": "meta", "needs_roll": False, "new_day": True}])
    c = make_client(fake, adjudicate_prompt_version="adjudicate-v3")
    cid = create(c)["id"]
    events = parse_sse(turn(c, cid, "Скільки триває сесія?").text)
    assert [d["node"] for e, d in events if e == "progress"] == ["adjudicate", "narrate"]


def test_rules_question_and_parse_failure(make_client):
    from langchain_core.exceptions import OutputParserException

    fake = FakeFactory(narrations=["Відповідь про правила.", "Продовжуємо."],
                       adjudications=[{"kind": "rules_question", "needs_roll": True, "roll_type": "skill",
                                       "target": "Spot Hidden"}, OutputParserException("bad json")])
    c = make_client(fake)
    cid = create(c)["id"]
    events = parse_sse(turn(c, cid, "Як працює штовхання кидка?").text)
    assert "roll" not in names(events) and events[-1][0] == "done"  # rules questions never roll
    events = parse_sse(turn(c, cid, "Я йду далі").text)
    assert events[-1][0] == "done"
