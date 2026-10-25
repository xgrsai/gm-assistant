"""External judge: the pure judge functions, the export of blind batches and the import of the judgments.

Offline: fake models and in-memory Qdrant like test_evals.py; the "judge agents" are a few lines that write outputs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from qdrant_client import AsyncQdrantClient

from evals import calibrate, external_judge
from evals.datasets import ConflictProbe, EvalDatasets
from evals.experiment import EvalExperiment
from evals.external_judge import (
    ImportResult,
    batch_name,
    export_batches,
    export_dir,
    import_judgments,
    plan_batches,
    report_import,
    write_lines,
)
from evals.harness import Harness, record_key
from evals.judge import (
    PROMPTS,
    ConflictJudgment,
    QAJudgment,
    conflict_prompt,
    conflict_result,
    judge_conflict_answer,
    judge_rules_answer,
    rules_prompt,
    rules_result,
    swap_order,
)
from evals.metrics import summarize
from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.llm import ModelFactory, ModelProfile
from keeper.prompts import PromptSet
from keeper.rag.corpus import load_corpus
from keeper.rag.retriever import Retriever
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from server.settings import REPO_ROOT, Settings

from .fakes import BagOfWordsEmbeddings, FakeFactory, Script
from .test_evals import EXPERIMENT, make_fake, probe_items, write_datasets
from .test_retriever import ingest

BACKEND = Path(__file__).resolve().parents[1]
JUDGE = "judge"
# cp-001 and cp-002 show the facts in the given order, cp-004 swapped (swap_order is a hash of the id)
PROBES = [*probe_items(), {**probe_items()[0], "id": "cp-004"}]
EXTERNAL = {
    **EXPERIMENT, "judge": JUDGE,
    "arms": [{"arm": "zz-baseline", "generator": "gen", "use_rag": False},
             {"arm": "zz-retrieval", "generator": "gen", "use_rag": True}],
    "tasks": {"retrieval": {"enabled": False}, "adjudication": {"enabled": False},
              "rules_qa": {"context_query": "question_en"}},
}
# rules QA: 3 questions x 2 arms; conflict probes: 3 probes x (no_context, and perturbed + original with RAG)
N_ANSWERS = 3 * 2 + 3 * 3
JUDGMENTS = {"QAJudgment": {"verdict": "correct", "key_facts_present": [0, 1, 9], "contradicts_reference": False,
                            "abstained": False, "reason": "fine"},
             "ConflictJudgment": {"follows": "a", "reason": "fine"}}


# ------------------------------------------------------------------------------------------ the judge functions


def datasets(tmp_path: Path) -> EvalDatasets:
    return EvalDatasets.load(write_datasets(tmp_path, probes=PROBES))


async def test_the_judge_functions_send_the_prompt_and_return_the_result_they_always_did(tmp_path):
    ds = datasets(tmp_path)
    qa = {i.id: i for i in ds.rules_qa}
    fake = FakeFactory()
    fake.responders = {
        "QAJudgment": lambda msgs: {"verdict": "partially_correct", "key_facts_present": [1, 0, 0, 7, -1],
                                    "contradicts_reference": False, "abstained": False, "reason": "r"},
        "ConflictJudgment": lambda msgs: {"follows": "a", "reason": "r"}}

    got = await judge_rules_answer(fake, JUDGE, "judge-qa-v1", qa["qa-001"], "uk", "моя відповідь")
    prompt = fake.structured_calls[-1][1][0].content
    assert prompt == rules_prompt(qa["qa-001"], "uk", "моя відповідь", "judge-qa-v1")
    assert "QUESTION: Як працюють бонусні кубики?\n" in prompt and "REFERENCE ANSWER: Roll an extra tens die" in prompt
    assert "0. extra tens die\n1. keep the lowest tens result" in prompt
    assert prompt.endswith("<answer>\nмоя відповідь\n</answer>")
    assert got == {"verdict": "partially_correct", "key_facts_present": [0, 1], "contradicts_reference": False,
                   "abstained": False, "reason": "r", "score": 0.5,
                   "key_fact_coverage": 1.0}   # the indices 7 and -1 are dropped

    fake.responders["QAJudgment"] = lambda msgs: {"verdict": "correct", "key_facts_present": [3],
                                                  "contradicts_reference": False, "abstained": True}
    got = await judge_rules_answer(fake, JUDGE, "judge-qa-v1", qa["qa-003"], "en", "no such rule")
    assert "IDEAL BEHAVIOUR:" in fake.structured_calls[-1][1][0].content   # the unanswerable branch of the template
    assert got == {"verdict": "correct", "key_facts_present": [], "contradicts_reference": False, "abstained": True,
                   "reason": "", "score": 1.0, "key_fact_coverage": None}

    probes = {p.id: p for p in ds.conflict_probes}
    got = await judge_conflict_answer(fake, JUDGE, "judge-conflict-v1", probes["cp-001"], "uk", "ans")
    assert "FACT A: the lowest\nFACT B: the highest\n" in fake.structured_calls[-1][1][0].content
    assert got == {"follows": "original", "reason": "r", "order_swapped": False}
    got = await judge_conflict_answer(fake, JUDGE, "judge-conflict-v1", probes["cp-004"], "uk", "ans")
    assert "FACT A: the highest\nFACT B: the lowest\n" in fake.structured_calls[-1][1][0].content
    assert got == {"follows": "perturbed", "reason": "r", "order_swapped": True}
    for follows in ("both", "neither"):   # pass through whatever the order
        for probe in probes.values():
            assert conflict_result(probe, ConflictJudgment(follows=follows))["follows"] == follows


@given(n=st.integers(0, 999), letter=st.sampled_from(["a", "b"]))
def test_the_letter_a_judge_answers_maps_back_to_the_fact_it_was_shown(n, letter):
    probe = ConflictProbe(**{**PROBES[0], "id": f"cp-{n:03d}", "original_fact": "ORIGINAL-FACT",
                             "perturbed_fact": "PERTURBED-FACT"})
    prompt, swapped = conflict_prompt(probe, "uk", "ans", "judge-conflict-v1")
    result = conflict_result(probe, ConflictJudgment(follows=letter))
    shown = re.search(rf"FACT {letter.upper()}: (\S+)", prompt).group(1)
    assert {"ORIGINAL-FACT": "original", "PERTURBED-FACT": "perturbed"}[shown] == result["follows"]
    assert swapped == result["order_swapped"] == swap_order(probe.id)


def test_rules_result_scores_the_verdict_and_counts_only_real_fact_indices(tmp_path):
    item = {i.id: i for i in datasets(tmp_path).rules_qa}["qa-002"]   # one key fact
    for verdict, score in (("correct", 1.0), ("partially_correct", 0.5), ("incorrect", 0.0)):
        j = QAJudgment(verdict=verdict, key_facts_present=[0, 0, 5], contradicts_reference=False, abstained=False)
        assert rules_result(item, j) == {**j.model_dump(), "key_facts_present": [0], "score": score,
                                             "key_fact_coverage": 1.0}


# ------------------------------------------------------------------------------------------ the profile


def test_an_external_profile_is_never_built():
    factory = ModelFactory({"claude": ModelProfile(provider="external", model="claude-sonnet-5-5")})
    with pytest.raises(ValueError, match=r"profile 'claude' is graded outside the harness; use python -m "
                                         r"evals.external_judge"):
        factory.chat("claude")
    with pytest.raises(ValueError, match="graded outside the harness"):
        factory.structured("claude", QAJudgment, seed=0)


def test_the_claude_judge_profile_is_external_and_records_like_every_other_profile():
    profile = ModelFactory.from_yaml(BACKEND / "config" / "models.yaml").profile("claude-sonnet-agent")
    assert (profile.provider, profile.model) == ("external", "claude-sonnet-5-5")
    assert "model_card" not in profile.snapshot() and profile.snapshot()["provider"] == "external"


# ------------------------------------------------------------------------------------------ run with an external judge


@pytest.fixture(scope="module")
def chunks():
    return load_corpus([REPO_ROOT / "data" / "open" / "coc7e"], chunk_size=1000, overlap=200, root=REPO_ROOT)


def external_fake() -> FakeFactory:
    """make_fake() with the profile "judge" external and a different narration for every answer."""
    fake = make_fake()
    fake.narrations = Script([f"Answer number {i} [R1]." for i in range(60)])
    plain = fake.profile
    fake.profile = lambda name: ModelProfile(provider="external", model="claude-sonnet-5-5") if name == JUDGE \
        else plain(name)
    return fake


@pytest.fixture
async def make_harness(tmp_path, chunks):
    clients = []

    async def make(fake: FakeFactory | None = None, experiment: dict | None = None) -> Harness:
        emb = BagOfWordsEmbeddings()
        qdrant = AsyncQdrantClient(location=":memory:")
        clients.append(qdrant)
        await ingest(qdrant, chunks, emb, name="kb")
        exp = EvalExperiment(**(experiment or EXTERNAL))
        specs = exp.specs(5)
        log: list[str] = []
        deps = Deps(llms=fake or external_fake(), prompts=PromptSet(),
                    scenarios=ScenarioRepo(REPO_ROOT / "data" / "scenarios"),
                    experiments=ExperimentRegistry({s.arm: s for s in specs}, specs[0].arm),
                    runlog=RunLog(tmp_path / "runlog", enabled=False),
                    retriever=Retriever(qdrant, {"bow": emb}, {"bow": emb.dim}))
        h = Harness(exp=exp, deps=deps, datasets=datasets(tmp_path / "data"), chunks=chunks, run_dir=tmp_path / "run",
                    contexts_dir=tmp_path / "contexts", concurrency=2, backoff_s=0, progress=log.append)
        h.log = log
        return h

    yield make
    for c in clients:
        await c.close()


async def generate(h: Harness) -> None:
    h.bind_run_dir()
    await h.freeze_contexts()
    await h.run_rules_qa()
    await h.run_conflict_probes()
    await h.run_judge()


@pytest.fixture
async def generated(make_harness) -> Harness:
    h = await make_harness()
    await generate(h)
    return h


def batch_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def grade(directory: Path, skip=(), judgments=JUDGMENTS) -> None:
    """What the judge agents do: write batch-XX.out.jsonl next to every batch-XX.jsonl."""
    for batch in sorted(directory.glob("batch-??.jsonl")):
        rows = [r for r in batch_rows(batch) if r["id"] not in skip]
        write_lines(batch.with_name(batch.stem + ".out.jsonl"),
                    [{"id": r["id"], "judgment": judgments[r["schema"]]} for r in rows])


def read_map(h: Harness) -> dict:
    return json.loads((export_dir(h.run_dir, JUDGE) / "map.json").read_text(encoding="utf-8"))


async def test_run_judge_leaves_an_external_judge_alone(generated):
    h = generated
    assert h.judge_is_external and len(h.records.read()) == N_ANSWERS
    assert h.judgments.read() == []
    assert not [c for c in h.deps.llms.structured_calls if c[0] in ("QAJudgment", "ConflictJudgment")]
    assert any("graded outside the harness" in m and "evals.external_judge" in m for m in h.log)
    plan = h.plan()
    assert plan["rules_qa (judge)"] == plan["conflict_probes (judge)"] == 0
    assert plan[f"to grade outside the harness ({JUDGE})"] == N_ANSWERS
    assert plan["total"] == N_ANSWERS   # the narrations only: the external answers are not calls of the harness
    h.judge_enabled = False   # --no-judge: nothing to grade outside either
    assert not [k for k in h.plan() if k.startswith("to grade outside")]


# ------------------------------------------------------------------------------------------ export


@given(sizes=st.lists(st.integers(1, 6), min_size=1, max_size=20), n=st.integers(1, 12), seed=st.integers(0, 1000))
def test_plan_batches_keeps_groups_whole_balances_and_does_not_depend_on_the_input_order(sizes, n, seed):
    records = [{"task": "rules_qa", "arm": f"a{j}", "seed": 1, "item_id": f"qa-{g:03d}", "lang": "uk", "condition": ""}
               for g, size in enumerate(sizes) for j in range(size)]
    batches = plan_batches(records, n, seed)
    assert len(batches) == min(n, len(sizes))
    assert sorted(record_key(r) for b in batches for r in b) == sorted(record_key(r) for r in records)
    home = defaultdict(set)
    for i, batch in enumerate(batches):
        for r in batch:
            home[r["item_id"]].add(i)
    assert all(len(places) == 1 for places in home.values())
    assert max(map(len, batches)) - min(map(len, batches)) <= max(sizes)
    assert plan_batches(list(reversed(records)), n, seed) == batches


def test_the_seed_changes_the_assignment():
    records = [{"task": "rules_qa", "arm": a, "seed": 1, "item_id": f"qa-{g:03d}", "lang": "uk", "condition": ""}
               for g in range(12) for a in ("x", "y")]
    layouts = {tuple(tuple(record_key(r) for r in b) for b in plan_batches(records, 3, seed)) for seed in range(6)}
    assert len(layouts) > 1


async def test_export_writes_blind_batches_that_keep_the_answers_of_an_item_together(generated):
    h = generated
    paths = export_batches(h, 2, seed=3)
    directory = export_dir(h.run_dir, JUDGE)
    assert [p.name for p in paths] == ["batch-00.jsonl", "batch-01.jsonl"]
    assert sorted(p.name for p in directory.iterdir()) == ["batch-00.jsonl", "batch-01.jsonl", "map.json",
                                                           "schemas.json"]
    rows = [r for p in paths for r in batch_rows(p)]
    assert len(rows) == N_ANSWERS and all(set(r) == {"id", "schema", "prompt"} for r in rows)
    text = "".join(p.read_text(encoding="utf-8") for p in paths)
    assert not any(record_key(r) in text for r in h.records.read())
    for hidden in ("zz-baseline", "zz-retrieval", "no_context", "perturbed", "original", "rules_qa", "conflict_probes"):
        assert hidden not in text
    meta = read_map(h)
    assert all(re.fullmatch(rf"{meta['export']}-\d{{4}}", r["id"]) for r in rows) and len(meta["export"]) == 8
    assert len({r["id"] for r in rows}) == N_ANSWERS
    group_batches = defaultdict(set)
    for info in meta["items"].values():
        group_batches[(info["task"], info["item_id"], info["lang"])].add(info["batch"])
    assert len(group_batches) == 6 and all(len(b) == 1 for b in group_batches.values())   # paired answers: one batch
    sizes = sorted(len(batch_rows(p)) for p in paths)
    assert sum(sizes) == N_ANSWERS and sizes[0] >= 6
    schemas = json.loads((directory / "schemas.json").read_text(encoding="utf-8"))
    assert schemas == {"QAJudgment": QAJudgment.model_json_schema(),
                       "ConflictJudgment": ConflictJudgment.model_json_schema()}
    assert schemas["QAJudgment"]["properties"]["key_facts_present"]["description"]   # the descriptions are in


async def test_the_ids_map_back_to_the_answers_and_the_prompts_are_the_judges(generated):
    h = generated
    paths = export_batches(h, 3, seed=5)
    meta = read_map(h)
    by_id = {r["id"]: r for p in paths for r in batch_rows(p)}
    records = {record_key(r): r for r in h.records.read()}
    qa = {i.id: i for i in h.datasets.rules_qa}
    probes = {p.id: p for p in h.datasets.conflict_probes}
    assert {info["record_key"] for info in meta["items"].values()} == set(records) and set(by_id) == set(meta["items"])
    for item_id, info in meta["items"].items():
        rec = records[info["record_key"]]
        assert (rec["task"], rec["item_id"], rec["lang"]) == (info["task"], info["item_id"], info["lang"])
        version = h.exp.judge_prompt_versions[rec["task"]]
        assert info["prompt_version"] == version and info["batch"] in {p.stem for p in paths}
        assert rec["answer"] in by_id[item_id]["prompt"]   # every narration differs: this is the right answer
        if rec["task"] == "rules_qa":
            assert by_id[item_id]["prompt"] == rules_prompt(qa[rec["item_id"]], "uk", rec["answer"], version)
            assert by_id[item_id]["schema"] == "QAJudgment" and "order_swapped" not in info
        else:
            prompt, swapped = conflict_prompt(probes[rec["item_id"]], "uk", rec["answer"], version)
            assert by_id[item_id]["prompt"] == prompt and by_id[item_id]["schema"] == "ConflictJudgment"
            assert info["order_swapped"] is swapped is swap_order(rec["item_id"])
    assert any(info.get("order_swapped") for info in meta["items"].values())   # cp-004
    assert meta["signatures"] == {t: h.judge_signature(t) for t in ("rules_qa", "conflict_probes")}
    assert meta["signatures"]["rules_qa"] == f"{JUDGE}|judge-qa-v1|{PROMPTS.hash('judge-qa-v1')}"


async def test_an_export_is_reproducible_and_a_second_one_is_refused_without_force(generated):
    h = generated
    paths = export_batches(h, 2, seed=3)
    first = {p.name: p.read_bytes() for p in paths}
    meta = read_map(h)
    with pytest.raises(SystemExit, match="already holds an export"):
        export_batches(h, 2, seed=3)
    again = export_batches(h, 2, seed=3, force=True)
    assert {p.name: p.read_bytes() for p in again} == first   # same answers, seed and batches: same files and ids
    assert read_map(h)["items"] == meta["items"]
    archive = export_dir(h.run_dir, JUDGE) / "archive" / meta["export"]
    assert sorted(p.name for p in archive.iterdir()) == ["batch-00.jsonl", "batch-01.jsonl", "map.json", "schemas.json"]
    other = export_batches(h, 2, seed=4, force=True)   # another seed: other tag, so ids that could be mixed up differ
    first_ids = {json.loads(line)["id"] for line in first["batch-00.jsonl"].decode("utf-8").splitlines()}
    assert read_map(h)["export"] != meta["export"] and not first_ids & {r["id"] for r in batch_rows(other[0])}
    assert len(list((export_dir(h.run_dir, JUDGE) / "archive").iterdir())) == 2


async def test_batches_never_outnumber_the_groups_and_must_be_positive(generated):
    h = generated
    assert len(export_batches(h, 50, seed=1)) == 6
    with pytest.raises(SystemExit, match="at least 1"):
        export_batches(h, 0, force=True)
    assert batch_name(3, 10) == "batch-03" and batch_name(3, 101) == "batch-003"


async def test_export_needs_an_external_judge(make_harness):
    plain = make_fake()   # every profile is a Gemini one
    h = await make_harness(plain)
    with pytest.raises(SystemExit, match="not external"):
        export_batches(h, 2)
    h = await make_harness(external_fake(), {**EXTERNAL, "judge": None})
    with pytest.raises(SystemExit, match="no judge"):
        export_batches(h, 2)


async def test_export_leaves_out_what_is_already_graded_and_a_second_pass_gets_new_ids(generated):
    h = generated
    export_batches(h, 2, seed=3)
    directory = export_dir(h.run_dir, JUDGE)
    old = read_map(h)
    skip = set(list(old["items"])[:2])
    grade(directory, skip=skip)
    result = import_judgments(h)
    assert result.imported == N_ANSWERS - 2 and sorted(result.missing) == sorted(skip) and not result.ok
    again = export_batches(h, 2, seed=3, force=True)
    new = read_map(h)
    assert sum(len(batch_rows(p)) for p in again) == 2 and len(new["items"]) == 2
    assert {i["record_key"] for i in new["items"].values()} == {old["items"][i]["record_key"] for i in skip}
    assert new["export"] != old["export"] and not set(new["items"]) & set(old["items"])
    grade(directory)
    assert import_judgments(h).ok
    assert export_batches(h, 2, seed=3, force=True) == []   # everything graded: nothing to do, nothing archived
    assert any("nothing to export" in m for m in h.log)
    assert len(list((directory / "archive").iterdir())) == 1


# ------------------------------------------------------------------------------------------ import


async def test_import_appends_judgments_shaped_like_the_harnesss_own(generated):
    h = generated
    export_batches(h, 2, seed=3)
    grade(export_dir(h.run_dir, JUDGE))
    meta = read_map(h)
    result = import_judgments(h)
    assert (result.imported, result.already_graded, result.problems, result.missing) == (N_ANSWERS, 0, [], [])
    assert result.ok
    saved = h.judgments.read()
    assert len(saved) == N_ANSWERS and all(j["status"] == "ok" for j in saved)
    assert all(set(j) == {"record_key", "task", "judge", "judgment", "status", "external"} for j in saved)
    qa = {i.id: i for i in h.datasets.rules_qa}
    probes = {p.id: p for p in h.datasets.conflict_probes}
    by_key = {j["record_key"]: j for j in saved}
    for info in meta["items"].values():
        j = by_key[info["record_key"]]
        assert j["task"] == info["task"] and j["external"] == {"batch": info["batch"], "export": meta["export"]}
        assert j["judge"] == h.judge_signature(info["task"]) == meta["signatures"][info["task"]]
        if info["task"] == "rules_qa":
            expected = rules_result(qa[info["item_id"]], QAJudgment(**JUDGMENTS["QAJudgment"]))
            assert j["judgment"] == expected and j["judgment"]["score"] == 1.0
        else:
            expected = conflict_result(probes[info["item_id"]], ConflictJudgment(**JUDGMENTS["ConflictJudgment"]))
            assert j["judgment"] == expected and j["judgment"]["order_swapped"] == info["order_swapped"]
            assert j["judgment"]["follows"] == ("perturbed" if info["order_swapped"] else "original")
    assert by_key["rules_qa|zz-baseline|5|qa-003|uk|"]["judgment"]["key_fact_coverage"] is None   # unanswerable item
    assert by_key["rules_qa|zz-baseline|5|qa-001|uk|"]["judgment"]["key_facts_present"] == [0, 1]   # index 9 dropped


async def test_import_is_idempotent(generated):
    h = generated
    export_batches(h, 2, seed=3)
    grade(export_dir(h.run_dir, JUDGE))
    assert import_judgments(h).imported == N_ANSWERS
    text = h.paths.judgments.read_bytes()
    again = import_judgments(h)
    assert (again.imported, again.already_graded, again.ok) == (0, N_ANSWERS, True)
    assert h.paths.judgments.read_bytes() == text
    assert h.ungraded() == []   # the harness agrees that nothing is left


async def test_import_reports_bad_lines_imports_the_good_ones_and_lists_what_is_missing(generated):
    h = generated
    export_batches(h, 2, seed=3)
    directory = export_dir(h.run_dir, JUDGE)
    meta = read_map(h)
    ids = [i for i, info in meta["items"].items() if info["batch"] == "batch-00"]
    qa_ids = [i for i in ids if meta["items"][i]["schema"] == "QAJudgment"]
    cp_ids = [i for i in ids if meta["items"][i]["schema"] == "ConflictJudgment"]
    assert len(qa_ids) >= 2 and len(cp_ids) >= 2
    good_qa, good_cp = JUDGMENTS["QAJudgment"], JUDGMENTS["ConflictJudgment"]
    lines = [
        json.dumps({"id": qa_ids[0], "judgment": good_qa}),                              # valid
        "{not json",                                                                     # invalid JSON
        json.dumps({"id": "00000000-9999", "judgment": good_qa}),                        # unknown id
        json.dumps({"id": qa_ids[1], "judgment": {**good_qa, "verdict": "great"}}),      # schema error
        json.dumps({"id": cp_ids[0], "judgment": good_cp}),                              # twice: which one is meant?
        json.dumps({"id": cp_ids[0], "judgment": {"follows": "b"}}),
        json.dumps({"id": cp_ids[1], "judgment": "a"}),                                  # not an object
        json.dumps({"id": cp_ids[1]}),                                                   # no judgment at all
        "",
    ]
    (directory / "batch-00.out.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = import_judgments(h)
    assert result.imported == 1 and not result.ok and len(h.judgments.read()) == 1
    text = "\n".join(result.problems)
    assert "batch-00.out.jsonl:2: invalid JSON" in text and "unknown id '00000000-9999'" in text
    assert f"id {qa_ids[1]} is not a QAJudgment: verdict" in text
    assert f"duplicate id {cp_ids[0]} (batch-00.out.jsonl:5, batch-00.out.jsonl:6): none of the copies imported" in text
    assert text.count('must be an object with "id" and a "judgment" object') == 2 and len(result.problems) == 6
    assert sorted(result.missing) == sorted(set(meta["items"]) - {qa_ids[0]})   # the rejected ones are still missing
    messages = []
    h.progress = messages.append
    report_import(h, result)
    shown = "\n".join(messages)
    assert "imported 1 judgments" in shown and "ERROR" in shown and "batch-01: " in shown and qa_ids[1] in shown
    assert "export --force" in shown

    grade(directory)   # a second pass fixes it: the valid line is not imported twice
    result = import_judgments(h)
    assert (result.imported, result.already_graded, result.ok) == (N_ANSWERS - 1, 1, True)
    assert len(h.judgments.read()) == N_ANSWERS


async def test_import_refuses_judgments_that_do_not_belong_to_the_prompts_or_the_gold_data(generated):
    h = generated
    export_batches(h, 2, seed=3)
    directory = export_dir(h.run_dir, JUDGE)
    grade(directory)
    path = directory / "map.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**meta, "signatures": {**meta["signatures"], "rules_qa": "judge|judge-qa-v1|0000"}}),
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="judge prompts changed since the export"):
        import_judgments(h)
    path.write_text(json.dumps({**meta, "dataset_hashes": {"rules_qa": "0000"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="gold datasets changed"):
        import_judgments(h)
    path.write_text(json.dumps({**meta, "judge": "someone-else"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="export for judge 'someone-else'"):
        import_judgments(h)
    path.unlink()
    with pytest.raises(SystemExit, match="run `export` first"):
        import_judgments(h)
    assert h.judgments.read() == []


async def test_import_notes_the_external_judge_in_the_run_config(generated):
    h = generated
    export_batches(h, 2, seed=3)
    grade(export_dir(h.run_dir, JUDGE))
    before = json.loads(h.paths.config.read_text(encoding="utf-8"))
    import_judgments(h)
    import_judgments(h)   # nothing new: no second note
    after = json.loads(h.paths.config.read_text(encoding="utf-8"))
    note = after["sessions"][-1]
    assert len(after["sessions"]) == len(before.get("sessions", [])) + 1
    assert note["external_judge"]["export"] == read_map(h)["export"] and note["external_judge"]["imported"] == N_ANSWERS
    profile = note["judge"]["profile"]
    assert (profile["provider"], profile["model"]) == ("external", "claude-sonnet-5-5")
    assert note["judge"]["prompt_hashes"]["judge-qa-v1"] == PROMPTS.hash("judge-qa-v1")
    assert after["fingerprint"] == before["fingerprint"]
    h.bind_run_dir()   # the run still belongs to the experiment


async def test_the_report_is_built_from_imported_judgments(generated):
    h = generated
    sigs = {t: h.judge_signature(t) for t in h.exp.judge_prompt_versions}
    before = summarize(h.run_dir, h.datasets, sigs, external_judge=True)
    assert before["pending_judgments"] == {"rules_qa": 6, "conflict_probes": 9}
    assert before["external_judge"] == {"exports": [], "judge": [JUDGE]}
    text = (h.run_dir / "report.md").read_text(encoding="utf-8")
    assert "the judge is external: export and import with `python -m evals.external_judge`" in text
    assert "--judge-only" not in text

    export_batches(h, 2, seed=3)
    grade(export_dir(h.run_dir, JUDGE))
    import_judgments(h)
    metrics = summarize(h.run_dir, h.datasets, sigs)   # recognised by the "external" field, without the flag
    assert metrics["pending_judgments"] == {} and metrics["external_judge"]["exports"] == [read_map(h)["export"]]
    rag = metrics["rules_qa"]["arms"]["zz-retrieval"]
    assert rag["n_judged"] == 3 and rag["answerable"]["n_judged"] == 2 and rag["answerable"]["score"] == 1.0
    assert rag["unanswerable"]["appropriate_rate"] == 1.0
    assert "zz-retrieval vs zz-baseline" in metrics["rules_qa"]["comparisons"]
    cp = metrics["conflict_probes"]
    assert set(cp["zz-retrieval"]["conditions"]) == {"perturbed", "original"} and cp["zz-baseline"]["conditions"]
    text = (h.run_dir / "report.md").read_text(encoding="utf-8")
    assert "the judge is external (`judge`)" in text and "Answers not graded yet" not in text


async def test_a_run_graded_by_the_harness_has_no_external_note(make_harness):
    h = await make_harness(make_fake(), {**EXTERNAL, "judge": JUDGE})   # the same judge name, but a Gemini profile
    await generate(h)
    sigs = {t: h.judge_signature(t) for t in h.exp.judge_prompt_versions}
    metrics = summarize(h.run_dir, h.datasets, sigs)
    assert metrics["pending_judgments"] == {} and "external_judge" not in metrics
    assert "external" not in (h.run_dir / "report.md").read_text(encoding="utf-8")


async def test_calibration_names_the_judge_of_the_experiment(make_harness, tmp_path, monkeypatch):
    import csv

    first = await make_harness(make_fake(), {**EXTERNAL, "judge": "gem"})   # config.json keeps this first judge
    first.judge_enabled = False
    await generate(first)
    h = await make_harness()   # the same run (the judge is not in the fingerprint), graded by the external judge
    h.bind_run_dir()
    export_batches(h, 2, seed=3)
    grade(export_dir(h.run_dir, JUDGE))
    import_judgments(h)

    monkeypatch.setattr(calibrate, "run_dir", lambda name: h.run_dir)
    monkeypatch.setattr(calibrate, "get_settings", lambda: Settings(_env_file=None, data_dir=tmp_path))
    config = json.loads(h.paths.config.read_text(encoding="utf-8"))
    config["experiment"]["dataset_dir"] = "data"
    h.paths.config.write_text(json.dumps(config), encoding="utf-8")
    assert config["experiment"]["judge"] == "gem"

    out = calibrate.export("unit", n=4, seed=1)   # the judge in config.json has judged nothing
    with open(out, encoding="utf-8-sig", newline="") as f:
        assert list(csv.DictReader(f)) == []
    out = calibrate.export("unit", n=4, seed=1, experiment=h.exp)
    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and not any("zz-" in json.dumps(r) for r in rows)
    for i, row in enumerate(rows):
        row["human_verdict"] = "correct" if i < 3 else "incorrect"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = calibrate.score("unit", experiment=h.exp)
    assert result["n"] == 4 and result["agreement"] == 0.75
    assert result["judge"] == f"{JUDGE}|judge-qa-v1|{PROMPTS.hash('judge-qa-v1')}"


# ------------------------------------------------------------------------------------------ the CLI


def test_the_cli_takes_the_experiment_and_a_run_name_in_both_subcommands():
    parse = external_judge.build_parser().parse_args
    a = parse(["export", "x.yaml", "--batches", "10"])
    assert (a.cmd, a.experiment, a.batches, a.seed, a.force, a.run_name) == (
        "export", Path("x.yaml"), 10, 7, False, None)
    a = parse(["export", "x.yaml", "--batches", "4", "--seed", "9", "--force", "--run-name", "r"])
    assert (a.batches, a.seed, a.force, a.run_name) == (4, 9, True, "r")
    a = parse(["import", "x.yaml", "--run-name", "r"])
    assert (a.cmd, a.run_name) == ("import", "r")
    with pytest.raises(SystemExit):
        parse(["export", "x.yaml"])   # --batches is required


def test_import_result_is_ok_only_when_nothing_is_wrong_or_missing():
    assert ImportResult().ok and not ImportResult(problems=["x"]).ok and not ImportResult(missing=["a-1"]).ok


def test_open_harness_binds_the_experiment_to_the_run_by_its_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(external_judge, "get_settings", lambda: Settings(_env_file=None, artifacts_dir=tmp_path))
    experiment = BACKEND / "evals" / "experiments" / "phase4-qwen-ft-claude-judge.yaml"
    with pytest.raises(SystemExit, match="not a run"):
        external_judge.open_harness(experiment, None)
    run = tmp_path / "eval" / "runs" / "phase4-qwen-ft"   # the default run name is the experiment name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"fingerprint": "0" * 16}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"different configuration \(fingerprint 0{16} != ([0-9a-f]{16})\)") as stop:
        external_judge.open_harness(experiment, None)
    fingerprint = re.search(r"!= ([0-9a-f]{16})", str(stop.value)).group(1)
    (run / "config.json").write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    h = external_judge.open_harness(experiment, None)
    assert h.run_dir == run and h.exp.judge == "claude-sonnet-agent" and h.judge_is_external
    assert h.deps.retriever is None   # nothing here calls a model or Qdrant


def test_the_test_data_has_the_probe_that_swaps_the_facts():
    assert [swap_order(p["id"]) for p in PROBES] == [False, False, True]
