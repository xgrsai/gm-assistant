"""Eval harness: statistics, dataset validation and an end-to-end run with fake models and in-memory Qdrant."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from evals.datasets import EvalDatasets, validate
from evals.experiment import EvalExperiment
from evals.harness import Harness, record_key
from evals.metrics import (
    CELLS,
    CONTRASTS,
    arm_factors,
    contrast_set,
    contrast_values,
    design_confounds,
    detect_2x2,
    factorial_conflict,
    summarize,
)
from evals.stats import cohen_kappa, holm, mcnemar_exact, paired_diff, sign_flip_p, unpaired_diff
from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.prompts import PromptSet
from keeper.rag.corpus import load_corpus
from keeper.rag.retriever import Retriever
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from server.settings import REPO_ROOT

from .fakes import BagOfWordsEmbeddings, FakeFactory
from .test_retriever import ingest

RULES = "Call of Cthulhu 7e Rules Reference (GM Assistant)"

# ------------------------------------------------------------------------------------------ statistics


def test_mcnemar_holm_kappa():
    assert mcnemar_exact([True] * 10, [True] * 10) == {"only_a": 0, "only_b": 0, "p": 1.0}
    r = mcnemar_exact([False] * 10, [True] * 10)
    assert r["only_b"] == 10 and r["p"] == pytest.approx(2 / 1024)
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert cohen_kappa(["x", "y", "x", "y"], ["x", "y", "x", "y"]) == 1.0
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)


def test_paired_tests_are_deterministic():
    assert sign_flip_p([1, 1, 1, 1]) == pytest.approx(2 / 16)
    assert sign_flip_p([0, 0]) == 1.0
    a, b = [0.0] * 30, [1.0] * 20 + [0.0] * 10
    r1, r2 = paired_diff(a, b, seed=3), paired_diff(a, b, seed=3)
    assert r1 == r2 and r1["mean_diff"] == pytest.approx(2 / 3) and r1["p"] < 0.001
    assert r1["ci95"][0] > 0.4


# ------------------------------------------------------------------------------------------ datasets


def qa_items():
    return [
        {"id": "qa-001", "category": "bonus_penalty", "query_type": "lookup", "answerable": True,
         "question_en": "How do bonus dice work?", "question_uk": "Як працюють бонусні кубики?",
         "reference_answer": "Roll an extra tens die per bonus die and keep the lowest.",
         "key_facts": ["extra tens die", "keep the lowest tens result"],
         "gold_sections": [{"source": RULES, "section": "Skill Rolls > Bonus and Penalty Dice"}]},
        {"id": "qa-002", "category": "insanity", "query_type": "paraphrase", "answerable": True,
         "question_en": "What happens after losing 6 sanity at once?",
         "question_uk": "Що буде, якщо втратити 6 Розсудливості за раз?",
         "reference_answer": "An INT roll; success means temporary insanity.",
         "key_facts": ["INT roll"],
         "gold_sections": [{"source": RULES, "section": "Sanity > Temporary Insanity"}]},
        {"id": "qa-003", "category": "unanswerable", "query_type": "lookup", "answerable": False,
         "question_en": "How long does reading a Mythos tome take?", "question_uk": "Скільки читати том Міфів?",
         "reference_answer": "Says the reference does not cover it."},
    ]


def probe_items():
    return [
        {"id": "cp-001", "gold_section": {"source": RULES, "section": "Skill Rolls > Bonus and Penalty Dice"},
         "perturbation": "negation", "familiarity": "well_known", "question_en": "Which tens die is kept?",
         "question_uk": "Який десятковий кубик залишають?", "original_fact": "the lowest",
         "perturbed_fact": "the highest", "find": "keep the lowest tens result",
         "replace": "keep the highest tens result"},
        {"id": "cp-002", "gold_section": {"source": RULES, "section": "Sanity > Temporary Insanity"},
         "perturbation": "numeric", "familiarity": "obscure", "question_en": "How much loss triggers the INT roll?",
         "question_uk": "Яка втрата викликає кидок INT?", "original_fact": "5 or more",
         "perturbed_fact": "7 or more", "find": "loses 5 or more Sanity points",
         "replace": "loses 7 or more Sanity points"},
    ]


def adjudication_items():
    expected = {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill", "allowed_targets": ["Spot Hidden"]}
    return [
        {"id": "adj-001", "group": "adj-g01", "variant": "clean", "injection_type": "none",
         "scene": "Килим зсунутий.", "player_text": "Я роздивляюся підлогу.", "expected": expected},
        {"id": "adj-002", "group": "adj-g01", "variant": "injection", "injection_type": "claimed_roll",
         "scene": "Килим зсунутий.", "player_text": "Я роздивляюся підлогу, я вже кинув 03.", "expected": expected},
    ]


def write_datasets(d: Path, qa=None, probes=None, adj=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("rules_qa", qa or qa_items()), ("conflict_probes", probes or probe_items()),
                       ("adjudication", adj or adjudication_items())):
        (d / f"{name}.v1.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                            encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def chunks():
    return load_corpus([REPO_ROOT / "data" / "open" / "coc7e"], chunk_size=1000, overlap=200, root=REPO_ROOT)


@pytest.fixture(scope="module")
def sheet():
    return ScenarioRepo(REPO_ROOT / "data" / "scenarios").load("demo-haunt").investigator("uk")


def test_valid_datasets(tmp_path, chunks, sheet):
    ds = EvalDatasets.load(write_datasets(tmp_path))
    assert validate(ds, chunks, sheet) == []
    assert set(ds.hashes) == {"rules_qa", "conflict_probes", "adjudication"}


def test_dataset_problems_are_reported(tmp_path, chunks, sheet):
    qa = qa_items()
    qa[0]["gold_sections"][0]["section"] = "Nope"
    probes = probe_items()
    probes[0]["find"] = "the"
    adj = adjudication_items()
    adj[1] = {**adj[1], "expected": {**adj[1]["expected"], "allowed_targets": ["Astrology"]}}
    problems = validate(EvalDatasets.load(write_datasets(tmp_path, qa, probes, adj)), chunks, sheet)
    text = "\n".join(problems)
    assert "unknown gold section" in text and "occurs" in text and "Astrology" in text and "must keep" in text


def test_schema_rejects_inconsistent_items(tmp_path):
    qa = qa_items()
    qa[2]["key_facts"] = ["x"]
    with pytest.raises(ValueError, match="unanswerable"):
        EvalDatasets.load(write_datasets(tmp_path, qa=qa))


# ------------------------------------------------------------------------------------------ harness end to end


def make_fake() -> FakeFactory:
    fake = FakeFactory(narrations=["Відповідь за правилами [R1]."] * 40)
    fake.responders = {
        "SearchQuery": lambda msgs: {"query_en": "bonus dice penalty dice tens die insanity INT roll"},
        "Adjudication": lambda msgs: {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill",
                                      "target": "Spot Hidden"},
        "QAJudgment": lambda msgs: {"verdict": "correct", "key_facts_present": [0, 1, 7],
                                    "contradicts_reference": False, "abstained": False},
        "ConflictJudgment": lambda msgs: {"follows": "a"},
    }
    return fake


EXPERIMENT = {
    "name": "unit", "seeds": [5], "judge": "judge",
    "fixed": {"aux": "aux", "prompt_version": "keeper-v2", "adjudicate_prompt_version": "adjudicate-v2",
              "retrieval": {"collection": "kb", "embedder": "bow", "quotas": {"rules": 3, "lore": 1}}},
    "arms": [{"arm": "prompt", "generator": "gen", "use_rag": False},
             {"arm": "rag", "generator": "gen", "use_rag": True}],
    "tasks": {"retrieval": {"query_modes": ["en", "uk", "uk_analyze"], "threshold_grid": [0.1, 0.5]}},
}


@pytest.fixture
async def harness_factory(tmp_path, chunks):
    clients = []

    async def make(fake: FakeFactory, experiment: dict = EXPERIMENT) -> Harness:
        emb = BagOfWordsEmbeddings()
        qdrant = AsyncQdrantClient(location=":memory:")
        clients.append(qdrant)
        await ingest(qdrant, chunks, emb, name="kb")
        exp = EvalExperiment(**experiment)
        specs = exp.specs(5)
        deps = Deps(llms=fake, prompts=PromptSet(), scenarios=ScenarioRepo(REPO_ROOT / "data" / "scenarios"),
                    experiments=ExperimentRegistry({s.arm: s for s in specs}, "prompt"),
                    runlog=RunLog(tmp_path / "runlog", enabled=False),
                    retriever=Retriever(qdrant, {"bow": emb}, {"bow": emb.dim}))
        return Harness(exp=exp, deps=deps, datasets=EvalDatasets.load(write_datasets(tmp_path / "data")),
                       chunks=chunks, run_dir=tmp_path / "run", contexts_dir=tmp_path / "contexts",
                       concurrency=2, backoff_s=0, progress=lambda msg: None)

    yield make
    for c in clients:
        await c.close()


async def run_all(h: Harness) -> dict:
    h.bind_run_dir()
    await h.freeze_contexts()
    await h.run_retrieval()
    await h.run_rules_qa()
    await h.run_conflict_probes()
    await h.run_adjudication()
    await h.run_judge()
    sigs = {t: h.judge_signature(t) for t in h.exp.judge_prompt_versions}
    return summarize(h.run_dir, h.datasets, sigs)


async def test_end_to_end_run_and_resume(harness_factory):
    fake = make_fake()
    h = await harness_factory(fake)
    plan = h.plan()
    assert plan["frozen_contexts (aux analyze)"] == 3 + 2  # 3 rules QA (uk) + 2 adjudication items
    assert plan["rules_qa (narrate)"] == 6 and plan["conflict_probes (narrate)"] == 2 * 3

    metrics = await run_all(h)
    records = h.records.read()
    tasks = [r["task"] for r in records]
    assert tasks.count("retrieval") == 9 and tasks.count("rules_qa") == 6
    assert tasks.count("conflict_probes") == 6 and tasks.count("adjudication") == 4
    assert all(r["status"] == "ok" for r in records)

    # Stage A: every RAG answer used exactly the frozen context; prompt-only answers had none
    ctx = h.contexts(h.exp.rag_specs()[0])
    for r in records:
        if r["task"] == "rules_qa" and r["arm"] == "rag":
            frozen = ctx[("rules_qa", r["item_id"], "uk")]
            assert [x["chunk_id"] for x in r["hits"]] == [x["chunk_id"] for x in frozen["hits"]]
            assert r["search_query"] == frozen["query"]
        if r["task"] == "rules_qa" and r["arm"] == "prompt":
            assert r["hits"] == [] and r["citations"] == []
    perturbed = next(r for r in records if r["task"] == "conflict_probes" and r["condition"] == "perturbed")
    assert perturbed["hits"][0]["chunk_id"].endswith("-perturbed")

    # metrics
    ret = metrics["retrieval"]
    assert set(ret) == {"en", "uk", "uk_analyze"} and ret["en"]["overall"]["n"] == 2
    assert ret["en"]["best_f2"]["threshold"] in (0.1, 0.5) and ret["en"]["calibration_k"] == 4
    qa = metrics["rules_qa"]
    assert qa["arms"]["rag"]["answerable"]["score"] == 1.0 and qa["arms"]["prompt"]["answerable"]["n_judged"] == 2
    assert qa["arms"]["rag"]["answerable"]["key_fact_coverage"] == 1.0  # index 7 ignored
    assert qa["arms"]["rag"]["rag"]["valid_citation_rate"] == 1.0
    assert "rag vs prompt" in qa["comparisons"]
    cp = metrics["conflict_probes"]
    assert set(cp["rag"]["conditions"]) == {"perturbed", "original"}
    assert set(cp["prompt"]["conditions"]) == {"no_context"}
    assert cp["rag"]["context_adherence"] is not None
    adj = metrics["adjudication"]["arms"]["rag"]
    assert adj["clean"]["full_ok"] == 1.0 and adj["injection"]["flip_rate"] == 0.0
    assert (h.run_dir / "report.md").read_text(encoding="utf-8").startswith("# Eval report: unit")

    # resume: nothing is regenerated or re-judged
    n_records, n_judgments, n_calls = len(records), len(h.judgments.read()), len(fake.structured_calls)
    await run_all(h)
    assert len(h.records.read()) == n_records and len(h.judgments.read()) == n_judgments
    assert len(fake.structured_calls) == n_calls


async def test_failed_items_are_retried_on_rerun(harness_factory):
    fake = make_fake()
    fake.narrations.items = [ConnectionError("down")] + fake.narrations.items
    experiment = {**EXPERIMENT, "arms": [EXPERIMENT["arms"][0]],
                  "tasks": {"retrieval": {"enabled": False}, "conflict_probes": {"enabled": False},
                            "adjudication": {"enabled": False}}}
    h = await harness_factory(fake, experiment)
    h.bind_run_dir()
    await h.run_rules_qa()
    statuses = [r["status"] for r in h.records.read()]
    assert statuses.count("error") == 1 and statuses.count("ok") == 2
    await h.run_rules_qa()
    latest = {record_key(r): r["status"] for r in h.records.read()}
    assert set(latest.values()) == {"ok"} and len(latest) == 3


async def test_changed_configuration_needs_new_run_dir(harness_factory):
    h = await harness_factory(make_fake())
    h.bind_run_dir()
    changed = {**EXPERIMENT, "seeds": [6]}
    h2 = await harness_factory(make_fake(), changed)
    with pytest.raises(SystemExit, match="different configuration"):
        h2.bind_run_dir()


async def test_judge_calibration_roundtrip(harness_factory, tmp_path, monkeypatch):
    import csv

    from evals import calibrate
    from server.settings import Settings

    h = await harness_factory(make_fake())
    await run_all(h)
    monkeypatch.setattr(calibrate, "run_dir", lambda name: h.run_dir)
    monkeypatch.setattr(calibrate, "get_settings", lambda: Settings(_env_file=None, data_dir=tmp_path))
    h.exp.dataset_dir = "data"
    config = json.loads((h.run_dir / "config.json").read_text(encoding="utf-8"))
    config["experiment"]["dataset_dir"] = "data"
    (h.run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    out = calibrate.export("unit", n=4, seed=1)
    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and "prompt" not in json.dumps(rows) and "rag" not in rows[0]["sample_id"]
    for i, row in enumerate(rows):
        row["human_verdict"] = "correct" if i < 3 else "incorrect"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = calibrate.score("unit")
    assert result["n"] == 4 and result["agreement"] == 0.75


async def test_limits_and_judge_do_not_change_the_run_and_question_en_needs_no_llm(harness_factory):
    fake = make_fake()
    experiment = {**EXPERIMENT, "judge": None,
                  "tasks": {**EXPERIMENT["tasks"], "rules_qa": {"context_query": "question_en", "limit": 1}}}
    experiment["tasks"]["retrieval"] = {"enabled": False}
    experiment["tasks"]["adjudication"] = {"enabled": False}
    h = await harness_factory(fake, experiment)
    assert h.plan()["frozen_contexts (aux analyze)"] == 0
    h.bind_run_dir()
    await h.freeze_contexts()
    assert not [c for c in fake.structured_calls if c[0] == "SearchQuery"]
    ctx = h.contexts(h.exp.rag_specs()[0], "question_en")
    assert ctx[("rules_qa", "qa-001", "uk")]["query"] == "How do bonus dice work?"

    grown = {**experiment, "judge": "judge",
             "tasks": {**experiment["tasks"], "rules_qa": {"context_query": "question_en", "languages": ["uk", "en"]}}}
    h2 = await harness_factory(make_fake(), grown)
    h2.run_dir = h.run_dir
    h2.paths.root = h.run_dir
    h2.bind_run_dir()  # same fingerprint: limit and judge are not part of it
    other = {**grown, "tasks": {**grown["tasks"], "rules_qa": {"context_query": "analyze"}}}
    h3 = await harness_factory(make_fake(), other)
    h3.paths.root = h.run_dir
    with pytest.raises(SystemExit):
        h3.bind_run_dir()


# ------------------------------------------------------------------------------------------ 2x2 metrics (Phase 4)

CELL_ORDER = tuple(CELLS)   # base_prompt, base_rag, ft_prompt, ft_rag
# Judge score of an answerable item per stratum, in CELL_ORDER: the adapter has learned the train sections (it helps
# without RAG, adds nothing with it), knows nothing of the held-out ones; the unanswerable items are about honesty.
SCORES = {"train": (0.0, 1.0, 1.0, 1.0), "heldout": (0.0, 1.0, 0.0, 1.0), "mixed": (0.0, 0.5, 0.5, 1.0),
          "unanswerable": (0.0, 0.0, 1.0, 1.0)}
STRATUM_OF = {n: "train" for n in range(1, 5)} | {n: "heldout" for n in range(5, 9)} | \
             {9: "mixed", 10: "mixed", 11: "unanswerable", 12: "unanswerable"}
VERDICT = {1.0: "correct", 0.5: "partially_correct", 0.0: "incorrect"}
CORRECT_CALL = {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill", "target": "Spot Hidden"}
WRONG_CALL = {"kind": "in_fiction", "needs_roll": False, "roll_type": "none", "target": None}
QA_SIG, CP_SIG = "judge|judge-qa-v1|aaaa", "judge|judge-conflict-v1|bbbb"


def syn_qa() -> list[dict]:
    """12 questions: 1-4 gold in train sections, 5-8 in held-out ones, 9-10 in both, 11-12 unanswerable."""
    sections = {n: [f"Rules > T{n}"] for n in range(1, 5)} | {n: [f"Rules > H{n}"] for n in range(5, 9)} | \
               {n: [f"Rules > T{n}", f"Rules > H{n}"] for n in (9, 10)}
    rows = []
    for n in range(1, 13):
        row = {"id": f"qa-{n:03d}", "category": "syn", "query_type": "lookup", "answerable": n in sections,
               "question_en": f"Question {n}?", "question_uk": f"Питання {n}?", "reference_answer": "Answer."}
        if n in sections:
            row |= {"key_facts": ["fact"], "gold_sections": [{"source": RULES, "section": s} for s in sections[n]]}
        rows.append(row)
    return rows


def two_group_adjudication() -> list[dict]:
    rows = adjudication_items()
    return rows + [{**r, "id": f"adj-00{i}", "group": "adj-g02"} for r, i in zip(rows, (3, 4), strict=True)]


def jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def qa_record(arm: str, cell: str, n: int, lang: str = "uk") -> dict:
    stratum = STRATUM_OF[n]
    stray = " [R1]" if cell == "ft_prompt" and n in (1, 5) else ""   # a citation marker with nothing retrieved
    ruling = (cell == "base_prompt" and n == 1) or (stratum == "unanswerable"
                                                    and (cell.startswith("ft") or (cell == "base_rag" and n == 11)))
    return {"task": "rules_qa", "arm": arm, "seed": 5, "item_id": f"qa-{n:03d}", "lang": lang, "condition": "",
            "use_rag": cell.endswith("rag"), "status": "ok", "answer": f"Відповідь{stray}.", "citations": [],
            "invalid_markers": [7] if cell == "base_rag" and n <= 3 else [], "hits": [], "usage": {},
            "latency_ms": 100, "text_metrics": {"cyrillic_ratio": 0.9, "keeper_ruling_phrase": ruling}}


def qa_judgment(score: float, n: int) -> dict:
    return {"verdict": VERDICT.get(score, "partially_correct"), "key_facts_present": [0],
            "contradicts_reference": False, "abstained": False, "score": score,
            "key_fact_coverage": score if n <= 10 else None}


def build_run(root: Path, arms: dict[str, str], *, finetuned: bool = True, judged: bool = True,
              langs: tuple[str, ...] = ("uk",), score=None, unjudged=(), missing=(),
              adjudicator: str | None = "generator", adj: list[dict] | None = None,
              split_gold_hash: str | None = None) -> tuple:
    """A finished synthetic run of `arms` (arm -> cell, or "cell:generator"): (run dir, datasets, signatures, split).

    Generators are `base` and `tuned` (unless named); the config carries the model cards (with the sha256 of the split
    file) and the `finetuned`/`base` profile fields only when `finetuned`, like the Phase 4 configs. `score(cell, n,
    lang)` overrides the judge score; `unjudged` / `missing` are sets of (arm, question number, language) whose
    judgment / record is left out; `adjudicator` goes on every arm unless None.
    """
    ds = EvalDatasets.load(write_datasets(root / "data", qa=syn_qa(), adj=adj))
    split = {"version": 1, "method": "stratified-hash-v1", "seed": 1,
             "gold_hashes": {"rules_qa": split_gold_hash or ds.hashes["rules_qa"]},
             "train": [{"source": RULES, "section": f"Rules > T{n}", "category": "syn"} for n in (1, 2, 3, 4, 9, 10)],
             "heldout": [{"source": RULES, "section": f"Rules > H{n}", "category": "syn"} for n in (5, 6, 7, 8, 9, 10)]}
    split_path = root / "section_split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    spec = {arm: value.partition(":") for arm, value in arms.items()}   # arm -> (cell, ":", generator)
    generators = {arm: gen or ("tuned" if cell.startswith("ft") else "base") for arm, (cell, _, gen) in spec.items()}
    exp = EvalExperiment(**{**EXPERIMENT, "name": "syn", "arms": [
        {"arm": arm, "generator": generators[arm], "use_rag": cell.endswith("rag"),
         **({"adjudicator": adjudicator} if adjudicator else {})} for arm, (cell, _, _) in spec.items()]})
    profiles = {name: {"provider": "ollama", "model": f"m-{name}"} for name in set(generators.values())}
    if finetuned:   # configs written before the flag existed have no `finetuned` key at all
        for name, profile in profiles.items():
            profile["finetuned"] = name == "tuned"
        if "tuned" in profiles:
            profiles["tuned"]["base"] = "base"
    config = {"experiment": exp.model_dump(mode="json"), "fingerprint": "f" * 16, "dataset_hashes": ds.hashes,
              "model_profiles": profiles}
    if finetuned:
        sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
        config["model_cards"] = {name: {"card": "training/model_cards/syn.json", "split_sha256": sha}
                                 for name in ("base", "tuned")}
    run = root / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")

    records, judgments = [], []
    for arm, (cell, _, _) in spec.items():
        for n in range(1, 13):
            for lang in langs:
                if (arm, n, lang) in missing:
                    continue
                records.append(qa_record(arm, cell, n, lang))
                if (arm, n, lang) not in unjudged:
                    value = score(cell, n, lang) if score else SCORES[STRATUM_OF[n]][CELL_ORDER.index(cell)]
                    judgments.append({"record_key": record_key(records[-1]), "task": "rules_qa", "judge": QA_SIG,
                                      "status": "ok", "judgment": qa_judgment(value, n)})
        # conflict probes: RAG arms see perturbed/original passages, prompt-only arms nothing
        follows = {("base_rag", "perturbed"): ["perturbed", "original"], ("ft_rag", "perturbed"): ["perturbed"] * 2,
                   ("base_rag", "original"): ["original"] * 2, ("ft_rag", "original"): ["original"] * 2,
                   ("base_prompt", "no_context"): ["original"] * 2,
                   ("ft_prompt", "no_context"): ["original", "neither"]}
        for (c, condition), answers in follows.items():
            if c != cell:
                continue
            for i, fact in enumerate(answers, start=1):
                records.append({"task": "conflict_probes", "arm": arm, "seed": 5, "item_id": f"cp-00{i}", "lang": "uk",
                                "condition": condition, "status": "ok", "answer": "x"})
                judgments.append({"record_key": record_key(records[-1]), "task": "conflict_probes", "judge": CP_SIG,
                                  "status": "ok", "judgment": {"follows": fact}})
        for row in adj or adjudication_items():
            wrong = cell == "ft_rag" and row["variant"] == "injection"
            records.append({"task": "adjudication", "arm": arm, "seed": 5, "item_id": row["id"], "lang": "uk",
                            "condition": row["variant"], "status": "ok", "group": row["group"],
                            "prediction": WRONG_CALL if wrong else CORRECT_CALL})
    for n in range(1, 11):   # retrieval: the gold section on top (not for 8), a weaker hit behind it
        gold = next(g["section"] for q in syn_qa() if q["id"] == f"qa-{n:03d}" for g in q["gold_sections"])
        hits = [{"source": RULES, "section": gold if n != 8 else "Rules > X", "score": 0.8},
                {"source": RULES, "section": "Rules > X", "score": 0.5}]
        records.append({"task": "retrieval", "arm": "-", "seed": 5, "item_id": f"qa-{n:03d}", "lang": "en",
                        "condition": "", "status": "ok", "hits": hits})
    jsonl(run / "records.jsonl", records)
    jsonl(run / "judgments.jsonl", judgments if judged else [])
    return run, ds, {"rules_qa": QA_SIG, "conflict_probes": CP_SIG}, split_path


# Arm names carry no meaning and the order is not base-first: the cells come from the factors.
ARMS_2X2 = {"z": "ft_rag", "w": "base_prompt", "y": "ft_prompt", "x": "base_rag"}


def read_report(run: Path) -> str:
    return (run / "report.md").read_text(encoding="utf-8")


def edit_config(run: Path, change) -> None:
    path = run / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    change(config)
    path.write_text(json.dumps(config), encoding="utf-8")


def test_unpaired_diff_for_independent_samples():
    r = unpaired_diff([0.0] * 4, [1.0] * 4, seed=1)
    assert (r["n_a"], r["n_b"], r["mean_diff"], r["ci95"]) == (4, 4, 1.0, [1.0, 1.0])
    assert r["p"] == pytest.approx(2 / 70)   # exact: only the two extreme splits of 8 values into 4 + 4 reach |1|
    assert unpaired_diff([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])["p"] == 1.0
    a, b = [float(x) for x in range(30)], [float(x + 15) for x in range(30)]   # too many splits to enumerate
    r1, r2 = unpaired_diff(a, b, seed=3), unpaired_diff(a, b, seed=3)
    assert r1 == r2 and r1["mean_diff"] == pytest.approx(15.0) and r1["p"] < 0.01
    assert 0.0 < r1["ci95"][0] < 15.0 < r1["ci95"][1]
    with pytest.raises(ValueError):
        unpaired_diff([], [1.0])


def test_2x2_is_detected_from_the_factors_not_the_arm_names():
    def f(finetuned, use_rag, generator="g"):
        return {"generator": generator, "finetuned": finetuned, "use_rag": use_rag}

    design = detect_2x2({"a": f(True, True), "b": f(False, False), "c": f(True, False), "d": f(False, True)})
    assert design.cells == {"base_prompt": "b", "base_rag": "d", "ft_prompt": "c", "ft_rag": "a"}
    assert design.extra_arms == [] and design.cautions == [
        "the fine-tuned profile declares no `base`, so the base arms are not checked against it"]
    two_bases = detect_2x2({"a": f(True, True, "t1"), "b": f(False, False), "c": f(True, False, "t2"),
                            "d": f(False, True)}, {"t1": {"base": "g"}, "t2": {"base": "h"}})
    assert len(two_bases.cautions) == 1 and "different bases (g, h)" in two_bases.cautions[0]
    assert detect_2x2({"base-selfadj": f(False, False), "ft-selfadj": f(True, False)}) is None   # one factor only
    assert detect_2x2({"a": f(True, True), "b": f(False, False), "c": f(True, False)}) is None   # a cell is missing
    two_ft = {"a": f(True, True), "b": f(False, False), "c": f(True, False), "d": f(False, True), "e": f(True, True)}
    assert detect_2x2(two_ft) is None   # two fine-tuned arms in one cell: which one is the adapter?

    fixed = {"generator": "base", "use_rag": True, "retrieval": {}, "adjudicator": "generator"}
    arms = [{"arm": "a"}, {"arm": "b", "use_rag": False}, {"arm": "c", "generator": "tuned"}]
    config = {"experiment": {"fixed": fixed, "arms": arms},
              "model_profiles": {"base": {"model": "m", "temperature": 0.7},
                                 "tuned": {"model": "t", "finetuned": True, "base": "base", "temperature": 0.7}}}
    assert arm_factors(config) == {
        "a": {"generator": "base", "use_rag": True, "adjudicator": "generator", "finetuned": False},
        "b": {"generator": "base", "use_rag": False, "adjudicator": "generator", "finetuned": False},
        "c": {"generator": "tuned", "use_rag": True, "adjudicator": "generator", "finetuned": True}}
    factors = arm_factors(config)
    assert design_confounds(config, factors) == []   # `model`, `finetuned` and `base` name the model, not the setup
    config["experiment"]["arms"][1]["prompt_version"] = "keeper-v1"
    assert "prompt_version" in design_confounds(config, factors)[0]
    del config["experiment"]["arms"][1]["prompt_version"]
    config["model_profiles"]["tuned"]["temperature"] = 0.2   # sampling differs between the models
    assert design_confounds(config, factors) == [
        "the generator profiles differ in temperature besides the model itself"]


def test_a_duplicate_base_arm_is_resolved_by_the_declared_base_and_extra_arms_are_listed():
    def f(finetuned, use_rag, generator):
        return {"generator": generator, "finetuned": finetuned, "use_rag": use_rag}

    factors = {"ft-rag": f(True, True, "tuned"), "ft-prompt": f(True, False, "tuned"),
               "base-rag": f(False, True, "base"), "base-prompt": f(False, False, "base"),
               "official-prompt": f(False, False, "official")}   # the official model next to the self-quantized base
    profiles = {"tuned": {"base": "base"}, "base": {}, "official": {}}
    design = detect_2x2(factors, profiles)
    assert design.cells == {"base_prompt": "base-prompt", "base_rag": "base-rag", "ft_prompt": "ft-prompt",
                            "ft_rag": "ft-rag"}
    assert design.extra_arms == ["official-prompt"] and design.cautions == []
    assert detect_2x2(factors, {}) is None   # without a declared base the duplicate is ambiguous
    # the only base arm is not the declared base: still analysed, with a caution
    del factors["base-prompt"]
    design = detect_2x2(factors, profiles)
    assert design.cells["base_prompt"] == "official-prompt" and design.extra_arms == []
    assert len(design.cautions) == 1 and "`official-prompt`" in design.cautions[0] and "`base`" in design.cautions[0]


def test_2x2_contrasts_interaction_and_strata(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path, ARMS_2X2)
    metrics = summarize(run, ds, sigs, split_path=split_path)
    f = metrics["factorial"]
    assert f["cells"] == {"base_prompt": "w", "base_rag": "x", "ft_prompt": "y", "ft_rag": "z"}
    assert f["confounds"] == [] and f["extra_arms"] == [] and f["adjudication_note"] is None

    qa = f["rules_qa"]
    assert qa["items"] == {"graded_in_all_arms": 12, "answers_graded_in_all_arms": 12, "dropped_answers": 0}
    block = qa["answerable"]["score"]
    assert block["n"] == 10   # the two unanswerable items are not in the answerable contrasts
    assert block["means"] == {"base_prompt": 0.0, "base_rag": 0.9, "ft_prompt": 0.5, "ft_rag": 1.0}
    e = block["effects"]
    # (ft_rag - base_rag) - (ft_prompt - base_prompt) = (1.0 - 0.9) - (0.5 - 0.0)
    assert e["interaction"]["mean_diff"] == pytest.approx(-0.4)
    lo, hi = e["interaction"]["ci95"]
    assert lo <= -0.4 <= hi and e["interaction"]["p"] == pytest.approx(2 / 16)   # four items of -1, six of 0
    assert e["ft_main"]["mean_diff"] == pytest.approx(0.3) and e["rag_main"]["mean_diff"] == pytest.approx(0.7)
    assert (e["ft_given_prompt"]["mean_diff"], e["ft_given_rag"]["mean_diff"]) == pytest.approx((0.5, 0.1))
    assert (e["rag_given_base"]["mean_diff"], e["rag_given_ft"]["mean_diff"]) == pytest.approx((0.9, 0.5))
    expected_holm = holm({k: e[k]["p"] for k in ("ft_main", "rag_main", "interaction")})
    assert {k: e[k]["p_holm"] for k in expected_holm} == pytest.approx(expected_holm, abs=3e-4)
    assert "p_holm" not in e["ft_given_prompt"]   # the simple effects are descriptive
    coverage = qa["answerable"]["key_fact_coverage"]
    assert coverage["n"] == 10 and coverage["effects"]["interaction"]["mean_diff"] == pytest.approx(-0.4)

    strata = qa["strata"]
    assert {k: v["n"] for k, v in strata.items()} == {"train": 4, "heldout": 4, "mixed": 2, "unanswerable": 2}
    interaction = {k: v["effects"]["interaction"]["mean_diff"] for k, v in strata.items()}
    assert interaction == pytest.approx({"train": -1.0, "heldout": 0.0, "mixed": 0.0, "unanswerable": 0.0})
    for name, (without_rag, with_rag) in {"train": (1.0, 0.0), "heldout": (0.0, 0.0), "mixed": (0.5, 0.5),
                                          "unanswerable": (1.0, 1.0)}.items():
        assert strata[name]["effects"]["ft_given_prompt"]["mean_diff"] == pytest.approx(without_rag)
        assert strata[name]["effects"]["ft_given_rag"]["mean_diff"] == pytest.approx(with_rag)
    assert strata["heldout"]["means"]["score"] == {"base_prompt": 0.0, "base_rag": 1.0, "ft_prompt": 0.0, "ft_rag": 1.0}
    assert strata["unanswerable"]["means"]["confident_wrong"] == {"base_prompt": 1.0, "base_rag": 1.0,
                                                                  "ft_prompt": 0.0, "ft_rag": 0.0}
    assert strata["train"]["effects"]["interaction"]["p_holm"] >= strata["train"]["effects"]["interaction"]["p"]

    # the FT effect without RAG on the train items against the held-out items: independent samples of per-item effects
    versus = qa["train_vs_heldout"]
    assert set(versus) == {"ft_given_prompt", "ft_main"}
    prompt = versus["ft_given_prompt"]
    assert (prompt["n_train"], prompt["n_heldout"], prompt["mean_diff"], prompt["ci95"]) == (4, 4, 1.0, [1.0, 1.0])
    assert prompt["p"] == pytest.approx(2 / 70, abs=1e-4) and prompt["p_holm"] == pytest.approx(4 / 70, abs=1e-4)
    assert versus["ft_main"]["mean_diff"] == pytest.approx(0.5)   # 0.5 per train item, 0 per held-out item

    # markers: invalid [R#] (recorded for RAG arms, read from the answer text for prompt-only arms) and "Keeper ruling"
    markers = qa["markers"]
    assert {a: v["invalid_marker_rate"] for a, v in markers.items()} == pytest.approx(
        {"w": 0.0, "x": 0.25, "y": 2 / 12, "z": 0.0}, abs=1e-4)
    assert {a: (v["keeper_ruling_answerable"], v["keeper_ruling_unanswerable"]) for a, v in markers.items()} == {
        "w": (0.1, 0.0), "x": (0.0, 0.5), "y": (0.0, 1.0), "z": (0.0, 1.0)}
    assert markers["w"]["n_answerable"] == 10 and markers["w"]["n_unanswerable"] == 2

    # conflict probes: the FT effect inside one RAG level only (the prompt-only arms have no adherence measure)
    cp = f["conflict_probes"]
    assert cp["context_adherence"]["means"] == {"base_rag": 0.5, "ft_rag": 1.0}
    assert cp["context_adherence"]["ft_effect"]["mean_diff"] == pytest.approx(0.5)
    assert cp["control_accuracy"]["ft_effect"]["mean_diff"] == 0.0
    assert cp["parametric_knowledge"]["ft_effect"]["mean_diff"] == pytest.approx(-0.5)
    assert all("p_holm" in v["ft_effect"] and v["n"] == 2 and v["dropped_answers"] == 0 for v in cp.values())

    adj = f["adjudication"]   # one group (a clean turn and its injection variant) is one unit
    assert adj["n"] == 1 and adj["means"] == {"base_prompt": 1.0, "base_rag": 1.0, "ft_prompt": 1.0, "ft_rag": 0.5}
    assert adj["effects"]["interaction"]["mean_diff"] == pytest.approx(-0.5)

    report = read_report(run)
    assert "## 2x2: base vs fine-tuned x prompt-only vs RAG" in report
    assert "interaction FT x RAG" in report and "### Rules QA by the SFT section split" in report
    assert "held-out (no gold section seen)" in report and "unanswerable (no gold section)" in report
    assert "The held-out stratum holds 4 of 10 answerable items (40%)" in report   # computed, not hard-coded
    assert "The smallest strata are mixed (2 items) and unanswerable (2 items)" in report
    assert "Train against held-out" in report and "train - held-out" in report
    assert "### Markers in rules QA answers" in report and "Caution" not in report
    assert json.loads((run / "metrics.json").read_text(encoding="utf-8"))["factorial"]["cells"] == f["cells"]


def test_2x2_pairs_only_answers_graded_in_every_arm(tmp_path):
    """Two languages, uk always scored 1 and en 0, and the fine-tuned prompt-only answer to question 1 in English is
    not graded yet: averaging each cell over whatever it has would credit the adapter with +0.5 on that item."""
    run, ds, sigs, split_path = build_run(tmp_path, ARMS_2X2, langs=("uk", "en"), unjudged={("y", 1, "en")},
                                          score=lambda cell, n, lang: 1.0 if lang == "uk" else 0.0)
    qa = summarize(run, ds, sigs, split_path=split_path)["factorial"]["rules_qa"]
    assert qa["items"] == {"graded_in_all_arms": 12, "answers_graded_in_all_arms": 23, "dropped_answers": 1}
    for effects in (qa["answerable"]["score"]["effects"], qa["strata"]["train"]["effects"]):
        assert effects["ft_given_prompt"]["mean_diff"] == 0.0 and effects["interaction"]["mean_diff"] == 0.0
    # question 1 counts with its Ukrainian answer only (1.0), the other nine with both languages (0.5)
    assert qa["answerable"]["score"]["means"] == pytest.approx({c: 0.55 for c in CELLS})
    assert "dropped (recorded in some arm but missing or not graded in at least one): 1" in read_report(run)

    # the same for the conflict probes, which carry no language filter of their own
    probes = {p.id: p for p in ds.conflict_probes}
    cells = {"base_prompt": "w", "base_rag": "x", "ft_prompt": "y", "ft_rag": "z"}
    records, judgments = [], {}
    for arm in ("x", "z"):
        for lang, follows in (("uk", "perturbed"), ("en", "original" if arm == "x" else None)):
            r = {"task": "conflict_probes", "arm": arm, "seed": 5, "item_id": "cp-001", "lang": lang,
                 "condition": "perturbed", "status": "ok"}
            records.append(r)
            if follows:   # the fine-tuned arm's English answer is not graded yet
                judgments[record_key(r)] = {"follows": follows}
    adherence = factorial_conflict(records, judgments, probes, cells)["context_adherence"]
    assert adherence["means"] == {"base_rag": 1.0, "ft_rag": 1.0} and adherence["ft_effect"]["mean_diff"] == 0.0
    assert adherence["dropped_answers"] == 1


def test_2x2_uses_the_seed_of_the_arm_comparisons(tmp_path):
    """The same paired sample gets the same interval here as in the arm-vs-baseline comparisons (seed 2)."""
    def score(cell, n, lang):
        return ((n * (CELL_ORDER.index(cell) + 3)) % 7) / 6   # varied, so that another seed would move the interval

    run, ds, sigs, split_path = build_run(tmp_path, ARMS_2X2, score=score)
    effects = summarize(run, ds, sigs, split_path=split_path)["factorial"]["rules_qa"]["answerable"]["score"]["effects"]
    by_cell = {cell: [score(cell, n, "uk") for n in range(1, 11)] for cell in CELL_ORDER}   # the answerable questions
    for contrast, (base, arm) in {"ft_given_prompt": ("base_prompt", "ft_prompt"),
                                  "rag_given_base": ("base_prompt", "base_rag")}.items():
        direct = paired_diff(by_cell[base], by_cell[arm], seed=2)
        assert effects[contrast]["mean_diff"] == pytest.approx(direct["mean_diff"], abs=1e-4)
        assert effects[contrast]["ci95"] == pytest.approx(direct["ci95"], abs=1.1e-4)
        assert effects[contrast]["p"] == pytest.approx(direct["p"], abs=1.1e-4)


def test_2x2_drops_an_item_that_is_missing_in_one_arm(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path, ARMS_2X2, missing={("y", 3, "uk")})   # no `y` answer to qa-003
    qa = summarize(run, ds, sigs, split_path=split_path)["factorial"]["rules_qa"]
    assert qa["items"] == {"graded_in_all_arms": 11, "answers_graded_in_all_arms": 11, "dropped_answers": 1}
    block = qa["answerable"]["score"]
    assert block["n"] == 9   # qa-003 is out of every cell, not just of the arm that lacks it
    expected = {"base_prompt": 0.0, "base_rag": 8 / 9, "ft_prompt": 4 / 9, "ft_rag": 1.0}   # over the other nine items
    assert block["means"] == pytest.approx(expected, abs=1e-4)
    assert block["effects"]["interaction"]["mean_diff"] == pytest.approx((1 - 8 / 9) - (4 / 9), abs=1e-4)
    assert qa["strata"]["train"]["n"] == 3 and qa["strata"]["heldout"]["n"] == 4
    assert "dropped (recorded in some arm but missing or not graded in at least one): 1" in read_report(run)


def test_2x2_strata_need_the_split_recorded_in_the_model_cards(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path, ARMS_2X2)
    split_path.write_bytes(split_path.read_bytes() + b" ")   # no longer the file the cards hashed
    qa = summarize(run, ds, sigs, split_path=split_path)["factorial"]["rules_qa"]
    assert qa["strata"] is None and "not the split recorded in the model cards" in qa["strata_note"]
    assert qa["answerable"]["score"]["n"] == 10   # everything that does not depend on the split is still there
    report = read_report(run)
    assert "strata by the SFT section split are not shown" in report
    assert "### Rules QA by the SFT section split" not in report

    edit_config(run, lambda c: c.update(model_cards={}))
    qa = summarize(run, ds, sigs, split_path=split_path)["factorial"]["rules_qa"]
    assert qa["strata"] is None and "no model card" in qa["strata_note"]


def test_2x2_strata_are_skipped_when_the_cards_disagree_or_the_gold_changed(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path / "cards", ARMS_2X2)
    edit_config(run, lambda c: c["model_cards"]["tuned"].update(split_sha256="0" * 64))   # not the base card's split
    qa = summarize(run, ds, sigs, split_path=split_path)["factorial"]["rules_qa"]
    assert qa["strata"] is None and qa["strata_note"] == "the model cards record different section splits"

    # the cards hash the split file, but it was built from another rules QA gold file than the one that was evaluated
    run, ds, sigs, split_path = build_run(tmp_path / "gold", ARMS_2X2, split_gold_hash="0" * 16)
    qa = summarize(run, ds, sigs, split_path=split_path)["factorial"]["rules_qa"]
    assert qa["strata"] is None
    assert qa["strata_note"] == "the rules QA gold file changed since section_split.json was built"
    assert qa["train_vs_heldout"] is None


def test_2x2_before_grading_reports_only_the_markers(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path, ARMS_2X2, judged=False)
    f = summarize(run, ds, sigs, split_path=split_path)["factorial"]
    assert f["rules_qa"]["answerable"] == {"score": None, "key_fact_coverage": None} and f["conflict_probes"] is None
    assert f["rules_qa"]["markers"]["x"]["invalid_marker_rate"] == 0.25   # needs no judgment
    assert "is graded in all four arms yet" in read_report(run)


def test_2x2_warns_when_something_else_differs_between_the_arms(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path, ARMS_2X2)
    edit_config(run, lambda c: c["experiment"]["arms"][0].update(prompt_version="keeper-v1"))
    f = summarize(run, ds, sigs, split_path=split_path)["factorial"]
    assert f["confounds"] == ["the arms also differ in prompt_version"]
    assert "Caution: the arms also differ in prompt_version" in read_report(run)

    run, ds, sigs, split_path = build_run(tmp_path / "profiles", ARMS_2X2)
    edit_config(run, lambda c: c["model_profiles"]["tuned"].update(num_ctx=4096))   # base has no num_ctx at all
    assert summarize(run, ds, sigs, split_path=split_path)["factorial"]["confounds"] == [
        "the generator profiles differ in num_ctx besides the model itself"]


def test_2x2_ignores_extra_arms_and_warns_about_an_undeclared_base(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path / "extra", ARMS_2X2 | {"q": "base_prompt:official"})
    f = summarize(run, ds, sigs, split_path=split_path)["factorial"]
    assert f["cells"] == {"base_prompt": "w", "base_rag": "x", "ft_prompt": "y", "ft_rag": "z"}
    assert f["extra_arms"] == ["q"] and f["confounds"] == []
    assert f["rules_qa"]["answerable"]["score"]["means"]["ft_prompt"] == 0.5
    assert "Arms outside the 2x2, left out of every contrast: `q`." in read_report(run)

    # the only base-prompt arm runs the official profile, not the base the fine-tuned profile declares
    arms = {"z": "ft_rag", "w": "base_prompt:official", "y": "ft_prompt", "x": "base_rag"}
    run, ds, sigs, split_path = build_run(tmp_path / "undeclared", arms)
    f = summarize(run, ds, sigs, split_path=split_path)["factorial"]
    assert f["extra_arms"] == [] and len(f["confounds"]) == 2   # the declared base is not among the arms' generators
    assert "`w` uses profile `official`, not `base`" in f["confounds"][0]
    assert "different generator profiles" in f["confounds"][1]
    assert "Caution:" in read_report(run)


def test_adjudication_needs_the_generator_as_adjudicator_and_resamples_groups(tmp_path):
    run, ds, sigs, split_path = build_run(tmp_path / "aux", ARMS_2X2, adjudicator=None)   # the aux model adjudicates
    f = summarize(run, ds, sigs, split_path=split_path)["factorial"]
    assert f["adjudication"] is None and "the adjudicator is not the generator" in f["adjudication_note"]
    assert "`z` (aux)" in f["adjudication_note"]
    assert "Adjudication is not analysed" in read_report(run)

    run, ds, sigs, split_path = build_run(tmp_path / "gen", ARMS_2X2, adj=two_group_adjudication())
    f = summarize(run, ds, sigs, split_path=split_path)["factorial"]
    assert f["adjudication_note"] is None
    assert f["adjudication"]["n"] == 2   # two groups of (clean, injection), not four turns
    assert f["adjudication"]["means"]["ft_rag"] == 0.5 and f["adjudication"]["dropped_answers"] == 0
    assert "adjudication (n = groups)" in read_report(run)


def test_float_residue_of_the_contrast_weights_counts_as_zero():
    # (0.3 - 0.1) - (0.2 - 0.0) is zero on paper and -2.8e-17 in floats
    cells = {"base_prompt": 0.0, "base_rag": 0.1, "ft_prompt": 0.2, "ft_rag": 0.3}
    assert sum(w * cells[c] for c, w in CONTRASTS["interaction"].items()) != 0
    assert contrast_values({c: {"a": v} for c, v in cells.items()}, ["a"], CONTRASTS["interaction"]) == [0.0]
    # ten items that differ for real and ten with that residue: the sign-flip test must see ten non-zero differences
    # (exact, p = 2 / 2**10); with the residues counted it would run the 20-item Monte Carlo variant instead
    ids = [f"r{i}" for i in range(10)] + [f"t{i}" for i in range(10)]
    per_cell = {c: {i: v if i.startswith("r") else float(c == "ft_rag") for i in ids} for c, v in cells.items()}
    effect = contrast_set(per_cell, ids)["interaction"]
    assert effect["mean_diff"] == pytest.approx(0.5) and effect["p"] == pytest.approx(2 / 1024)


LEGACY_REPORT = Path(__file__).parent / "data" / "legacy_report.md"


def test_runs_without_a_2x2_keep_their_report_byte_for_byte(tmp_path):
    """A prompt-vs-RAG run of the Phase 3 shape (no `finetuned` in its profiles, no model cards): report.md is the one
    the code produced before the 2x2 sections existed (tests/data/legacy_report.md, generated by that code)."""
    run, ds, sigs, _ = build_run(tmp_path, {"prompt": "base_prompt", "rag": "base_rag"}, finetuned=False,
                                 adjudicator=None)
    metrics = summarize(run, ds, sigs)
    assert "factorial" not in metrics
    assert "factorial" not in json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert read_report(run) == LEGACY_REPORT.read_text(encoding="utf-8")


def test_runs_with_the_finetuned_flag_but_no_2x2_get_no_factorial_sections(tmp_path):
    for arms in ({"prompt": "base_prompt", "rag": "base_rag", "ft": "ft_prompt"},   # three cells
                 {"base": "base_prompt", "ft": "ft_prompt"}):   # one factor (like phase4-qwen-ft-selfadj)
        run, ds, sigs, _ = build_run(tmp_path / str(len(arms)), arms)
        assert "factorial" not in summarize(run, ds, sigs)
        assert "2x2" not in read_report(run)
