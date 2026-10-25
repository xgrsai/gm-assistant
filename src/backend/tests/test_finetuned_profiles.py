"""Phase 4 model profiles in evals: model cards in the run config, the Ollama digest check, the Phase 4 experiments.

Offline: model cards live in a tmp repo root, Ollama is a callable returning a dict, no model is called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evals import model_cards, run
from evals.datasets import EvalDatasets, file_sha256
from evals.experiment import EvalExperiment
from evals.harness import Harness, fingerprint_view
from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.llm import ModelFactory
from keeper.prompts import PromptSet
from keeper.rag.corpus import load_corpus
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from server.settings import REPO_ROOT

from .test_evals import write_datasets

BACKEND = Path(__file__).resolve().parents[1]
EXPERIMENTS = BACKEND / "evals" / "experiments"
CARD = "training/model_cards/keeper-qwen25-3b-sft-v1.json"
FT_PROFILE, BASE_PROFILE = "keeper-qwen25-3b-ollama", "qwen25-3b-selfq4-ollama"
FT_TAG, BASE_TAG = "keeper-qwen25-3b:sft-v1", "qwen2.5-3b-selfq4:v1"
FT_DIGEST, BASE_DIGEST = "a1" * 32, "b2" * 32
FT_WEIGHTS, BASE_WEIGHTS = "c3" * 32, "c4" * 32
LEGACY_PROFILE_KEYS = {"provider", "model", "temperature", "max_tokens", "thinking_level", "context_window", "num_ctx",
                       "finetuned", "base", "quant"}
LEGACY_SNAPSHOT_KEYS = {"experiment", "dataset_hashes", "spec_hashes", "model_profiles", "prompt_hashes",
                        "corpus_chunk_ids_hash", "judge", "fingerprint"}


@pytest.fixture(scope="module")
def llms() -> ModelFactory:
    return ModelFactory.from_yaml(BACKEND / "config" / "models.yaml")


@pytest.fixture(scope="module")
def chunks():
    return load_corpus([REPO_ROOT / "data" / "open" / "coc7e"], chunk_size=1000, overlap=200, root=REPO_ROOT)


def write_card(root: Path, **override) -> Path:
    """A card shaped like the one sft/check_adapter.py writes (only the keys the harness reads, plus noise)."""
    card = {"schema": 1, "status": "passed", "generated_at": "2026-10-25T10:00:00+00:00",
            "ollama_digests": {"ft": FT_DIGEST, "base": BASE_DIGEST},
            "gguf": {"sha256": FT_WEIGHTS, "base_sha256": BASE_WEIGHTS,
                     "files": {"ft.gguf": {"role": "ft", "sha256": FT_WEIGHTS},
                               "base.gguf": {"role": "base", "sha256": BASE_WEIGHTS}}},
            "data": {"train": {"sha256": "d4" * 32}, "val": {"sha256": "e5" * 32}}, "split": {"sha256": "f6" * 32},
            "check_adapter": {"elapsed_s": 41.0}, "git": {"commit": "abc"}, **override}
    path = root / CARD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card), encoding="utf-8")
    return path


def make_harness(name: str, llms, chunks, root: Path, *, judge: bool = True) -> Harness:
    exp = EvalExperiment.load(EXPERIMENTS / f"{name}.yaml")
    specs = exp.specs(exp.seeds[0])
    deps = Deps(llms=llms, prompts=PromptSet(), scenarios=ScenarioRepo(REPO_ROOT / "data" / "scenarios"),
                experiments=ExperimentRegistry({s.arm: s for s in specs}, specs[0].arm),
                runlog=RunLog(root / "runlog", enabled=False), retriever=None)
    return Harness(exp=exp, deps=deps, datasets=EvalDatasets.load(write_datasets(root / "data")), chunks=chunks,
                   run_dir=root / "run", contexts_dir=root / "contexts", repo_root=root, judge_enabled=judge)


def listing(ft: str | None = FT_DIGEST, base: str | None = BASE_DIGEST) -> dict[str, str]:
    return {tag: digest for tag, digest in ((FT_TAG, ft), (BASE_TAG, base)) if digest is not None}


# ------------------------------------------------------------------------------------------ run configuration


@pytest.mark.parametrize("path", sorted(EXPERIMENTS.glob("phase3-*.yaml")), ids=lambda p: p.stem)
def test_phase3_run_configs_keep_their_keys(path, llms, chunks, tmp_path):
    """A run without a model card records exactly what it recorded before model cards existed."""
    snap = make_harness(path.stem, llms, chunks, tmp_path).config_snapshot()
    assert set(snap) == LEGACY_SNAPSHOT_KEYS
    assert snap["model_profiles"] and all(set(p) == LEGACY_PROFILE_KEYS for p in snap["model_profiles"].values())
    assert set(snap["judge"]["profile"]) == LEGACY_PROFILE_KEYS
    assert "model_cards" not in snap and "model_cards" not in fingerprint_view(snap)


def test_phase4_run_config_records_the_card_of_both_models(llms, chunks, tmp_path):
    card = write_card(tmp_path)
    snap = make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()
    shared = {"card": CARD, "card_sha256": file_sha256(card), "status": "passed", "train_sha256": "d4" * 32,
              "val_sha256": "e5" * 32, "split_sha256": "f6" * 32}
    assert snap["model_cards"] == {   # each profile records the weights and the digest of its OWN model
        FT_PROFILE: {**shared, "role": "ft", "ollama_digest": FT_DIGEST, "weights_sha256": FT_WEIGHTS},
        BASE_PROFILE: {**shared, "role": "base", "ollama_digest": BASE_DIGEST, "weights_sha256": BASE_WEIGHTS}}
    profiles = snap["model_profiles"]
    assert profiles[FT_PROFILE]["model_card"] == profiles[BASE_PROFILE]["model_card"] == CARD
    assert set(profiles["gemini-aux"]) == LEGACY_PROFILE_KEYS   # profiles without a card are recorded as before
    assert "allow_failed_card" not in snap
    assert all("card_sha256" not in c for c in fingerprint_view(snap)["model_cards"].values())


def test_weights_come_from_the_files_of_the_role_else_from_the_gguf_summary(llms, chunks, tmp_path):
    def weights(**gguf):
        write_card(tmp_path, gguf=gguf)
        cards = make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()["model_cards"]
        return cards[FT_PROFILE]["weights_sha256"], cards[BASE_PROFILE]["weights_sha256"]

    assert weights(sha256=FT_WEIGHTS, base_sha256=BASE_WEIGHTS) == (FT_WEIGHTS, BASE_WEIGHTS)   # no files listed
    assert weights(sha256="01" * 32, base_sha256="02" * 32, files={
        "b.gguf": {"role": "base", "sha256": BASE_WEIGHTS}, "f.gguf": {"role": "ft", "sha256": FT_WEIGHTS}}) == (
        FT_WEIGHTS, BASE_WEIGHTS)   # the files of the roles win, whatever their order
    assert weights() == (None, None)


def test_the_fingerprint_follows_what_identifies_the_models(llms, chunks, tmp_path):
    def fingerprint(**card):
        write_card(tmp_path, **card)
        return make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()["fingerprint"]

    first = fingerprint()
    first_file = file_sha256(tmp_path / CARD)
    assert fingerprint() == first
    # a repeated check of the same models rewrites timestamps, timings and git state: the run must stay resumable
    card = write_card(tmp_path, generated_at="2026-11-01T00:00:00+00:00", git={"commit": "def"},
                      check_adapter={"elapsed_s": 99.0})
    snap = make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()
    assert snap["fingerprint"] == first
    assert snap["model_cards"][FT_PROFILE]["card_sha256"] == file_sha256(card) != first_file   # recorded, not hashed in
    # while anything that identifies a model changes it
    assert fingerprint(gguf={"sha256": "99" * 32, "base_sha256": BASE_WEIGHTS}) != first    # other FT weights
    assert fingerprint(gguf={"sha256": FT_WEIGHTS, "base_sha256": "99" * 32}) != first      # other base weights
    assert fingerprint(ollama_digests={"ft": "99" * 32, "base": BASE_DIGEST}) != first
    assert fingerprint(data={"train": {"sha256": "99" * 32}, "val": {"sha256": "e5" * 32}}) != first
    assert fingerprint(split={"sha256": "99" * 32}) != first
    assert fingerprint(status="failed") != first


def test_card_fields_that_are_absent_are_recorded_as_null(llms, chunks, tmp_path):
    write_card(tmp_path, gguf=None, data={}, split=None, status=None)
    entry = make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()["model_cards"][FT_PROFILE]
    assert entry["weights_sha256"] is None and entry["train_sha256"] is None and entry["split_sha256"] is None
    assert entry["status"] is None and entry["ollama_digest"] == FT_DIGEST


def test_a_failed_card_override_is_recorded_but_is_not_part_of_the_fingerprint(llms, chunks, tmp_path):
    write_card(tmp_path, status="failed")
    plain = make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()
    allowed = make_harness("phase4-qwen-ft", llms, chunks, tmp_path)
    allowed.allow_failed_card = True
    snap = allowed.config_snapshot()
    assert "allow_failed_card" not in plain and snap["allow_failed_card"] is True
    assert snap["fingerprint"] == plain["fingerprint"] and snap["model_cards"][FT_PROFILE]["status"] == "failed"
    write_card(tmp_path)   # a card that passed needs no override: the flag leaves no trace
    allowed = make_harness("phase4-qwen-ft", llms, chunks, tmp_path)
    allowed.allow_failed_card = True
    assert "allow_failed_card" not in allowed.config_snapshot()


def test_a_session_remembers_the_card_file_and_the_override(llms, chunks, tmp_path):
    write_card(tmp_path, status="failed")
    first = make_harness("phase4-qwen-ft", llms, chunks, tmp_path)
    first.allow_failed_card = True
    first.bind_run_dir()
    card = write_card(tmp_path, status="failed", generated_at="2026-11-01T00:00:00+00:00")   # check rerun, same models
    again = make_harness("phase4-qwen-ft", llms, chunks, tmp_path)
    again.allow_failed_card = True
    config = again.bind_run_dir()
    session = config["sessions"][-1]
    assert session["card_sha256"] == {FT_PROFILE: file_sha256(card), BASE_PROFILE: file_sha256(card)}
    assert session["allow_failed_card"] is True and config["allow_failed_card"] is True


def test_run_config_needs_the_card(llms, chunks, tmp_path):
    with pytest.raises(model_cards.ModelCardError, match="check_adapter"):
        make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()
    (tmp_path / CARD).parent.mkdir(parents=True)
    (tmp_path / CARD).write_text("{not json", encoding="utf-8")
    with pytest.raises(model_cards.ModelCardError, match="not valid JSON"):
        make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()


# ------------------------------------------------------------------------------------------ the experiments


def test_phase4_profiles_form_a_matched_pair(llms):
    official, base, tuned = (llms.profile(n) for n in ("qwen25-3b-ollama", BASE_PROFILE, FT_PROFILE))
    assert (base.model, tuned.model) == (BASE_TAG, FT_TAG) and official.model == "qwen2.5:3b"
    assert tuned.finetuned and tuned.base == BASE_PROFILE and not base.finetuned and base.base is None
    assert base.model_card == tuned.model_card == CARD and official.model_card is None   # phase3 profile untouched
    assert "self-quantized" in base.quant and "merged LoRA" in tuned.quant and "model card" in tuned.quant
    # sampling, num_ctx, context window and output budget are those of the phase3 profile: only the model differs
    other = {"model", "finetuned", "base", "quant", "model_card"}
    for profile in (base, tuned):
        assert {k: v for k, v in profile.snapshot().items() if k not in other} == {
            k: v for k, v in official.snapshot().items() if k not in other}
    differing = {k for k, v in tuned.snapshot().items() if v != base.snapshot()[k]}
    assert differing == {"model", "finetuned", "base", "quant"}


def test_phase4_experiment_differs_from_phase3_local_qwen_only_by_the_models():
    phase3 = EvalExperiment.load(EXPERIMENTS / "phase3-local-qwen.yaml")
    ft = EvalExperiment.load(EXPERIMENTS / "phase4-qwen-ft.yaml")
    assert [(a["arm"], a["generator"], a["use_rag"]) for a in ft.arms] == [
        ("base-prompt", BASE_PROFILE, False), ("base-rag", BASE_PROFILE, True),
        ("ft-prompt", FT_PROFILE, False), ("ft-rag", FT_PROFILE, True)]
    assert ft.fixed == phase3.fixed and ft.tasks == phase3.tasks and ft.judge == phase3.judge == "gemini-judge"
    assert (ft.seeds, ft.scenario_id, ft.dataset_dir, ft.judge_prompt_versions) == (
        phase3.seeds, phase3.scenario_id, phase3.dataset_dir, phase3.judge_prompt_versions)
    assert ft.tasks.rules_qa.context_query == "question_en" and not ft.tasks.adjudication.enabled


def test_claude_judge_experiment_is_phase4_with_another_judge_and_binds_to_its_run(llms, chunks, tmp_path):
    write_card(tmp_path)
    plain = EvalExperiment.load(EXPERIMENTS / "phase4-qwen-ft.yaml")
    claude = EvalExperiment.load(EXPERIMENTS / "phase4-qwen-ft-claude-judge.yaml")
    assert claude.judge == "claude-sonnet-agent" and llms.profile("claude-sonnet-agent").provider == "external"
    differing = {k for k, v in plain.model_dump().items() if v != claude.model_dump()[k]}
    assert differing == {"description", "judge"}   # the name stays: it is part of the fingerprint
    a = make_harness("phase4-qwen-ft", llms, chunks, tmp_path).config_snapshot()
    b = make_harness("phase4-qwen-ft-claude-judge", llms, chunks, tmp_path).config_snapshot()
    assert a["fingerprint"] == b["fingerprint"]   # the judge is not part of it: this file binds to the existing run
    assert a["judge"]["profile"]["provider"] == "google_genai" and b["judge"]["profile"]["provider"] == "external"
    assert set(b["judge"]["profile"]) == LEGACY_PROFILE_KEYS
    assert b["judge"]["prompt_hashes"] == a["judge"]["prompt_hashes"]   # the same judge prompts
    # nothing is sent anywhere: the harness reports the answers to grade outside instead of Gemini calls
    plan = make_harness("phase4-qwen-ft-claude-judge", llms, chunks, tmp_path).plan()
    assert plan["of which Gemini (API quota)"] == 0 and plan["rules_qa (judge)"] == plan["conflict_probes (judge)"] == 0
    assert plan["to grade outside the harness (claude-sonnet-agent)"] == (
        plan["rules_qa (narrate)"] + plan["conflict_probes (narrate)"])
    assert plan["total"] == plan["rules_qa (narrate)"] + plan["conflict_probes (narrate)"]


def test_selfadj_experiment_isolates_the_adjudicator(llms):
    exp = EvalExperiment.load(EXPERIMENTS / "phase4-qwen-ft-selfadj.yaml")
    specs = exp.specs(exp.seeds[0])
    assert [s.generator for s in specs] == [BASE_PROFILE, FT_PROFILE]
    assert all(s.adjudicator == "generator" and not s.use_rag and s.aux == "qwen25-3b-ollama-aux" for s in specs)
    assert exp.judge is None and exp.rag_specs() == []
    t = exp.tasks
    assert [t.retrieval.enabled, t.rules_qa.enabled, t.conflict_probes.enabled, t.adjudication.enabled] == [
        False, False, False, True]
    assert {s.generator for s in specs} | {s.aux for s in specs} <= set(llms.profiles)


def test_selfadj_run_config_records_the_card_but_not_the_aux_model(llms, chunks, tmp_path):
    write_card(tmp_path)
    snap = make_harness("phase4-qwen-ft-selfadj", llms, chunks, tmp_path).config_snapshot()
    assert set(snap["model_cards"]) == {FT_PROFILE, BASE_PROFILE} and snap["judge"]["profile"] is None
    assert set(snap["model_profiles"]["qwen25-3b-ollama-aux"]) == LEGACY_PROFILE_KEYS


def test_dry_run_counts_gemini_calls_separately(llms, chunks, tmp_path):
    no_judge = make_harness("phase4-qwen-ft", llms, chunks, tmp_path, judge=False).plan()
    assert no_judge["of which Gemini (API quota)"] == 0 and no_judge["frozen_contexts (aux analyze)"] == 0
    assert no_judge["rules_qa (narrate)"] > 0 and no_judge["rules_qa (judge)"] == 0   # narration is local
    graded = make_harness("phase4-qwen-ft", llms, chunks, tmp_path).plan()
    assert graded["of which Gemini (API quota)"] == graded["rules_qa (judge)"] + graded["conflict_probes (judge)"] > 0
    selfadj = make_harness("phase4-qwen-ft-selfadj", llms, chunks, tmp_path).plan()
    assert selfadj["adjudication (adjudicator model)"] > 0 and selfadj["of which Gemini (API quota)"] == 0
    everything = make_harness("phase3-gemini", llms, chunks, tmp_path).plan()   # all Gemini: the line equals the total
    assert everything["of which Gemini (API quota)"] == everything["total"] > 0


# ------------------------------------------------------------------------------------------ digest check


def check(llms, tmp_path, lister, name: str = "phase4-qwen-ft", **kw) -> None:
    run.check_model_cards(llms, EvalExperiment.load(EXPERIMENTS / f"{name}.yaml"), root=tmp_path, listing=lister, **kw)


def test_matching_digests_pass(llms, tmp_path):
    write_card(tmp_path)
    check(llms, tmp_path, listing)
    check(llms, tmp_path, lambda: {**listing(), "unrelated:1b": "0" * 64})
    write_card(tmp_path, ollama_digests={"ft": FT_DIGEST[:12], "base": "sha256:" + BASE_DIGEST.upper()})
    check(llms, tmp_path, listing)   # `ollama list` short ids and a sha256: prefix are accepted
    check(llms, tmp_path, listing, name="phase4-qwen-ft-selfadj")


@pytest.mark.parametrize(("lister", "message"), [
    (lambda: listing(ft="c3" * 32), "sft-v1.* has digest c3c3c3c3c3c3 but .*ollama_digests.ft is a1a1"),
    (lambda: listing(base="c3" * 32), "selfq4:v1.* has digest c3c3c3c3c3c3 but .*ollama_digests.base is b2b2"),
    (lambda: listing(ft=None), "Ollama has no model 'keeper-qwen25-3b:sft-v1'"),
    (lambda: listing(base=None), "Ollama has no model 'qwen2.5-3b-selfq4:v1'"),
])
def test_a_rewritten_tag_stops_the_run(llms, tmp_path, lister, message):
    write_card(tmp_path)
    with pytest.raises(SystemExit, match=message):
        check(llms, tmp_path, lister)


def test_the_base_model_is_verified_even_without_a_finetuned_arm(llms, tmp_path):
    write_card(tmp_path)
    raw = yaml.safe_load((EXPERIMENTS / "phase4-qwen-ft.yaml").read_text(encoding="utf-8"))
    exp = EvalExperiment(**{**raw, "arms": raw["arms"][:2]})   # base-prompt and base-rag only
    run.check_model_cards(llms, exp, root=tmp_path, listing=lambda: {BASE_TAG: BASE_DIGEST})   # no FT tag needed
    with pytest.raises(SystemExit, match="has digest c3c3c3c3c3c3"):
        run.check_model_cards(llms, exp, root=tmp_path, listing=lambda: {BASE_TAG: "c3" * 32})


def test_an_unreachable_ollama_stops_the_run(llms, tmp_path):
    write_card(tmp_path)

    def down():
        raise ConnectionError("Failed to connect to Ollama")

    with pytest.raises(SystemExit, match="cannot list the Ollama models.*ConnectionError"):
        check(llms, tmp_path, down)


def test_the_card_must_exist_and_carry_both_digests(llms, tmp_path, capsys):
    with pytest.raises(SystemExit, match="not found.*python -m sft.check_adapter"):
        check(llms, tmp_path, listing)
    check(llms, tmp_path, listing, warn_only=True)   # a dry run only warns
    assert capsys.readouterr().out.count("WARNING: model card") == 1   # the pair shares one card: reported once
    write_card(tmp_path, ollama_digests={"base": BASE_DIGEST})
    with pytest.raises(SystemExit, match="no ollama_digests.ft"):
        check(llms, tmp_path, listing)
    write_card(tmp_path, ollama_digests={"ft": FT_DIGEST})
    with pytest.raises(SystemExit, match="no ollama_digests.base"):
        check(llms, tmp_path, listing)


def test_a_card_that_did_not_pass_stops_the_run_unless_overridden(llms, tmp_path, capsys):
    write_card(tmp_path, status="failed")
    with pytest.raises(SystemExit, match=r"has status 'failed'.*check_adapter.*--allow-failed-card") as stop:
        check(llms, tmp_path, listing)
    assert str(stop.value).count("has status") == 1   # the pair shares one card: reported once
    check(llms, tmp_path, listing, warn_only=True)   # a dry run warns
    assert capsys.readouterr().out.count("WARNING: model card") == 1
    check(llms, tmp_path, listing, allow_failed_card=True)   # the deliberate override continues, loudly
    assert "continuing: --allow-failed-card" in capsys.readouterr().out
    write_card(tmp_path, status=None)   # no status at all is not a pass
    with pytest.raises(SystemExit, match="has status None"):
        check(llms, tmp_path, listing)
    with pytest.raises(SystemExit, match="has status None"):
        check(llms, tmp_path, None)   # --judge-only grades answers of a failed adapter just as pointlessly
    write_card(tmp_path)
    check(llms, tmp_path, listing, allow_failed_card=True)
    assert capsys.readouterr().out == ""   # a passed card needs no override and prints nothing


def test_the_override_does_not_hide_a_changed_tag(llms, tmp_path):
    write_card(tmp_path, status="failed")
    with pytest.raises(SystemExit, match="has digest c3c3c3c3c3c3"):
        check(llms, tmp_path, lambda: listing(ft="c3" * 32), allow_failed_card=True)


@pytest.mark.parametrize(("flags", "expected"), [
    ({}, (True, True, False)),                                  # a real run: cards and Ollama, stop on problems
    ({"dry_run": True}, (True, True, True)),                    # --dry-run: the same checks, only warnings
    ({"judge_only": True}, (True, False, False)),               # --judge-only: cards (fingerprint) but not Ollama
    ({"summarize_only": True}, (False, False, False)),          # --summarize-only: no model is touched
    ({"dry_run": True, "judge_only": True}, (True, True, True)),   # the dry run returns before any judging
    ({"summarize_only": True, "dry_run": True, "judge_only": True}, (False, False, False)),
])
def test_card_check_mode_per_cli_mode(flags, expected):
    mode = run.card_check_mode(**{"summarize_only": False, "dry_run": False, "judge_only": False, **flags})
    assert (mode.enabled, mode.ollama, mode.warn_only) == expected


def test_allow_failed_card_is_an_explicit_flag():
    parse = run.build_parser().parse_args
    assert parse(["x.yaml"]).allow_failed_card is False
    assert parse(["x.yaml", "--allow-failed-card"]).allow_failed_card is True


def test_without_a_listing_only_the_cards_are_checked(llms, tmp_path):
    with pytest.raises(SystemExit, match="not found"):   # --judge-only still needs the card for the fingerprint
        check(llms, tmp_path, None)
    write_card(tmp_path)
    check(llms, tmp_path, None)


def test_runs_without_a_model_card_never_ask_ollama(llms, tmp_path):
    def forbidden():
        raise AssertionError("Ollama must not be contacted")

    for name in ("phase3-local-qwen", "phase3-gemini"):
        check(llms, tmp_path, forbidden, name=name)   # no card file at all, and none is needed


def test_ollama_listing_maps_tags_to_digests(monkeypatch):
    import ollama

    class Model:
        def __init__(self, model, digest):
            self.model, self.digest = model, digest

    class FakeClient:
        def __init__(self, host=None, **kw):
            self.host, self.kw = host, kw

        def list(self):
            return type("Listing", (), {"models": [Model(FT_TAG, FT_DIGEST), Model("plain", BASE_DIGEST),
                                                   Model(None, "x"), Model("no-digest:1", None)]})()

    monkeypatch.setattr(ollama, "Client", FakeClient)
    assert model_cards.ollama_listing("http://ollama:11434") == {FT_TAG: FT_DIGEST, "plain:latest": BASE_DIGEST}


@pytest.mark.parametrize(("a", "b", "same"), [
    ("a1" * 32, "a1" * 32, True), ("a1" * 32, "a1" * 6, True), ("sha256:" + "A1" * 6, "a1" * 32, True),
    ("a1" * 32, "a1" * 5, False),   # fewer than 12 digits identify nothing
    ("a1" * 32, "a2" + "a1" * 31, False), ("a1" * 6, "a1" * 6 + "ff", True), ("a1" * 6, "b2" * 6, False)])
def test_same_digest(a, b, same):
    assert model_cards.same_digest(a, b) is same
