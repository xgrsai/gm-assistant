"""SFT input builder: seeds -> frozen inputs (tiny corpus, tiny training scenario, no real data, no models)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evals.datasets import EvalDatasets
from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.game.coc7e import rules
from keeper.nodes import Nodes, narrate_messages
from keeper.rag.corpus import chunk_markdown
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from sft import build
from sft.build import BuildContext, BuildError, build_all, dice_seed, load_seeds, render_briefs, render_inputs
from sft.decontam import Decontaminator
from sft.schemas import SftInput, SftReview, SftSeed, SftTarget
from sft.split import SectionSplit, chunk_ids_hash

from .fakes import FakeFactory
from .test_evals import RULES, probe_items, qa_items
from .test_sft import adj_items

LORE = "Lviv Notes (test)"
BONUS = (RULES, "Skill Rolls > Bonus and Penalty Dice")
SANITY = (RULES, "Sanity > Temporary Insanity")          # held-out
FIREARMS = (RULES, "Combat > Firearms")
DODGE = (RULES, "Combat > Dodge")
SPELLS = (RULES, "Magic > Spells")
OPPOSED = (RULES, "Skill Rolls > Opposed Rolls")           # sibling of BONUS (same top-level heading)
STREETS = (LORE, "Lviv > Streets")


def filler(topic: str) -> str:
    return (f"{topic} is described here in a few plain sentences. The text is only long enough to be a useful "
            f"passage and says nothing about any real rule. Keep reading the {topic} notes carefully, because "
            f"they are padded to pass the minimum length of a distractor chunk.")


CORPUS_MD = f"""---
source_title: {RULES}
source_type: rules
game_system: coc7e
---
# Call of Cthulhu 7e Rules Reference
## Skill Rolls
### Bonus and Penalty Dice
A bonus die means you roll an extra tens die and keep the lowest tens result. {filler("Bonus dice")}
### Opposed Rolls
{filler("Opposed rolls")}
## Sanity
### Temporary Insanity
An investigator who loses 5 or more Sanity points in one roll must make an INT roll. {filler("Insanity")}
## Combat
### Firearms
{filler("Firearms")}
### Dodge
{filler("Dodge")}
## Magic
### Spells
{filler("Spells")}
"""

LORE_MD = f"""---
source_title: {LORE}
source_type: lore
game_system: coc7e
---
# {LORE}
## Lviv
### Streets
{filler("Streets")}
"""

SCENARIO = {
    "id": "train-cellar", "game_system": "coc7e",
    "title": {"uk": "Підвал", "en": "The Cellar"}, "era": {"uk": "Львів, 1927", "en": "Lviv, 1927"},
    "blurb": {"uk": "Тестовий сценарій.", "en": "A test scenario."},
    "opening": {"uk": "Ви стоїте перед важкими дверима підвалу. Що ви робите?",
                "en": "You stand before the heavy cellar door. What do you do?"},
    "keeper_notes": "TRUTH: the cellar hides a hole in the wall.",
    "sanity_events": {"ghoul_seen": "0/1D6"},
    "investigators": [{
        "id": "inv-1", "name": {"uk": "Ірина", "en": "Iryna"}, "occupation": {"uk": "Лікарка", "en": "Doctor"},
        "age": 30, "luck": 50,
        "characteristics": {"STR": 50, "CON": 50, "SIZ": 50, "DEX": 50, "APP": 50, "INT": 60, "POW": 60, "EDU": 70},
        "skills": {"Spot Hidden": 50, "Library Use": 60, "Dodge": 40, "Cthulhu Mythos": 0}}],
}


SANITY_REF = [{"source": SANITY[0], "section": SANITY[1]}]


def seed_rows() -> list[dict]:
    plain = {"kind": "in_fiction", "needs_roll": False, "roll_type": "none"}
    return [
        {"id": "sft-roll", "scenario_id": "train-cellar", "language": "uk", "kind": "in_fiction", "use_rag": False,
         "history": [{"role": "player", "text": "Я обережно підходжу до дверей і прислухаюся."},
                     {"role": "keeper", "text": "За дверима чути слабке капання води."}],
         "player_text": "Я присідаю й уважно оглядаю підлогу біля порога.",
         "adjudication": {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill", "target": "Spot Hidden",
                          "difficulty": "hard", "reason": "hidden traces"}},
        {"id": "sft-oracle", "scenario_id": "train-cellar", "language": "uk", "kind": "rules_question",
         "use_rag": True,
         "player_text": "Поясни, чим саме відрізняється додатковий кубик від штрафного під час перевірки?",
         "adjudication": {**plain, "kind": "rules_question"},
         "kb": {"mode": "oracle", "oracle_sections": [{"source": BONUS[0], "section": BONUS[1]}],
                "n_distractors": 2}},
        {"id": "sft-ruling", "scenario_id": "train-cellar", "language": "uk", "kind": "rules_question",
         "use_rag": True, "player_text": "Скільки коштує аренда підводу на цілий день у місті?",
         "adjudication": {**plain, "kind": "rules_question"},
         "kb": {"mode": "distractor_only", "n_distractors": 2,
                "exclude_sections": [{"source": SPELLS[0], "section": SPELLS[1]}]}},
    ]


def with_seed(**overrides) -> list[SftSeed]:
    """One variant of the first RAG seed (id sft-var) for the rejection tests."""
    row = {**seed_rows()[1], "id": "sft-var", **overrides}
    return [SftSeed.model_validate(row)]


@pytest.fixture
def chunks():
    return chunk_markdown(CORPUS_MD, source_path="t.md") + chunk_markdown(LORE_MD, source_path="l.md")


@pytest.fixture
def make_ctx(tmp_path, chunks):
    root = tmp_path / "scenarios"
    (root / "train-cellar").mkdir(parents=True)
    (root / "train-cellar" / "scenario.yaml").write_text(
        yaml.safe_dump(SCENARIO, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gold = EvalDatasets(rules_qa=qa_items(), conflict_probes=probe_items(), adjudication=adj_items(),
                        hashes={"rules_qa": "x", "conflict_probes": "y", "adjudication": "z"})
    train = [BONUS, OPPOSED, FIREARMS, DODGE, SPELLS, STREETS]
    split = SectionSplit(seed=7, method="t", corpus_chunk_ids_hash=chunk_ids_hash(chunks), gold_hashes=gold.hashes,
                         train=[{"source": s, "section": p} for s, p in train],
                         heldout=[{"source": SANITY[0], "section": SANITY[1]}])

    def make() -> BuildContext:   # a fresh context each time: nothing is cached between two builds
        return BuildContext(ScenarioRepo(root), chunks, split, Decontaminator.from_datasets(gold, split, chunks=chunks))

    return make


def seeds() -> list[SftSeed]:
    return [SftSeed.model_validate(r) for r in seed_rows()]


def problems_of(seed_list, ctx) -> str:
    with pytest.raises(BuildError) as e:
        build_all(seed_list, ctx)
    return str(e.value)


# ------------------------------------------------------------------------------------------ determinism


def test_two_builds_are_byte_identical(make_ctx):
    a, b = build_all(seeds(), make_ctx()), build_all(seeds(), make_ctx())
    assert render_inputs(a) == render_inputs(b) and render_briefs(a) == render_briefs(b)
    ids = [json.loads(line)["id"] for line in render_inputs(a).splitlines()]
    assert ids == ["sft-roll", "sft-oracle", "sft-ruling"]
    # the dice of one example do not depend on the other seeds or on their order
    reordered = build_all(list(reversed(seeds())), make_ctx())
    assert {x.input.id: x.input.rolls for x in reordered} == {x.input.id: x.input.rolls for x in a}
    assert {x.input.id: x.input.input_hash for x in reordered} == {x.input.id: x.input.input_hash for x in a}


def test_cli_writes_the_same_bytes_twice_and_check_detects_drift(make_ctx, tmp_path, monkeypatch):
    seeds_path = tmp_path / "seeds.jsonl"
    seeds_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in seed_rows()) + "\n", encoding="utf-8")
    monkeypatch.setattr(build, "load_context", lambda scenarios_dir: make_ctx())
    out, briefs = tmp_path / "inputs.jsonl", tmp_path / "out" / "briefs.jsonl"

    def run(**flags) -> int:
        return build.cmd_inputs(argparse.Namespace(seeds=seeds_path, scenarios=tmp_path, out=out, briefs=briefs,
                                                   write=flags.get("write", False), check=flags.get("check", False)))

    assert run(check=True) == 1 and not out.exists()                 # missing file differs
    assert run() == 0 and not out.exists()                           # default: validate only
    assert run(write=True) == 0
    first = out.read_bytes(), briefs.read_bytes()
    assert b"\r" not in first[0] and run(write=True) == 0 and (out.read_bytes(), briefs.read_bytes()) == first
    assert run(check=True) == 0
    out.write_bytes(first[0].replace(b"sft-roll", b"sft-rolL"))
    assert run(check=True) == 1
    lines = [SftInput.model_validate_json(x) for x in first[0].decode("utf-8").splitlines()]
    assert len(lines) == 3


# ------------------------------------------------------------------------------------------ dice and rendering


def test_rolls_are_made_by_the_rules_code_and_frozen(make_ctx):
    built = {b.input.id: b.input for b in build_all(seeds(), make_ctx())}
    roll = built["sft-roll"]
    pack = ScenarioRepo(make_ctx().scenarios.root).load("train-cellar")
    expected = rules.resolve(pack.investigator("uk"), roll.adjudication, [], seed=dice_seed("sft-roll"), turn_no=2,
                             sanity_events=pack.sanity_events)
    assert roll.turn_no == 2 and roll.dice_seed == dice_seed("sft-roll")
    assert roll.rolls == expected.records and len(roll.rolls) == 1
    assert roll.rolls[0]["target"] == "Spot Hidden" and roll.rolls[0]["value"] == 50
    assert roll.rolls[0]["difficulty"] == "hard"
    system = roll.messages[0].content
    assert json.dumps(roll.rolls, ensure_ascii=False) in system            # the narrator sees the frozen rolls
    assert built["sft-oracle"].rolls == [] and "no roll this turn" in built["sft-oracle"].messages[0].content


def test_messages_are_the_production_narrator_chat(make_ctx):
    ctx = make_ctx()
    built = {b.input.id: b.input for b in build_all(seeds(), ctx)}
    roll = built["sft-roll"]
    assert [m.role for m in roll.messages] == ["system", "user", "assistant", "user"]
    assert roll.messages[1].content == "Я обережно підходжу до дверей і прислухаюся."   # opening is trimmed off
    assert roll.messages[-1].content == roll.player_text
    assert roll.render.prompt_version == "keeper-v2" and roll.render.generator_profile["model"] == "qwen2.5:3b"
    assert "model_card" not in roll.render.generator_profile   # the frozen inputs file keeps its profile keys
    assert roll.render.prompt_hash == ctx.prompts.hash("keeper-v2") and roll.render.split_seed == 7
    assert "KNOWLEDGE BASE" not in roll.messages[0].content and roll.hits == []     # prompt-only example
    assert "demo-haunt" not in json.dumps([m.model_dump() for m in roll.messages])


def test_sanity_roll_shows_the_post_roll_sheet(make_ctx):
    row = {**seed_rows()[0], "id": "sft-san", "use_rag": False,
           "adjudication": {"kind": "in_fiction", "needs_roll": True, "roll_type": "sanity",
                            "sanity_event": "ghoul_seen", "sanity_loss": "9/9"}}     # the scenario magnitude wins
    [b] = build_all([SftSeed.model_validate(row)], make_ctx())
    rec = b.input.rolls[0]
    assert rec["notation"] == "0/1D6" and rec["san_after"] == 60 - rec["san_loss"]
    assert f'"san": {rec["san_after"]}' in b.input.messages[0].content


# ------------------------------------------------------------------------------------------ knowledge base


def test_kb_hits_look_like_retriever_hits_and_distractors_are_train_only(make_ctx, chunks):
    ctx = make_ctx()
    built = {b.input.id: b for b in build_all(seeds(), ctx)}
    oracle = built["sft-oracle"].input
    by_id = {c.id: c for c in chunks}
    assert [h.marker for h in oracle.hits] == [1, 2, 3]
    assert sorted(h.role for h in oracle.hits) == ["distractor", "distractor", "oracle"]
    assert len({h.chunk_id for h in oracle.hits}) == 3
    for h in oracle.hits:
        c = by_id[h.chunk_id]
        assert (h.source, h.section) == (c.metadata["source_title"], c.metadata["section_path"])
        assert f"[R{h.marker}] ({h.source_type}: {h.source} > {h.section}) {c.text}" in oracle.messages[0].content
        assert (h.source, h.section) in ctx.train and (h.source, h.section) != SANITY
    assert [(h.source, h.section) for h in oracle.hits if h.role == "oracle"] == [BONUS]
    assert {(s.source, s.section) for s in oracle.context_sections} == {(h.source, h.section) for h in oracle.hits}
    # rules hits come before lore hits, like the retriever's per-source-type quotas
    types = [h.source_type for h in oracle.hits]
    assert types == sorted(types, key={"rules": 0, "lore": 1}.get)
    brief = built["sft-oracle"].brief
    assert [p["role"] for p in brief["passages"]] == [h.role for h in oracle.hits]
    assert any("[R#]" in g for g in brief["guidance"])


def test_distractor_only_has_no_oracle_and_changes_with_the_id(make_ctx):
    ctx = make_ctx()
    [a] = build_all([s for s in seeds() if s.id == "sft-ruling"], ctx)
    assert a.input.oracle_sections == [] and {h.role for h in a.input.hits} == {"distractor"}
    assert any("Keeper ruling" in g for g in a.brief["guidance"])
    rows = [{**seed_rows()[2], "id": f"sft-r{i}"} for i in range(8)]
    picks = {tuple(h.chunk_id for h in b.input.hits) for b in build_all([SftSeed.model_validate(r) for r in rows], ctx)}
    assert len(picks) > 1                                                     # seeded by id, not constant


def test_hits_respect_the_retrieval_quotas(make_ctx, monkeypatch):
    two_sections = [{"source": s, "section": p} for s, p in (BONUS, FIREARMS)]
    # fixture: 4 train rules sections and 1 lore section; the rendering spec allows 4 rules + 1 lore hits
    [ok] = build_all([SftSeed.model_validate({**seed_rows()[2], "id": "sft-many",
                                              "kb": {"mode": "distractor_only", "n_distractors": 4,
                                                       "exclude_sections": SANITY_REF}})], make_ctx())
    counts = {t: [h.source_type for h in ok.input.hits].count(t) for t in ("rules", "lore")}
    assert sum(counts.values()) == 4 and counts["rules"] <= 4 and counts["lore"] <= 1
    kb = {"mode": "oracle", "oracle_sections": two_sections, "n_distractors": 4}
    assert "cannot add distractor" in problems_of(with_seed(kb=kb), make_ctx())      # only SPELLS and STREETS are left

    tight = build.render_spec

    def one_rules_hit(use_rag):
        spec = tight(use_rag)
        quotas = {"rules": 1, "lore": 1}
        return spec.model_copy(update={"retrieval": spec.retrieval.model_copy(update={"quotas": quotas})})

    monkeypatch.setattr(build, "render_spec", one_rules_hit)
    kb = {"mode": "oracle", "oracle_sections": two_sections}
    text = problems_of(with_seed(kb=kb), make_ctx())
    assert "needs 2 'rules' chunks in the prompt but the retrieval quota allows 1" in text
    kb = {"mode": "oracle", "oracle_sections": two_sections[:1], "n_distractors": 2}
    assert "no train section left within the quotas" in problems_of(with_seed(kb=kb), make_ctx())
    [b] = build_all(with_seed(kb={**kb, "n_distractors": 1}), make_ctx())           # the only free slot is the lore one
    assert [h.source_type for h in b.input.hits] == ["rules", "lore"]


# ------------------------------------------------------------------------------------------ rejections


def test_unknown_skill_is_rejected_with_the_seed_id(make_ctx):
    row = {**seed_rows()[0], "id": "sft-bad-skill"}
    row["adjudication"] = {**row["adjudication"], "target": "Astrology"}
    text = problems_of([SftSeed.model_validate(row)], make_ctx())
    assert "seed sft-bad-skill:" in text and "'Astrology'" in text and "not on the sheet" in text
    row["adjudication"] = {**row["adjudication"], "target": "spot hidden"}          # exact name, not case-folded
    assert "not on the sheet" in problems_of([SftSeed.model_validate(row)], make_ctx())
    row["adjudication"] = {**row["adjudication"], "roll_type": "characteristic", "target": "LUCK"}
    assert "not one of" in problems_of([SftSeed.model_validate(row)], make_ctx())


def test_held_out_and_unknown_oracle_sections_are_rejected(make_ctx):
    kb = {"mode": "oracle", "oracle_sections": [{"source": SANITY[0], "section": SANITY[1]}]}
    text = problems_of(with_seed(kb=kb), make_ctx())
    assert "seed sft-var:" in text and "held-out" in text
    kb = {"mode": "oracle", "oracle_sections": [{"source": RULES, "section": "Nope > Nothing"}]}
    assert "not a train section" in problems_of(with_seed(kb=kb), make_ctx())


def test_unknown_scenario_investigator_event_and_engine_errors(make_ctx):
    assert "unknown training scenario" in problems_of(with_seed(scenario_id="nope"), make_ctx())
    assert "reserved for evaluation" in problems_of(with_seed(scenario_id="demo-haunt"), make_ctx())
    assert "no investigator 'ghost'" in problems_of(with_seed(investigator_id="ghost"), make_ctx())
    san = {"kind": "in_fiction", "needs_roll": True, "roll_type": "sanity"}
    assert "sanity_event 'nope'" in problems_of(with_seed(kind="in_fiction", kb={}, use_rag=False,
                                                          adjudication={**san, "sanity_event": "nope"}), make_ctx())
    assert "rules engine rejected" in problems_of(with_seed(kind="in_fiction", kb={}, use_rag=False,
                                                            adjudication={**san, "sanity_loss": "lots"}), make_ctx())


def test_gold_copy_is_a_decontamination_failure(make_ctx):
    text = problems_of(with_seed(player_text="Які втрати на сан? Як працюють бонусні кубики?"), make_ctx())
    assert "seed sft-var: decontamination" in text
    assert "[ngram]" in text or "[subset]" in text or "[jaccard]" in text


def test_all_problems_are_reported_together(make_ctx):
    bad_skill = {**seed_rows()[0], "id": "sft-a"}
    bad_skill["adjudication"] = {**bad_skill["adjudication"], "target": "Astrology"}
    held = {**seed_rows()[1], "id": "sft-b", "kb": {"mode": "oracle", "n_distractors": 0,
                                                    "oracle_sections": [{"source": SANITY[0], "section": SANITY[1]}]}}
    e = problems_of([SftSeed.model_validate(r) for r in (bad_skill, held)], make_ctx()).splitlines()
    assert len(e) == 2 and e[0].startswith("seed sft-a:") and e[1].startswith("seed sft-b:")


def test_seed_schema_rejects_inconsistent_seeds():
    base = seed_rows()[0]

    def bad(**kw):
        return pytest.raises(ValueError, match=kw.pop("match")), {**base, **kw}

    for match, kw in [
        ("alternate", {"history": [{"role": "keeper", "text": "x"}]}),
        ("differs from kind", {"kind": "meta"}),
        ("use_rag is false", {"kb": {"mode": "distractor_only", "n_distractors": 1, "exclude_sections": SANITY_REF}}),
        ("needs a target", {"adjudication": {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill"}}),
        ("is_push", {"adjudication": {**base["adjudication"], "is_push": True}}),
        ("needs_roll must be true", {"adjudication": {"kind": "in_fiction", "needs_roll": False, "roll_type": "luck"}}),
        ("takes no roll", {"kind": "meta", "adjudication": {"kind": "meta", "needs_roll": True, "roll_type": "luck"}}),
        ("bonus_dice", {"adjudication": {**base["adjudication"], "bonus_dice": 3}}),
        ("needs oracle_sections", {"use_rag": True, "kb": {"mode": "oracle"}}),
        ("needs exclude_sections", {"use_rag": True, "kb": {"mode": "distractor_only", "n_distractors": 1}}),
        ("both an oracle and excluded", {"use_rag": True, "kb": {
            "mode": "oracle", "oracle_sections": [{"source": "a", "section": "b"}],
            "exclude_sections": [{"source": "a", "section": "b"}]}}),
        ("not NFC", {"player_text": "Я бачу кутову шафу, и" + chr(0x0306) + "ду до неї."}),
        ("trailing whitespace", {"player_text": "Я оглядаю підлогу.\n"}),
        ("blank", {"history": [{"role": "player", "text": "  "}, {"role": "keeper", "text": "x"}]}),
        ("not NFC", {"history": [{"role": "player", "text": "Я"}, {"role": "keeper", "text": "Чи" + chr(0x0306)}]}),
        ("leading or trailing", {"history": [{"role": "player", "text": "Я"}, {"role": "keeper", "text": " Так."}]}),
        ("Extra inputs", {"surprise": 1}),
    ]:
        ctx, row = bad(match=match, **kw)
        with ctx:
            SftSeed.model_validate(row)


def test_load_seeds_reports_file_positions_and_duplicates(tmp_path):
    rows = seed_rows()
    path = tmp_path / "seeds.jsonl"
    path.write_text("\n".join([json.dumps(rows[0], ensure_ascii=False), "{not json", json.dumps(rows[0]),
                               json.dumps({**rows[1], "kind": "nope"})]) + "\n", encoding="utf-8")
    with pytest.raises(BuildError) as e:
        load_seeds(path)
    text = str(e.value)
    assert "seeds.jsonl:2: invalid JSON" in text and "seed sft-oracle (seeds.jsonl:4):" in text and "kind" in text
    assert "seed sft-roll: duplicate id" in text
    with pytest.raises(BuildError, match="seeds file not found"):
        load_seeds(tmp_path / "missing.jsonl")


def test_target_and_review_schemas():
    prov = {"author": "claude", "model": "m", "created": "2026-10-23"}
    assert SftTarget.model_validate({"id": "sft-roll", "text": "Відповідь.", "provenance": prov}).input_hash is None
    with pytest.raises(ValueError, match="blank"):
        SftTarget.model_validate({"id": "x", "text": "  ", "provenance": prov})
    assert SftReview(id="x", verdict="edit", edited_text="Краще.").verdict == "edit"
    with pytest.raises(ValueError, match="needs edited_text"):
        SftReview(id="x", verdict="edit")
    with pytest.raises(ValueError, match="takes no edited_text"):
        SftReview(id="x", verdict="accept", edited_text="a")


def test_seed_files_stay_out_of_the_app_scenarios():
    assert "train-cellar" not in ScenarioRepo(Path(build.REPO_ROOT) / "data" / "scenarios").ids()
    assert build.SCENARIOS_DIR != Path(build.REPO_ROOT) / "data" / "scenarios"


# ------------------------------------------------------------------------------------------ review fixes


def test_distractors_never_come_from_excluded_sections_or_their_siblings(make_ctx):
    question = "Як працює бонусний кубик?"                      # the repro: a related section must not be a distractor
    excl = [{"source": BONUS[0], "section": BONUS[1]}]
    rows = [{**seed_rows()[2], "id": f"sft-q{i}", "player_text": question,
             "kb": {"mode": "distractor_only", "n_distractors": 3, "exclude_sections": excl}} for i in range(40)]
    built = build_all([SftSeed.model_validate(r) for r in rows], make_ctx())
    shown = {(h.source, h.section) for b in built for h in b.input.hits}
    assert shown and BONUS not in shown and OPPOSED not in shown                # the sibling is excluded as well
    assert FIREARMS in shown and STREETS in shown                               # other groups are still used
    unknown = {"mode": "distractor_only", "n_distractors": 1, "exclude_sections": [{"source": "x", "section": "y"}]}
    assert "exclude section" in problems_of(with_seed(kb=unknown), make_ctx())


def test_oracle_siblings_are_not_distractors(make_ctx):
    kb = {"mode": "oracle", "n_distractors": 3, "oracle_sections": [{"source": BONUS[0], "section": BONUS[1]}]}
    for i in range(20):
        [b] = build_all([SftSeed.model_validate({**seed_rows()[1], "id": f"sft-o{i}", "kb": kb})], make_ctx())
        assert OPPOSED not in {(h.source, h.section) for h in b.input.hits}


def test_decomposed_letters_and_outer_whitespace_are_rejected_when_seeds_are_loaded(tmp_path):
    row = {**seed_rows()[0], "id": "sft-nfc", "player_text": "Я бачу и" + chr(0x0306) + "ого."}
    path = tmp_path / "seeds.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(BuildError, match=r"seed sft-nfc \(seeds.jsonl:1\): player_text: .*not NFC"):
        load_seeds(path)
    path.write_text(json.dumps({**row, "player_text": "Я оглядаюсь.\n"}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(BuildError, match="seed sft-nfc .*leading or trailing whitespace"):
        load_seeds(path)


def test_rendered_chat_goes_through_the_training_gate(make_ctx):
    ctx = make_ctx()
    pack = SCENARIO | {"id": "train-nfc", "keeper_notes": "TRUTH: " + "и" + chr(0x0306) + " the hole."}
    (ctx.scenarios.root / "train-nfc").mkdir()
    (ctx.scenarios.root / "train-nfc" / "scenario.yaml").write_text(
        yaml.safe_dump(pack, allow_unicode=True, sort_keys=False), encoding="utf-8")
    plain = {"kind": "rules_question", "needs_roll": False, "roll_type": "none"}
    text = problems_of(with_seed(scenario_id="train-nfc", kb={}, use_rag=False, adjudication=plain), ctx)
    assert "seed sft-var: training gate:" in text and "NFC" in text


async def test_inputs_match_what_the_narrate_node_sends(make_ctx, tmp_path):
    ctx = make_ctx()
    for built in build_all(seeds(), ctx):
        fake = FakeFactory(narrations=["Відповідь."])
        fake.profile = lambda name: ctx.profile              # the base model's window, as in the build
        deps = Deps(llms=fake, prompts=ctx.prompts, scenarios=ctx.scenarios,
                    experiments=ExperimentRegistry({built.spec.arm: built.spec}, built.spec.arm),
                    runlog=RunLog(tmp_path / "runlog", enabled=False))
        runtime = SimpleNamespace(context=SimpleNamespace(spec=built.spec, campaign_id="c"),
                                  stream_writer=lambda _: None)
        await Nodes(deps).narrate(built.state, runtime)
        sent = [{"role": {"system": "system", "human": "user", "ai": "assistant"}[m.type], "content": m.text}
                for m in fake.chat_inputs[0]]
        assert sent == [m.model_dump() for m in built.input.messages], built.input.id


def test_rules_question_prompt_equals_the_eval_harness_rendering(make_ctx):
    ctx = make_ctx()
    built = next(b for b in build_all(seeds(), ctx) if b.input.id == "sft-oracle")
    forced = {"kind": "rules_question", "needs_roll": False, "roll_type": "none"}    # the harness' forced_kind form
    system = narrate_messages(ctx.prompts, ctx.profile, built.spec, {**built.state, "adjudication": forced})[0]
    assert system.content == built.input.messages[0].content


def test_siblings_are_grouped_by_parent_heading_path():
    from sft.build import _group

    lore = "Lviv 1927 Setting Notes (GM Assistant)"
    places = (lore, "Lviv in 1927: Setting Notes > Places > Beneath the Old City")
    university = (lore, "Lviv in 1927: Setting Notes > Places > The University and Its Anatomy Rooms")
    police = (lore, "Lviv in 1927: Setting Notes > People an Investigator Might Meet > Police")
    assert _group(places) == _group(university) != _group(police)
    rules = "Call of Cthulhu 7e Rules Reference (GM Assistant)"
    assert _group((rules, "Skill Rolls > Bonus and Penalty Dice")) == _group((rules, "Skill Rolls > Opposed Rolls"))
    assert _group((rules, "Luck > Luck Rolls")) != _group((rules, "Skill Rolls > Opposed Rolls"))
    assert _group((rules, "Investigator Development")) == (rules, "Investigator Development")
