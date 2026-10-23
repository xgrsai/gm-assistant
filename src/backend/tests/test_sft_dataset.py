"""SFT dataset builder: inputs + targets + review -> train/val chat files, lints, manifest, review CSV.

Tiny fixtures from tests/test_sft_build.py (tiny corpus, tiny training scenario, fake gold); no models.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import pytest

from evals.harness import KEEPER_RULING
from sft import build, dataset, gates
from sft.build import BuildError, build_all, sha16
from sft.dataset import (
    DatasetContext,
    allocate,
    assemble,
    lint_text,
    read_review_csv,
    sample_for_review,
    split_rows,
)
from sft.schemas import Message, Provenance, SftInput, SftReview, SftSeed, SftTarget

from .test_sft_build import SCENARIO, chunks, make_ctx, seed_rows  # noqa: F401  (fixtures)

PROV = Provenance(author="claude", model="test", created="2026-10-23")
GOLD_HASHES = {"rules_qa": "x", "conflict_probes": "y", "adjudication": "z"}
BOM = chr(0xFEFF)

NARRATION = (
    "Ви опускаєтесь біля порога й повільно проводите поглядом по вологих дошках підлоги. Між двома половицями ледь "
    "помітно блищить тонка смужка слизу, яка тягнеться від дверей углиб підвалу, ніби щось повільно тягли туди за "
    "собою. Запах цвілі стає гострішим, а десь за стіною знову починає падати вода, рівно й безжально, мов чийсь "
    "відлік. Ви помічаєте також подряпини на нижній петлі, свіжіші за решту, і холодний протяг, що тягне з-під "
    "дверей, несучи з собою дух старого каміння та сирої землі. Лампа в руці тремтить, і тіні на стінах починають "
    "ворушитися.")
ORACLE_TEXT = (
    "Додатковий кубик дає вам шанс на кращий результат: ви кидаєте зайвий кубик десятків і беремо до уваги найнижчий "
    "із них, а штрафний кубик працює навпаки і лишає найвищий [R{m}]. Обидва типи кубиків скасовують один одного за "
    "кількістю, тож лишається лише різниця між ними. Це правило стосується будь-якої перевірки навички, тож "
    "застосовуйте його щоразу, коли обставини допомагають або заважають. Повертаємося до сцени: двері підвалу так і "
    "стоять перед вами. Що ви робите?")
RULING_TEXT = (
    "Тут я не знайду підстави у наведених уривках, тож це рішення Хранителя: оренда підводи на день коштує "
    "недорого, достатньо домовитися з візником біля ринку. Ціни в місті трохи різняться залежно від пори дня і від "
    "того, наскільки поспішає клієнт, і торгуватися тут цілком звично. Keeper ruling: це проста побутова деталь, яка "
    "не потребує кидка. Повертаємося до сцени: ви все ще стоїте перед дверима підвалу. Що ви робите?")
PLAIN_TEXT = NARRATION + " Що ви робите?"


def many_seeds(per_kind: int = 10) -> list[SftSeed]:
    rows = seed_rows()
    return [SftSeed.model_validate({**row, "id": f"{row['id']}-{i}"}) for row in rows for i in range(per_kind)]


def outcome_words(roll: dict) -> str:
    if roll["passed"]:
        return "критичний успіх" if roll["level"] == "critical" else "успіх"
    return "фатальний провал" if roll["level"] == "fumble" else "провал"


def dice_line(inp: SftInput, marker: int | None = None) -> str:
    roll = inp.rolls[0]
    cite = f" [R{marker}]" if marker else ""
    outcome = outcome_words(roll)
    return f"🎲 Пильність (Spot Hidden): складний кидок, випало {roll['roll']}, результат — {outcome}.{cite}"


def roll_text(inp: SftInput, marker: int | None = None) -> str:
    return f"{NARRATION}\n{dice_line(inp, marker)}\nЩо ви робите?"


def good_text(inp: SftInput) -> str:
    if inp.kb_mode == "oracle":
        return ORACLE_TEXT.format(m=next(h.marker for h in inp.hits if h.role == "oracle"))
    if inp.kb_mode == "distractor_only":
        return RULING_TEXT
    return roll_text(inp)


def targets_for(inputs: list[SftInput]) -> list[SftTarget]:
    return [SftTarget(id=i.id, text=good_text(i), provenance=PROV, input_hash=i.input_hash) for i in inputs]


def review_for(target: SftTarget, verdict: str, **kw) -> SftReview:
    return SftReview(id=target.id, verdict=verdict, target_hash=sha16(target.text), **kw)


@pytest.fixture
def world(make_ctx):  # noqa: F811
    bc = make_ctx()
    built = build_all(many_seeds(), bc)
    dctx = DatasetContext(decontam=bc.decontam, prompt_hash=bc.prompts.hash("keeper-v2"), corpus_hash=bc.corpus_hash,
                          split_seed=7, split_file="data/sft/coc7e/section_split.v1.json", split_sha256="a" * 64,
                          gold_hashes=GOLD_HASHES)
    return dctx, [b.input for b in built]


def pick(inputs: list[SftInput], prefix: str) -> SftInput:
    return next(i for i in inputs if i.id.startswith(prefix))


def codes(inp: SftInput, text: str) -> list[str]:
    return [p.code for p in lint_text(inp, text)]


def with_roll(inp: SftInput, passed: bool, level: str, **extra) -> SftInput:
    return inp.model_copy(update={"rolls": [{**inp.rolls[0], "passed": passed, "level": level, **extra}]})


def oracle_with_roll(inputs: list[SftInput]) -> SftInput:
    """An in_fiction turn with a roll AND an oracle passage (the fields the citation lint reads; not re-rendered)."""
    oracle = pick(inputs, "sft-oracle")
    return pick(inputs, "sft-roll").model_copy(update={"use_rag": True, "kb_mode": "oracle", "hits": oracle.hits})


def oracle_marker(inp: SftInput) -> int:
    return next(h.marker for h in inp.hits if h.role == "oracle")


def distractor_marker(inp: SftInput) -> int:
    return next(h.marker for h in inp.hits if h.role == "distractor")


# ------------------------------------------------------------------------------------------ the good fixture


def test_good_targets_pass_every_lint(world):
    dctx, inputs = world
    out = assemble(inputs, targets_for(inputs), [], dctx)
    assert out.problems == [] and len(out.rows) == 30 and [r.inp.id for r in out.rows] == sorted(i.id for i in inputs)
    assert out.unreviewed == 30 and out.targets_without_input_hash == 0
    assert dataset.narration_words(NARRATION) >= dataset.WORD_RANGE["in_fiction"][0]
    assert {r.rolls[0]["passed"] for r in inputs if r.rolls} <= {True, False}


def test_ruling_phrase_is_a_subset_of_the_harness_regex():
    for phrase in ("Keeper ruling: ok", "Це рішення Хранителя.", "keeper RULING"):
        assert dataset.KEEPER_RULING_PHRASE.search(phrase) and KEEPER_RULING.search(phrase)
    assert not dataset.KEEPER_RULING_PHRASE.search("за рішенням Хранителя")         # only the nominative counts
    assert dataset.KEEPER_RULING_ANY.search("за рішенням Хранителя")


# ------------------------------------------------------------------------------------------ numbers


def test_number_lint(world):
    _, inputs = world
    roll = pick(inputs, "sft-roll")
    assert "number" not in codes(roll, roll_text(roll))
    roll_value = str(roll.rolls[0]["roll"])
    assert "number" not in codes(roll, roll_text(roll).replace("Лампа в руці", f"Лампа з номером {roll_value} в руці"))
    bad = lint_text(roll, roll_text(roll).replace("Що ви робите?", "У вас є 77 хвилин. Що ви робите?"))
    assert [p.code for p in bad if p.code == "number"] and "77" in bad[0].reason
    # the edition number is no longer whitelisted (this sheet happens to contain a 7), the percentile die is
    assert dataset.NUMBER_WHITELIST == {"100"}
    assert "number" in codes(roll, roll_text(roll).replace("Що ви робите?", "Це правила 13-ї редакції. Що ви робите?"))
    assert "number" not in codes(roll, roll_text(roll).replace("Що ви робите?", "Кидок d100 вирішує. Що ви робите?"))
    # [R#] digits are not numbers
    oracle = pick(inputs, "sft-oracle")
    assert "number" not in codes(oracle, good_text(oracle))
    assert "number" in codes(oracle, good_text(oracle).replace("Що ви робите?", "Це 3 кроки. Що ви робите?"))


def test_the_era_decade_is_allowed(world):
    _, inputs = world
    roll = pick(inputs, "sft-roll")
    assert {"1927", "1920"} <= dataset.allowed_numbers(roll)
    text = roll_text(roll).replace("Лампа в руці", "Лампа, як у 1920-ті, в руці")
    assert "number" not in codes(roll, text)
    assert "number" in codes(roll, text.replace("1920", "1910"))


def test_numbers_come_from_oracle_passages_but_not_from_distractors(world):
    _, inputs = world
    oracle = pick(inputs, "sft-oracle")
    system = oracle.messages[0].content

    def planted(marker: int, fact: str) -> SftInput:
        start = system.index(f"[R{marker}] (", system.index(dataset.KB_MARKER))
        at = system.index(") ", start) + 2
        edited = system[:at] + fact + " " + system[at:]
        return oracle.model_copy(update={"messages": [Message(role="system", content=edited), *oracle.messages[1:]]})

    text = good_text(oracle).replace("Що ви робите?", "Рахунок {n}. Що ви робите?")
    in_oracle = planted(oracle_marker(oracle), "Fact 4242.")
    in_distractor = planted(distractor_marker(oracle), "Fact 3131.")
    assert "4242" in dataset.allowed_numbers(in_oracle) and "3131" not in dataset.allowed_numbers(in_oracle)
    assert "3131" not in dataset.allowed_numbers(in_distractor)                 # the distractor is not a source
    assert "number" not in codes(in_oracle, text.format(n="4242"))
    assert "number" in codes(in_distractor, text.format(n="3131"))
    assert "number" in codes(in_oracle, text.format(n="3131"))


def test_the_dice_line_allows_only_the_roll_value(world):
    _, inputs = world
    roll = pick(inputs, "sft-roll")
    line = dice_line(roll)
    assert "number" not in codes(roll, roll_text(roll))
    extra = roll_text(roll).replace(line, line[:-1] + ", поріг 25.")
    bad = [p for p in lint_text(roll, extra) if p.code == "number"]
    assert bad and "25" in bad[0].reason and "FORMAT" in bad[0].reason
    no_roll_value = roll_text(roll).replace(f"випало {roll.rolls[0]['roll']}, ", "")
    assert "number" not in codes(roll, no_roll_value)                         # the value is allowed, not required


# ------------------------------------------------------------------------------------------ dice line, outcome


def test_dice_line_present_if_and_only_if_there_is_a_roll(world):
    _, inputs = world
    roll, oracle = pick(inputs, "sft-roll"), pick(inputs, "sft-oracle")
    no_dice = "\n".join(ln for ln in roll_text(roll).splitlines() if not ln.startswith("🎲"))
    assert "dice-line" in codes(roll, no_dice)
    assert "dice-line" in codes(roll, roll_text(roll) + "\n🎲 Ще один рядок.")
    assert "dice-line" in codes(roll, "🎲 Один рядок на початку. " + roll_text(roll))          # two dice characters
    assert "dice-line" in codes(oracle, good_text(oracle) + "\n🎲 Кидка не було.")
    assert "dice-line" not in codes(oracle, good_text(oracle)) and "dice-line" not in codes(roll, roll_text(roll))


def with_result(inp: SftInput, result: str) -> str:
    """The roll turn with `result` as the words after «результат —» on the 🎲 line."""
    roll = inp.rolls[0]
    line = f"🎲 Пильність (Spot Hidden): складний кидок, випало {roll['roll']}, результат — {result}."
    return f"{NARRATION}\n{line}\nЩо ви робите?"


def test_outcome_lint_checks_every_word_against_the_record(world):
    _, inputs = world
    base = pick(inputs, "sft-roll")
    failed, passed = with_roll(base, False, "failure"), with_roll(base, True, "regular", difficulty="regular")
    ok_fail, ok_pass = roll_text(failed), roll_text(passed)
    assert "outcome" not in codes(failed, ok_fail) and "outcome" not in codes(passed, ok_pass)
    # the wrong outcome
    assert "outcome" in codes(failed, ok_fail.replace("провал", "успіх"))
    assert "outcome" in codes(passed, ok_pass.replace("успіх", "провал"))
    # the opposite word next to the right one
    assert "outcome" in codes(failed, ok_fail.replace("результат — провал", "результат — провал, але успіх"))
    assert "outcome" in codes(passed, ok_pass.replace("результат — успіх", "результат — успіх, але провал"))
    # no outcome word at all, or only a negated failure instead of the success word
    assert "outcome" in codes(failed, ok_fail.replace("результат — провал", "результат відомий"))
    assert "outcome" in codes(passed, ok_pass.replace("результат — успіх", "результат — не провал"))
    # English vocabulary
    en = ok_fail.replace("складний кидок", "hard check").replace("провал", "failure")
    assert "outcome" not in codes(failed, en) and "outcome" in codes(failed, en.replace("failure", "success"))


@pytest.mark.parametrize("result", ["без успіху", "успіху немає", "no success", "майже успіх", "not a success",
                                    "кидок невдалий, але це успіх"])
def test_a_passed_roll_described_with_a_negated_success_is_an_error(world, result):
    _, inputs = world
    passed = with_roll(pick(inputs, "sft-roll"), True, "regular", difficulty="regular")
    assert "outcome" in codes(passed, with_result(passed, result))


@pytest.mark.parametrize("result", ["not a failure", "провалу не сталося", "уникнув провалу"])
def test_a_failed_roll_described_with_a_negated_failure_is_an_error(world, result):
    _, inputs = world
    failed = with_roll(pick(inputs, "sft-roll"), False, "failure")
    assert "outcome" in codes(failed, with_result(failed, result))


@pytest.mark.parametrize("result", ["невдалий кидок", "unsuccessful", "the roll fails", "не  вдалося",
                                    "кидок не вдався", "невдача", "failure", "провал"])
def test_failures_in_other_words_are_accepted_for_a_failed_roll(world, result):
    _, inputs = world
    failed = with_roll(pick(inputs, "sft-roll"), False, "failure")
    assert "outcome" not in codes(failed, with_result(failed, result))
    assert "outcome" in codes(failed, with_result(failed, result + ", але це успіх"))


@pytest.mark.parametrize("result", ["you succeed", "удалося", "успіх, провалу вдалося уникнути", "вдалий кидок",
                                    "успіх, а не провал", "success"])
def test_successes_in_other_words_are_accepted_for_a_passed_roll(world, result):
    _, inputs = world
    passed = with_roll(pick(inputs, "sft-roll"), True, "regular", difficulty="regular")
    assert "outcome" not in codes(passed, with_result(passed, result))


# (difficulty of the check, rolled level, phrase after «результат —», is it accepted?)
LEVEL_TABLE = [
    ("hard", "hard", "складний успіх", True), ("hard", "hard", "успіх", True), ("hard", "hard", "hard success", True),
    ("hard", "hard", "звичайний успіх", False),        # below the level the check required
    ("hard", "hard", "regular success", False), ("hard", "hard", "екстремальний успіх", False),  # above the roll
    ("hard", "hard", "extreme success", False), ("hard", "hard", "критичний успіх", False),
    ("regular", "regular", "звичайний успіх", True), ("regular", "regular", "регулярний успіх", True),
    ("regular", "regular", "складний успіх", False),
    ("regular", "hard", "складний успіх", True), ("regular", "hard", "звичайний успіх", True),  # down to the required
    ("regular", "hard", "екстремальний успіх", False),
    ("hard", "extreme", "екстремальний успіх", True), ("hard", "extreme", "extreme success", True),
    ("hard", "extreme", "успіх", False),               # extreme on a check that was not extreme must be named
    ("hard", "extreme", "складний успіх", False),
    ("extreme", "extreme", "успіх", True), ("extreme", "extreme", "екстремальний успіх", True),
    ("extreme", "extreme", "складний успіх", False),   # below the required level of an extreme check
    ("regular", "critical", "критичний успіх", True), ("regular", "critical", "critical success", True),
    ("regular", "critical", "успіх", False), ("regular", "critical", "складний успіх", False),
]


@pytest.mark.parametrize(("difficulty", "level", "result", "accepted"), LEVEL_TABLE)
def test_a_named_level_of_success_must_fit_the_roll(world, difficulty, level, result, accepted):
    _, inputs = world
    inp = with_roll(pick(inputs, "sft-roll"), True, level, difficulty=difficulty)
    found = lint_text(inp, with_result(inp, result))
    assert ("outcome" not in [p.code for p in found]) is accepted, [p.reason for p in found]


@pytest.mark.parametrize(("level", "result", "accepted"), [
    ("regular", "звичайний успіх, але для складного кидка замало — провал", True),
    ("regular", "regular success, but not enough for a hard check: failure", True),
    ("hard", "складний успіх, але для екстремального кидка замало — провал", True),
    ("regular", "складний успіх, але провал", False),          # the roll did not reach hard
    ("failure", "звичайний успіх, але провал", False),         # nothing was reached
    ("regular", "звичайний успіх", False),                     # a failure must still be stated
    ("failure", "провал", True)])
def test_a_level_named_to_explain_a_failure_is_not_a_claim_of_success(world, level, result, accepted):
    _, inputs = world
    failed = with_roll(pick(inputs, "sft-roll"), False, level, difficulty="hard" if level != "hard" else "extreme")
    found = lint_text(failed, with_result(failed, result))
    assert ("outcome" not in [p.code for p in found]) is accepted, [p.reason for p in found]


def test_difficulty_phrases_are_not_levels_of_success(world):
    _, inputs = world
    easy = with_roll(pick(inputs, "sft-roll"), True, "regular", difficulty="regular")
    text = with_result(easy, "успіх").replace("складний кидок", "складна перевірка, Hard difficulty, hard check")
    assert "outcome" not in codes(easy, text)
    hard = with_roll(easy, True, "hard", difficulty="hard")
    assert "outcome" not in codes(hard, text)
    failed = with_roll(easy, False, "failure", difficulty="extreme")
    text = with_result(failed, "провал").replace("складний кидок", "екстремальна перевірка")
    assert "outcome" not in codes(failed, text)


def test_critical_and_fumble_need_their_specific_word(world):
    _, inputs = world
    base = pick(inputs, "sft-roll")
    fumble, critical = with_roll(base, False, "fumble"), with_roll(base, True, "critical")
    assert "outcome" not in codes(fumble, roll_text(fumble)) and "фатальний провал" in roll_text(fumble)
    assert "outcome" in codes(fumble, roll_text(fumble).replace("фатальний провал", "провал"))
    assert "outcome" not in codes(fumble, roll_text(fumble).replace("фатальний провал", "fumble"))
    assert "outcome" in codes(critical, roll_text(critical).replace("критичний успіх", "успіх"))
    assert "outcome" not in codes(critical, roll_text(critical).replace("критичний успіх", "critical success"))
    # ... and may not be claimed for another level
    ordinary_fail, ordinary_pass = with_roll(base, False, "failure"), with_roll(base, True, "regular")
    assert "outcome" in codes(ordinary_fail, roll_text(ordinary_fail).replace("провал", "фатальний провал"))
    assert "outcome" in codes(ordinary_pass, roll_text(ordinary_pass).replace("успіх", "критичний успіх"))
    assert "outcome" not in codes(with_roll(base, False, "failure", difficulty="extreme"),
                                  roll_text(with_roll(base, False, "failure")))


def test_a_sanity_loss_stated_on_the_line_must_match(make_ctx):  # noqa: F811
    row = {**seed_rows()[0], "id": "sft-san", "use_rag": False,
           "adjudication": {"kind": "in_fiction", "needs_roll": True, "roll_type": "sanity",
                            "sanity_event": "ghoul_seen", "sanity_loss": "9/9"}}
    [built] = build_all([SftSeed.model_validate(row)], make_ctx())
    san = built.input
    assert len(san.rolls) == 1                                   # no INT check for a loss under 5
    rec = san.rolls[0]
    line = (f"🎲 Розсудливість (SAN): випало {rec['roll']}, результат — {outcome_words(rec)}, "
            f"втрата {rec['san_loss']} пунктів.")
    text = f"{NARRATION}\n{line}\nЩо ви робите?"
    assert [c for c in codes(san, text) if c in ("outcome", "number")] == []
    wrong = text.replace(f"втрата {rec['san_loss']} пунктів", f"втрата {rec['san_loss'] + 1} пунктів")
    found = codes(san, wrong)
    assert "outcome" in found and "number" in found
    no_loss = text.replace(f", втрата {rec['san_loss']} пунктів", "")
    assert [c for c in codes(san, no_loss) if c in ("outcome", "number")] == []

    def sanity_reasons(roll_value: int, loss: int, line_text: str) -> list[str]:
        inp = san.model_copy(update={"rolls": [{**rec, "roll": roll_value, "san_loss": loss, "passed": True,
                                                "level": "regular"}]})
        return [p.reason for p in lint_text(inp, f"{NARRATION}\n🎲 {line_text}\nЩо ви робите?")
                if p.code == "outcome" and "SAN loss" in p.reason]

    # the roll value after «втрата … —» is not a SAN loss: the number must follow «втрата» directly
    assert sanity_reasons(45, 0, "Розсудливість: втрата Розсудливості — кидок 45, успіх.") == []
    assert sanity_reasons(45, 0, "Розсудливість: втрата 0 пунктів, випало 45, успіх.") == []
    assert sanity_reasons(45, 0, "Розсудливість: втрата Розсудливості — 3, успіх.") != []
    assert sanity_reasons(45, 2, "Розсудливість: lost 3 SAN, rolled 45, success.") != []
    assert sanity_reasons(45, 2, "Розсудливість: втрата: 2, випало 45, успіх.") == []


# ------------------------------------------------------------------------------------------ citations


def test_citations_in_a_rules_question_with_an_oracle_passage(world):
    _, inputs = world
    oracle = pick(inputs, "sft-oracle")
    marker, distractor = oracle_marker(oracle), distractor_marker(oracle)
    text = good_text(oracle)
    assert "citation" not in codes(oracle, text)
    assert "citation" not in codes(oracle, text + f" [R{marker}, R{marker}]")
    problems = {p.reason for p in lint_text(oracle, text.replace(f"[R{marker}]", "[R9]")) if p.code == "citation"}
    assert any("not in the prompt" in r for r in problems) and any("no oracle passage" in r for r in problems)
    assert any("distractor" in p.reason for p in lint_text(oracle, text + f" [R{distractor}]"))
    assert any("no oracle passage" in p.reason for p in lint_text(oracle, text.replace(f" [R{marker}]", "")))
    assert "citation" in codes(oracle, text.replace(f"[R{marker}]", f"[R{marker}; R{distractor}]"))
    assert "citation-placement" not in codes(oracle, text)       # a rules answer is out of character: anywhere is fine


def test_citations_in_a_meta_turn_are_allowed_but_not_required(world):
    _, inputs = world
    oracle = pick(inputs, "sft-oracle")
    meta = oracle.model_copy(update={"kind": "meta"})
    marker, distractor = oracle_marker(oracle), distractor_marker(oracle)
    text = good_text(oracle)
    assert "citation" not in codes(meta, text)
    assert "citation" not in codes(meta, text.replace(f" [R{marker}]", ""))
    assert "citation" in codes(meta, text + f" [R{distractor}]")
    assert "citation" in codes(meta, text.replace(f"[R{marker}]", "[R9]"))


def test_citations_in_an_in_character_turn_with_a_roll_go_on_the_dice_line_only(world):
    _, inputs = world
    roll = oracle_with_roll(inputs)
    marker = oracle_marker(roll)
    on_line = roll_text(roll, marker)
    assert codes(roll, on_line).count("citation") == 0 and "citation-placement" not in codes(roll, on_line)
    in_narration = roll_text(roll).replace("Лампа в руці", f"Лампа [R{marker}] в руці")
    found = codes(roll, in_narration)
    assert "citation-placement" in found and "citation" in found            # misplaced, and the 🎲 line has none
    both = roll_text(roll, marker).replace("Лампа в руці", f"Лампа [R{marker}] в руці")
    assert "citation-placement" in codes(roll, both)
    assert any("required" not in p.reason and "no oracle passage is cited" in p.reason
               for p in lint_text(roll, roll_text(roll)) if p.code == "citation")      # a roll needs the citation
    distractor = distractor_marker(roll)
    assert "citation" in codes(roll, roll_text(roll, distractor))


def test_no_markers_at_all_in_an_in_character_turn_without_a_roll(world):
    _, inputs = world
    oracle = pick(inputs, "sft-oracle")
    scene = oracle.model_copy(update={"kind": "in_fiction"})
    marker = oracle_marker(oracle)
    assert [c for c in codes(scene, PLAIN_TEXT) if c.startswith("citation")] == []
    found = codes(scene, PLAIN_TEXT.replace("Лампа в руці", f"Лампа [R{marker}] в руці"))
    assert "citation-placement" in found


def test_no_citations_without_an_answering_passage(world):
    _, inputs = world
    roll, ruling = pick(inputs, "sft-roll"), pick(inputs, "sft-ruling")
    assert "citation" in codes(roll, roll_text(roll).replace("Що ви робите?", "Див. [R1]. Що ви робите?"))
    assert "citation" in codes(ruling, RULING_TEXT + " [R1]")
    none_mode = ruling.model_copy(update={"kb_mode": "none", "hits": []})
    assert "citation" in codes(none_mode, RULING_TEXT + " [R1]")


# ------------------------------------------------------------------------------------------ Keeper ruling


def test_ruling_is_required_only_for_rules_questions_the_kb_does_not_cover(world):
    _, inputs = world
    ruling = pick(inputs, "sft-ruling")
    none_mode = ruling.model_copy(update={"kb_mode": "none", "hits": []})
    no_phrase = RULING_TEXT.replace("рішення Хранителя", "міркування").replace("Keeper ruling", "Висновок")
    for inp in (ruling, none_mode):
        assert "ruling" in codes(inp, no_phrase)
        assert "ruling" not in codes(inp, RULING_TEXT)
        assert "ruling" not in codes(inp, no_phrase.replace("Висновок", "Рішення Хранителя"))
        # only the nominative form is the explicit phrase the prompt asks for
        assert "ruling" in codes(inp, no_phrase.replace("Висновок", "За рішенням Хранителя"))


def test_ruling_is_forbidden_everywhere_else(world):
    _, inputs = world
    oracle, ruling, roll = pick(inputs, "sft-oracle"), pick(inputs, "sft-ruling"), pick(inputs, "sft-roll")
    assert "ruling" in codes(oracle, good_text(oracle) + " Але це Keeper ruling.")      # the KB answers: cite instead
    for kind in ("in_fiction", "meta"):                                                 # not a rules question
        other = ruling.model_copy(update={"kind": kind})
        assert "ruling" in codes(other, PLAIN_TEXT + " Це Keeper ruling.")
        assert "ruling" in codes(other, PLAIN_TEXT + " Це рішення Хранителя.")
        assert "ruling" in codes(other, PLAIN_TEXT + " Так вирішено за рішенням Хранителя.")
        assert "ruling" not in codes(other, PLAIN_TEXT)
        # a scene's own «рішення хранителя архіву» (lower case: not the Keeper) is not the phrase
        assert "ruling" not in codes(other, PLAIN_TEXT + " Це було рішення хранителя архіву.")
    assert "ruling" in codes(roll, roll_text(roll).replace("Що ви робите?", "Keeper ruling. Що ви робите?"))
    # ... and the required phrase is the Keeper's, not the archive keeper's
    lower = RULING_TEXT.replace("рішення Хранителя", "рішення хранителя").replace("Keeper ruling", "Висновок")
    assert "ruling" in codes(ruling, lower)
    no_rag_question = oracle.model_copy(update={"use_rag": False, "kb_mode": "none", "hits": []})
    assert "ruling" in codes(no_rag_question, RULING_TEXT)                   # rule 6 is not rendered without RAG
    assert "ruling" not in codes(no_rag_question, RULING_TEXT.replace("рішення Хранителя", "думку")
                                 .replace("Keeper ruling", "Висновок"))


def test_guidance_matches_the_lints(world):
    _, inputs = world
    oracle, ruling, roll = pick(inputs, "sft-oracle"), pick(inputs, "sft-ruling"), pick(inputs, "sft-roll")

    def say(inp: SftInput) -> str:
        return "\n".join(build.guidance(inp))

    assert "(required) in the rules answer" in say(oracle) and "Do not say \"Keeper ruling\"" in say(oracle)
    assert "not required" in say(oracle.model_copy(update={"kind": "meta"}))
    scene = oracle_with_roll(inputs)
    assert "on the 🎲 line ONLY" in say(scene) and "critical" in say(scene) and "fumble" in say(scene)
    assert "No [R#] markers at all" in say(oracle.model_copy(update={"kind": "in_fiction"}))
    assert "nominative" in say(ruling) and "do not cite [R#]" in say(ruling)
    for kind in ("in_fiction", "meta"):
        assert "Do not say \"Keeper ruling\"" in say(ruling.model_copy(update={"kind": kind}))
    assert "no [R#] markers and no \"Keeper ruling\"" in say(roll) and "ONE short" in say(roll)
    for word in ("«успіх»", "«провал»", "«критичний успіх»", "«фатальний провал»", "«екстремальний успіх»",
                 "«складний»", "required level", "«втрата»", "d100 value"):     # the words the outcome lint accepts
        assert word in say(roll), word
    assert "never state a rule number" in say(roll)


# ------------------------------------------------------------------------------------------ the rest of the lints


def test_language_lint(world):
    _, inputs = world
    roll = pick(inputs, "sft-roll")
    english = ("You crouch by the threshold and slowly run your eyes over the damp boards of the floor. Between two "
               "planks a thin streak of slime gleams, running from the door deep into the cellar as if something had "
               "been dragged that way. The smell of mould grows sharper, and somewhere behind the wall water starts "
               "to drip again, evenly and mercilessly, like somebody counting. A cold draught creeps from under the "
               "door, carrying the breath of old stone and damp earth, and the lamp in your hand begins to tremble, "
               "so that the shadows on the walls start to stir.\n{line}\nWhat do you do?")
    line = dice_line(roll).split(":")[0] + f": hard check, rolled {roll.rolls[0]['roll']}, " + (
        "success." if roll.rolls[0]["passed"] else "failure.")
    english = english.format(line=line)
    assert "language" in codes(roll, english)
    assert "language" not in codes(roll, roll_text(roll))
    as_english = roll.model_copy(update={"language": "en"})
    assert "language" in codes(as_english, roll_text(roll)) and "language" not in codes(as_english, english)


def test_length_lint(world):
    _, inputs = world
    roll, oracle = pick(inputs, "sft-roll"), pick(inputs, "sft-oracle")
    assert dataset.WORD_RANGE["in_fiction"] == (70, 220)
    assert "length" in codes(roll, f"Ви бачите підлогу.\n{dice_line(roll)}\nЩо ви робите?")
    narration_text = NARRATION + " "
    assert "length" in codes(roll, narration_text * 3 + f"\n{dice_line(roll)}\nЩо ви робите?")        # too long
    assert "length" in codes(roll, " ".join(NARRATION.split()[:60]) + f"\n{dice_line(roll)}")       # 60 < 70 words
    assert "length" not in codes(roll, roll_text(roll))
    assert "length" in codes(oracle, "Так, [R1].")
    assert "🎲" not in dataset.narration(roll_text(roll)) and dataset.narration_words(roll_text(roll)) > 70


def test_token_estimate_lint(world, monkeypatch):
    _, inputs = world
    roll = pick(inputs, "sft-roll")
    text = roll_text(roll)
    est = dataset.estimate_tokens(roll, text)
    assert est > len(roll.messages[0].content) / 2.8 and "tokens" not in codes(roll, text)
    monkeypatch.setattr(dataset, "MAX_EST_TOKENS", est - 1)
    assert "tokens" in codes(roll, text)
    monkeypatch.setattr(dataset, "MAX_EST_TOKENS", est)
    assert "tokens" not in codes(roll, text)
    english = roll.model_copy(update={"language": "en"})                 # 4.0 chars per token vs 2.8
    assert dataset.estimate_tokens(english, text) < est


def test_training_gate_lint(world):
    _, inputs = world
    roll = pick(inputs, "sft-roll")
    text = roll_text(roll)
    assert "gate" in codes(roll, text + "\n")
    assert "gate" in codes(roll, " " + text)
    assert "gate" in codes(roll, text.replace("Ви помічаєте", "Ви и" + chr(0x0306) + " помічаєте"))


def test_decontamination_runs_on_the_target(world):
    dctx, inputs = world
    roll = pick(inputs, "sft-roll")
    leak = roll_text(roll).replace("Що ви робите?", "Як працюють бонусні кубики? Що ви робите?")
    targets = targets_for(inputs)
    targets = [t.model_copy(update={"text": leak}) if t.id == roll.id else t for t in targets]
    out = assemble(inputs, targets, [], dctx)
    assert any(p.startswith(f"{roll.id}: [decontam:") and "target" in p for p in out.problems)


def test_stale_and_foreign_inputs(world):
    dctx, inputs = world
    targets = targets_for(inputs)
    roll = pick(inputs, "sft-roll")
    stale = [t.model_copy(update={"input_hash": "0" * 16}) if t.id == roll.id else t for t in targets]
    assert any(p.startswith(f"{roll.id}: [stale]") for p in assemble(inputs, stale, [], dctx).problems)
    no_hash = [t.model_copy(update={"input_hash": None}) for t in targets]
    out = assemble(inputs, no_hash, [], dctx)                                          # an error by default ...
    assert sum("[no-input-hash]" in p for p in out.problems) == 30 and out.targets_without_input_hash == 30
    out = assemble(inputs, no_hash, [], dctx, allow_no_input_hash=True)                # ... unless allowed
    assert out.problems == [] and out.targets_without_input_hash == 30
    other_prompt = DatasetContext(**{**dctx.__dict__, "prompt_hash": "f" * 16})
    assert any("[input]" in p and "prompt" in p for p in assemble(inputs, targets, [], other_prompt).problems)
    other_corpus = DatasetContext(**{**dctx.__dict__, "corpus_hash": "f" * 16})
    assert any("[input]" in p and "corpus" in p for p in assemble(inputs, targets, [], other_corpus).problems)
    tampered = [i.model_copy(update={"input_hash": "0" * 16}) if i.id == roll.id else i for i in inputs]
    assert any("[input]" in p and "input_hash" in p for p in assemble(tampered, targets, [], dctx).problems)


def test_missing_orphan_and_duplicate_records(world):
    dctx, inputs = world
    targets = targets_for(inputs)
    out = assemble(inputs, targets[1:], [], dctx)
    assert out.problems == [f"{inputs[0].id}: no target"] and out.missing == [inputs[0].id]
    assert assemble(inputs, targets[1:], [], dctx, allow_missing=True).problems == []
    orphan = SftTarget(id="sft-nope", text="x", provenance=PROV)
    text = "\n".join(assemble(inputs, [*targets, orphan, targets[0]], [SftReview(id="sft-nah", verdict="reject")],
                              dctx).problems)
    assert "sft-nope: target for an unknown input id" in text
    assert f"{targets[0].id}: duplicate id in targets" in text
    assert "sft-nah: review for an id without a target" in text


# ------------------------------------------------------------------------------------------ review


def test_review_verdicts(world):
    dctx, inputs = world
    targets = targets_for(inputs)
    by_id = {t.id: t for t in targets}
    a, b, c = (inputs[0].id, inputs[1].id, inputs[2].id)
    better = by_id[c].text.replace("Запах цвілі", "Різкий запах цвілі")
    reviews = [review_for(by_id[a], "reject", note="meh"), review_for(by_id[b], "accept"),
               review_for(by_id[c], "edit", edited_text=better)]
    out = assemble(inputs, targets, reviews, dctx)
    assert out.problems == [] and out.rejected == [a] and out.edited == [c] and out.unreviewed == 27
    rows = {r.inp.id: r for r in out.rows}
    assert a not in rows and rows[b].origin == "target"
    assert rows[c].origin == "edited" and rows[c].text == better != by_id[c].text
    assert rows[c].target_text == by_id[c].text
    # an edit has to pass all lints too
    worse = review_for(by_id[c], "edit", edited_text=better.replace("Що ви робите?", "Є 77 шляхів. Що ви робите?"))
    bad = assemble(inputs, targets, [worse], dctx).problems
    assert len(bad) == 1 and bad[0].startswith(f"{c}: [number]")
    # a rejected example is not linted
    broken = [t.model_copy(update={"text": "x"}) if t.id == a else t for t in targets]
    assert assemble(inputs, broken, [review_for(broken[0], "reject")], dctx).problems == []
    assert any(p.startswith(f"{a}:") for p in assemble(inputs, broken, [review_for(broken[0], "accept")],
                                                      dctx).problems)


def test_edited_text_from_the_reviews_file_is_nfc_normalised(world):
    dctx, inputs = world
    targets = targets_for(inputs)
    target = targets[0]
    decomposed = target.text.replace(" й ", " и" + chr(0x0306) + " ", 1)
    assert decomposed != target.text and "и" + chr(0x0306) in decomposed
    out = assemble(inputs, targets, [review_for(target, "edit", edited_text=decomposed)], dctx)
    assert out.problems == []
    [row] = [r for r in out.rows if r.inp.id == target.id]
    assert row.text == target.text.replace(" й ", " й ", 1) and "и" + chr(0x0306) not in row.text
    assert gates.check_messages([dataset.chat_rows(row.inp, row.text)], "x") is None


def test_only_the_verdict_for_the_current_target_text_applies(world):
    dctx, inputs = world
    targets = targets_for(inputs)
    a, b = targets[0], targets[1]
    better = b.text.replace("Запах цвілі", "Різкий запах цвілі")
    rewritten = [t.model_copy(update={"text": t.text + " Тіні ворушаться."}) if t.id in (a.id, b.id) else t
                 for t in targets]
    old = [review_for(a, "reject"), review_for(b, "edit", edited_text=better)]
    assert assemble(inputs, targets, old, dctx).problems == []                         # current text: applied
    out = assemble(inputs, rewritten, old, dctx)
    assert out.problems == []                                                          # not an error any more
    assert out.rewritten == [a.id, b.id] and out.rejected == [] and out.edited == []
    assert out.unreviewed == 30
    rows = {r.inp.id: r for r in out.rows}
    assert rows[a.id].rewritten and rows[a.id].text == rewritten[0].text and rows[a.id].verdict == ""
    assert not rows[inputs[2].id].rewritten
    # a new verdict for the new text joins the history and takes over; the old ones stay as history
    new_a = rewritten[0]
    out = assemble(inputs, rewritten, [*old, review_for(new_a, "reject", note="still bad")], dctx)
    assert out.problems == [] and out.rejected == [a.id] and out.rewritten == [b.id]
    assert [r.inp.id for r in out.rejected_rows] == [a.id] and a.id not in {r.inp.id for r in out.rows}
    # two verdicts for the same text are ambiguous
    twice = assemble(inputs, rewritten, [review_for(new_a, "reject"), review_for(new_a, "accept")], dctx)
    assert twice.problems == [f"{a.id}: duplicate review for the same target text"]
    # a review without a hash is applied whatever the text, and counted
    unbound = SftReview(id=a.id, verdict="reject")
    out = assemble(inputs, rewritten, [unbound], dctx)
    assert out.rejected == [a.id] and out.reviews_without_target_hash == 1 and out.rewritten == []


# ------------------------------------------------------------------------------------------ split, files, manifest


def build_args(tmp_path: Path, inputs, targets, reviews=(), **flags) -> argparse.Namespace:
    def dump(name: str, rows) -> Path:
        path = tmp_path / name
        path.write_text("".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
        return path

    return argparse.Namespace(
        version="v1", inputs=dump("inputs.jsonl", inputs), targets=dump("targets.jsonl", targets),
        reviews=dump("reviews.jsonl", reviews) if reviews else tmp_path / "none.jsonl",
        out_dir=flags.pop("out_dir", tmp_path / "out"), lint_only=False, allow_missing=False,
        allow_no_input_hash=False, **flags)


@pytest.fixture
def patched(world, monkeypatch):
    dctx, inputs = world
    monkeypatch.setattr(dataset, "load_dataset_context", lambda: dctx)
    monkeypatch.setattr(dataset, "git_sha", lambda: "abc123")
    monkeypatch.setattr(dataset, "git_dirty", lambda: False)
    return dctx, inputs


def test_split_is_stratified_by_kind_and_by_hash_of_the_id(world):
    dctx, inputs = world
    rows = assemble(inputs, targets_for(inputs), [], dctx).rows
    train, val = split_rows(rows)
    assert len(train) + len(val) == 30 and not {r.inp.id for r in train} & {r.inp.id for r in val}
    kinds = [r.inp.kind for r in val]
    assert kinds.count("in_fiction") == 1 and kinds.count("rules_question") == 2
    again_train, again_val = split_rows(list(reversed(rows)))                  # the file order does not matter
    assert [r.inp.id for r in again_val] == [r.inp.id for r in val]
    assert {r.inp.id for r in again_train} == {r.inp.id for r in train}


def test_translation_pairs_land_on_the_same_side_of_the_split(world):
    dctx, inputs = world
    base = assemble(inputs, targets_for(inputs), [], dctx).rows
    template = next(r for r in base if r.inp.kind == "in_fiction")
    rows = [dataclasses.replace(template, inp=template.inp.model_copy(update={"id": f"scene{n:02d}-{lang}"}))
            for n in range(20) for lang in ("uk", "en")]
    train, val = split_rows(rows)
    assert dataset.group_key("scene03-uk") == dataset.group_key("scene03-en") == "scene03"
    assert dataset.group_key("sft-roll-3") == "sft-roll-3"                     # no suffix, no pairing
    val_groups = {dataset.group_key(r.inp.id) for r in val}
    assert val_groups and len(val) == 2 * len(val_groups) == 4                 # 10% of 40 examples = 2 pairs
    assert not val_groups & {dataset.group_key(r.inp.id) for r in train}
    assert [r.inp.id for r in split_rows(list(reversed(rows)))[1]] == [r.inp.id for r in val]


def test_build_writes_gate_compatible_files_and_manifest(patched, tmp_path):
    dctx, inputs = patched
    targets = targets_for(inputs)
    args = build_args(tmp_path, inputs, targets)
    assert dataset.cmd_build(args) == 0
    out = args.out_dir
    train, val = gates.load_jsonl(out / "train.jsonl"), gates.load_jsonl(out / "val.jsonl")
    assert len(train) + len(val) == 30 and len(val) == 3
    gates.check_messages(train, "train")
    gates.check_messages(val, "val")
    by_text = {t.text for t in targets}
    for row in train + val:
        assert set(row) == {"messages"} and row["messages"][-1]["role"] == "assistant"
        assert row["messages"][-1]["content"] in by_text
        assert row["messages"][0]["role"] == "system" and row["messages"][-2]["role"] == "user"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    # what training/train_unsloth.py does with it
    hashes = gates.verify_data_hashes(manifest, {"train.jsonl": out / "train.jsonl", "val.jsonl": out / "val.jsonl"})
    assert hashes["train.jsonl"] == gates.manifest_sha256(manifest, "train.jsonl")
    assert hashes["train.jsonl"] == gates.sha256_file(out / "train.jsonl")
    assert gates.manifest_sha256(manifest, "val.jsonl") == gates.sha256_file(out / "val.jsonl")
    assert manifest["files"]["train.jsonl"]["examples"] == len(train) and manifest["git_sha"] == "abc123"
    assert manifest["git_dirty"] is False and manifest["partial"] is False
    assert manifest["missing_targets"] == {"count": 0, "ids": []}
    assert manifest["prompt"] == {"version": "keeper-v2", "hash": dctx.prompt_hash}
    assert manifest["split"]["sha256"] == "a" * 64 and manifest["gold_hashes"] == GOLD_HASHES
    assert manifest["counts"]["all"]["kind"] == {"in_fiction": 10, "rules_question": 20}
    assert manifest["counts"]["all"]["use_rag"] == {"no_rag": 10, "rag": 20}
    assert manifest["counts"]["all"]["language"] == {"uk": 30} and manifest["counts"]["all"]["with_rolls"] == 10
    assert manifest["counts"]["all"]["kb_mode"] == {"distractor_only": 10, "none": 10, "oracle": 10}
    est = manifest["token_estimate"]
    assert 0 < est["p95"] <= est["max"] <= dataset.MAX_EST_TOKENS and est["limit"] == dataset.MAX_EST_TOKENS
    words = manifest["narration_words"]
    assert set(words) == {"all", "in_fiction", "rules_question"} and words["all"]["n"] == 30
    assert words["in_fiction"]["min"] >= dataset.WORD_RANGE["in_fiction"][0]
    assert sorted(manifest["ids"]["train"] + manifest["ids"]["val"]) == sorted(i.id for i in inputs)
    assert manifest["sources"]["inputs"]["sha256"] == dataset.file_sha256(args.inputs)
    assert manifest["sources"]["reviews"] is None and manifest["targets_without_input_hash"] == 0


def test_two_builds_have_identical_sha256(patched, tmp_path):
    _, inputs = patched
    targets = targets_for(inputs)
    first, second = tmp_path / "a", tmp_path / "b"
    assert dataset.cmd_build(build_args(tmp_path, inputs, targets, out_dir=first)) == 0
    assert dataset.cmd_build(build_args(tmp_path, list(reversed(inputs)), list(reversed(targets)),
                                        out_dir=second)) == 0
    for name in ("train.jsonl", "val.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    # the manifest differs only in the hash of the inputs / targets files, which were written in another order here
    manifests = [json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in (first, second)]
    assert [m.pop("sources") for m in manifests][0] != manifests[0]["files"]
    assert manifests[0] == manifests[1]
    assert b"\r" not in (first / "train.jsonl").read_bytes()


def test_build_lists_every_failing_id_and_writes_nothing(patched, tmp_path, capsys):
    _, inputs = patched
    targets = targets_for(inputs)
    bad = {inputs[0].id: targets[0].text + "\n", inputs[5].id: "Коротко."}
    targets = [t.model_copy(update={"text": bad[t.id]}) if t.id in bad else t for t in targets]
    args = build_args(tmp_path, inputs, targets)
    assert dataset.cmd_build(args) == 1
    shown = capsys.readouterr().out
    for example_id in bad:
        assert f"{example_id}: [" in shown
    assert not args.out_dir.exists()
    args.lint_only = True
    assert dataset.cmd_build(args) == 1 and not args.out_dir.exists()


def test_lint_only_and_missing_targets(patched, tmp_path, capsys):
    _, inputs = patched
    targets = targets_for(inputs)[:10]
    args = build_args(tmp_path, inputs, targets)
    assert dataset.cmd_build(args) == 1 and "no target" in capsys.readouterr().out and not args.out_dir.exists()
    args.lint_only = True
    assert dataset.cmd_build(args) == 0 and not args.out_dir.exists()
    assert "lint OK" in capsys.readouterr().out
    args.lint_only, args.allow_missing = False, True       # a partial build is allowed, and says so
    assert dataset.cmd_build(args) == 0
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["partial"] is True and manifest["missing_targets"]["count"] == 20
    assert manifest["missing_targets"]["ids"] == sorted(i.id for i in inputs[10:])
    capsys.readouterr()


def test_targets_without_an_input_hash_need_the_flag(patched, tmp_path, capsys):
    _, inputs = patched
    targets = [t.model_copy(update={"input_hash": None}) for t in targets_for(inputs)]
    args = build_args(tmp_path, inputs, targets)
    assert dataset.cmd_build(args) == 1 and "[no-input-hash]" in capsys.readouterr().out
    args.allow_no_input_hash = True
    assert dataset.cmd_build(args) == 0 and "warning: 30 target(s) have no input_hash" in capsys.readouterr().out
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["targets_without_input_hash"] == 30


def test_empty_val_split_is_an_error(patched, tmp_path, capsys):
    _, inputs = patched
    subset = [i for i in inputs if i.id in {"sft-roll-0", "sft-oracle-0"}]
    args = build_args(tmp_path, subset, targets_for(subset))
    assert dataset.cmd_build(args) == 1 and "must be non-empty" in capsys.readouterr().out


def test_reviews_file_changes_the_built_dataset(patched, tmp_path):
    _, inputs = patched
    targets = targets_for(inputs)
    by_id = {t.id: t for t in targets}
    edited = by_id["sft-roll-3"].text.replace("Запах цвілі", "Різкий запах цвілі")
    reviews = [review_for(by_id["sft-roll-3"], "edit", edited_text=edited),
               review_for(by_id["sft-oracle-2"], "reject")]
    args = build_args(tmp_path, inputs, targets, reviews)
    assert dataset.cmd_build(args) == 0
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reviews"] == {"edited": 1, "rejected": 1, "rejected_ids": ["sft-oracle-2"], "unreviewed": 28,
                                   "without_target_hash": 0, "rewritten": {"count": 0, "ids": []}}
    ids = manifest["ids"]["train"] + manifest["ids"]["val"]
    assert "sft-oracle-2" not in ids and len(ids) == 29
    rows = gates.load_jsonl(args.out_dir / "train.jsonl") + gates.load_jsonl(args.out_dir / "val.jsonl")
    assert any(r["messages"][-1]["content"] == edited for r in rows)
    assert manifest["sources"]["reviews"]["sha256"] == dataset.file_sha256(args.reviews)


# ------------------------------------------------------------------------------------------ review CSV


def test_allocate_is_proportional_with_one_per_stratum():
    assert allocate({"a": 10, "b": 10, "c": 10}, 9) == {"a": 3, "b": 3, "c": 3}
    assert allocate({"a": 100, "b": 1}, 10) == {"a": 9, "b": 1}
    assert allocate({"a": 2, "b": 2}, 10) == {"a": 2, "b": 2}
    assert sum(allocate({"a": 7, "b": 5, "c": 3, "d": 1}, 8).values()) == 8
    assert allocate({"a": 5, "b": 4, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}


def test_review_sample_is_stratified_seeded_and_exported(patched, tmp_path, capsys):
    dctx, inputs = patched
    rows = assemble(inputs, targets_for(inputs), [], dctx).rows
    sample = sample_for_review(rows, 9)
    strata = [f"{r.inp.kind}/{r.inp.kb_mode}" for r in sample]
    assert {s: strata.count(s) for s in set(strata)} == {
        "in_fiction/none": 3, "rules_question/oracle": 3, "rules_question/distractor_only": 3}
    assert [r.inp.id for r in sample_for_review(list(reversed(rows)), 9)] == [r.inp.id for r in sample]
    assert [r.inp.id for r in sample_for_review(rows, 9, seed=5)] != [r.inp.id for r in sample]
    assert len(sample_for_review(rows, 500)) == 30

    args = build_args(tmp_path, inputs, targets_for(inputs))
    args.sample, args.review, args.out, args.reviews_out = 9, None, tmp_path / "sample.csv", None
    assert dataset.cmd_review_export(args) == 0
    raw = (tmp_path / "sample.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw                  # BOM for Excel, LF
    assert raw.decode("utf-8-sig").split("\n", 1)[0] == ";".join(dataset.CSV_COLUMNS)

    with (tmp_path / "sample.csv").open(encoding="utf-8-sig", newline="") as f:
        table = list(csv.DictReader(f, delimiter=";"))
    assert [list(table[0])] == [dataset.CSV_COLUMNS] and len(table) == 9
    by_id = {i.id: i for i in inputs}
    for row in table:
        inp = by_id[row["id"]]
        assert row["target"] == good_text(inp) and row["player_text"] == inp.player_text
        assert row["target_hash"] == "h:" + sha16(good_text(inp))        # the prefix keeps Excel from reading a number
        assert row["verdict"] == row["edited_text"] == row["note"] == "" and row["lint"] == "ok"
        assert row["status"] == "new" and row["current_verdict"] == row["current_edit"] == ""
        assert row["kind"] == inp.kind and row["kb_mode"] == inp.kb_mode
    roll = next(r for r in table if r["id"].startswith("sft-roll"))
    assert "Spot Hidden" in roll["rolls"] and "hard" in roll["rolls"] and roll["passages"] == ""
    oracle = next(r for r in table if r["id"].startswith("sft-oracle"))
    assert "[R1]" in oracle["passages"] and "oracle:" in oracle["passages"] and "distractor:" in oracle["passages"]
    # the full passage texts as rendered into the prompt, one block per hit, labelled with the role
    inp = by_id[oracle["id"]]
    blocks = oracle["passages_text"].split("\n\n")
    assert len(blocks) == len(inp.hits) and roll["passages_text"] == ""
    for block, hit in zip(blocks, inp.hits, strict=True):
        assert block.startswith(f"{hit.role.upper()} [R{hit.marker}]") and block[len(hit.role):].strip() in \
            inp.messages[0].content
    capsys.readouterr()


def csv_text(rows: list[str], header: str = "id;kind;target;target_hash;verdict;edited_text;note") -> str:
    return "\r\n".join([header, *rows]) + "\r\n"


def test_review_csv_is_read_back_into_reviews(tmp_path):
    body = ['a-one;in_fiction;"t";h1;Accept;;"fine"',
            'a-two;meta;"t";h2;edit;"Новий текст\r\nз двох рядків  ";"fix"',
            'a-three;meta;"t";h3;REJECT;;',
            'a-four;meta;"t";h4;;;']
    path = tmp_path / "review.csv"
    path.write_bytes((BOM + csv_text(body)).encode("utf-8"))                       # Excel: BOM, CRLF, ';'
    reviews = read_review_csv(path)
    assert [(r.id, r.verdict, r.target_hash) for r in reviews] == [
        ("a-one", "accept", "h1"), ("a-two", "edit", "h2"), ("a-three", "reject", "h3")]
    assert reviews[1].edited_text == "Новий текст\nз двох рядків" and reviews[0].note == "fine"
    path.write_text("id,verdict,edited_text,note,target_hash\nx-1,accept,,,h9\n", encoding="utf-8")   # comma, no BOM
    assert [(r.id, r.target_hash) for r in read_review_csv(path)] == [("x-1", "h9")]


def test_review_csv_errors_name_the_row(tmp_path):
    path = tmp_path / "review.csv"
    rows = ["id,verdict,edited_text,note,target_hash", "a-1,maybe,,,h", "a-2,edit,,,h", "a-3,accept,leftover,,h",
            "a-4,accept,,,h", "a-4,reject,,,h", "a-5,accept,,,"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(BuildError) as e:
        read_review_csv(path)
    text = str(e.value)
    assert "review.csv:2 (a-1)" in text and "review.csv:3 (a-2): " in text and "needs edited_text" in text
    assert "review.csv:4 (a-3)" in text and "takes no edited_text" in text and "review.csv:6 (a-4): duplicate" in text
    assert "review.csv:7 (a-5): target_hash is empty" in text
    path.write_text("id,verdict\na,accept\n", encoding="utf-8")
    with pytest.raises(BuildError, match="missing column"):
        read_review_csv(path)


def test_a_csv_saved_in_a_legacy_encoding_gets_a_hint(tmp_path):
    path = tmp_path / "review.csv"
    path.write_bytes(csv_text(['a-1;m;"Привіт";h;accept;;']).encode("cp1251"))
    with pytest.raises(BuildError, match="CSV UTF-8"):
        read_review_csv(path)


def test_review_flag_merges_into_the_reviews_file_that_build_applies(patched, tmp_path, capsys):
    _, inputs = patched
    targets = targets_for(inputs)
    by_id = {t.id: t for t in targets}
    old = [review_for(by_id["sft-oracle-1"], "accept", note="first pass"), review_for(by_id["sft-roll-5"], "reject")]
    args = build_args(tmp_path, inputs, targets, old)
    csv_path = tmp_path / "review-v1.csv"
    csv_path.write_text(
        "id;verdict;edited_text;note;target_hash\n"
        f"sft-oracle-1;reject;;bad;{sha16(by_id['sft-oracle-1'].text)}\n"
        f"sft-roll-2;accept;;;{sha16(by_id['sft-roll-2'].text)}\n", encoding="utf-8")
    args.sample, args.review, args.out, args.reviews_out = None, csv_path, None, None
    assert dataset.cmd_review_export(args) == 0
    assert "1 replaced a verdict on the same text" in capsys.readouterr().out
    written = [SftReview.model_validate_json(x) for x in args.reviews.read_text(encoding="utf-8").splitlines()]
    assert [(r.id, r.verdict, r.note) for r in written] == [
        ("sft-oracle-1", "reject", "bad"), ("sft-roll-2", "accept", ""), ("sft-roll-5", "reject", "")]   # merged
    assert all(r.target_hash == sha16(by_id[r.id].text) for r in written)
    csv_path.write_text("id;verdict;edited_text;note;target_hash\nsft-ghost;reject;;;h\n", encoding="utf-8")
    assert dataset.cmd_review_export(args) == 1
    assert "review-v1.csv:2 (sft-ghost): review for an id without a target" in capsys.readouterr().out
    assert dataset.cmd_build(args) == 0
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["ids"]["train"] + manifest["ids"]["val"]
    assert "sft-oracle-1" not in ids and "sft-roll-5" not in ids and len(ids) == 28


def test_cli_parser_knows_the_new_commands(monkeypatch):
    import sys

    seen = {}
    monkeypatch.setattr(dataset, "cmd_build", lambda a: seen.setdefault("build", a) and 0)
    monkeypatch.setattr(dataset, "cmd_review_export", lambda a: seen.setdefault("export", a) and 0)
    for argv in (["build", "--version", "v1", "--lint-only", "--allow-no-input-hash"],
                 ["review-export", "--sample", "100"], ["review-export", "--review", "x.csv"]):
        monkeypatch.setattr(sys, "argv", ["sft.build", *argv])
        with pytest.raises(SystemExit) as e:
            build.main()
        assert e.value.code == 0
    assert seen["build"].lint_only and seen["build"].version == "v1" and seen["build"].allow_no_input_hash
    assert seen["export"].sample == 100 and seen["export"].review is None   # the first export call won setdefault


# ------------------------------------------------------------------------------------------ the review cycle


def fill_csv(path: Path, edits: dict[str, dict]) -> None:
    """What the author does in Excel: put verdicts into the exported sample."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        table = list(csv.DictReader(f, delimiter=";"))
    for row in table:
        row.update(edits.get(row["id"], {}))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dataset.CSV_COLUMNS, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


def read_table(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f, delimiter=";")}


def write_targets(path: Path, targets) -> None:
    path.write_text("".join(json.dumps(t.model_dump(mode="json"), ensure_ascii=False) + "\n" for t in targets),
                    encoding="utf-8")


def export(args, sample: int, **flags) -> dict[str, dict]:
    args.sample, args.review, args.out, args.reviews_out = sample, None, args.targets.parent / "sample.csv", None
    args.sample_stale, args.include_reviewed = flags.get("sample_stale", False), flags.get("include_reviewed", False)
    assert dataset.cmd_review_export(args) == 0
    return read_table(args.out)


def import_csv(args, path: Path) -> int:
    args.sample, args.review, args.out, args.reviews_out = None, path, None, None
    return dataset.cmd_review_export(args)


def built_ids(args) -> dict:
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"ids": manifest["ids"]["train"] + manifest["ids"]["val"], "reviews": manifest["reviews"]}


def test_reject_rewrite_re_review_cycle_end_to_end(patched, tmp_path, capsys):
    _, inputs = patched
    targets = targets_for(inputs)
    x, y = targets[0].id, targets[1].id                                  # two in-character turns with a roll
    args = build_args(tmp_path, inputs, targets)
    args.sample_stale = args.include_reviewed = False

    # 1. export, the author rejects x and accepts y, import: both verdicts are in the history, x is dropped
    table = export(args, 40)
    assert len(table) == 30 and {r["status"] for r in table.values()} == {"new"}
    old_csv = tmp_path / "old.csv"
    old_csv.write_bytes((tmp_path / "sample.csv").read_bytes())
    fill_csv(old_csv, {x: {"verdict": "reject", "note": "flat"}, y: {"verdict": "accept"}})
    assert import_csv(args, old_csv) == 0
    reviews = [SftReview.model_validate_json(ln) for ln in args.reviews.read_text(encoding="utf-8").splitlines()]
    assert [(r.id, r.verdict, r.target_hash) for r in reviews] == [
        (x, "reject", sha16(targets[0].text)), (y, "accept", sha16(targets[1].text))]
    assert dataset.cmd_build(args) == 0
    built = built_ids(args)
    assert x not in built["ids"] and y in built["ids"] and built["reviews"]["rejected_ids"] == [x]

    # 2. the author of the targets rewrites x. The build does not break: x is built as unreviewed, with a warning
    rewritten_x = targets[0].model_copy(update={"text": targets[0].text + " Тіні ворушаться."})
    write_targets(args.targets, [rewritten_x, *targets[1:]])
    capsys.readouterr()
    assert dataset.cmd_build(args) == 0
    shown = capsys.readouterr().out
    assert "rewritten after they were reviewed" in shown and x in shown
    built = built_ids(args)
    assert x in built["ids"] and built["reviews"]["rewritten"] == {"count": 1, "ids": [x]}
    assert built["reviews"]["rejected"] == 0 and built["reviews"]["unreviewed"] == 29       # y is reviewed, x is not

    # 3. importing the OLD csv again is refused, names the CSV line, and leaves the history alone
    history_before = args.reviews.read_bytes()
    assert import_csv(args, old_csv) == 1
    shown = capsys.readouterr().out
    assert "old.csv:" in shown and f"({x})" in shown and "not the current target" in shown
    assert args.reviews.read_bytes() == history_before

    # 4. the export offers x first, flagged; y (already reviewed for its current text) is left out by default
    table = export(args, 5)
    assert x in table and y not in table and len(table) == 5
    assert table[x]["status"] == "rewritten after review" and table[x]["target"] == rewritten_x.text
    assert table[x]["target_hash"] == "h:" + sha16(rewritten_x.text) != "h:" + sha16(targets[0].text)
    assert {r["status"] for i, r in table.items() if i != x} == {"new"}
    assert list(export(args, 5, sample_stale=True)) == [x]                                  # only the rewritten ones
    everything = export(args, 40, include_reviewed=True)
    assert everything[y]["status"] == "reviewed" and everything[y]["current_verdict"] == "accept"
    assert everything[x]["status"] == "rewritten after review" and everything[x]["current_verdict"] == ""

    # 5. the author rejects the new text, imports it: the history now has both texts of x, the new one applies
    new_csv = tmp_path / "new.csv"
    export(args, 5)
    new_csv.write_bytes((tmp_path / "sample.csv").read_bytes())
    fill_csv(new_csv, {x: {"verdict": "reject", "note": "still flat"}})
    assert import_csv(args, new_csv) == 0
    history = [(r.id, r.verdict, r.target_hash) for r in
               (SftReview.model_validate_json(ln) for ln in args.reviews.read_text(encoding="utf-8").splitlines())]
    assert sorted(history) == sorted([(x, "reject", sha16(targets[0].text)), (x, "reject", sha16(rewritten_x.text)),
                                      (y, "accept", sha16(targets[1].text))])
    assert dataset.cmd_build(args) == 0
    built = built_ids(args)
    assert x not in built["ids"] and built["reviews"]["rejected_ids"] == [x]
    assert built["reviews"]["rewritten"]["count"] == 0
    # ... a verdict on the same text can still be replaced (the later one wins), the older text's verdict is untouched
    again = tmp_path / "again.csv"
    export(args, 40, include_reviewed=True)
    again.write_bytes((tmp_path / "sample.csv").read_bytes())
    row = read_table(again)[x]
    assert row["status"] == "reviewed" and row["current_verdict"] == "reject" and row["lint"] == "(rejected)"
    fill_csv(again, {x: {"verdict": "accept", "note": "on reflection"}})
    assert import_csv(args, again) == 0
    assert dataset.cmd_build(args) == 0
    assert x in built_ids(args)["ids"] and built_ids(args)["reviews"]["rejected"] == 0
    # nothing left to review except what was never looked at; the reviewed ones are not offered again
    default = export(args, 40)
    assert x not in default and y not in default and len(default) == 28
    capsys.readouterr()


@pytest.mark.parametrize("text, success", [
    ("вдалося", True), ("удалося", True), ("вдалось", True), ("вдався", True), ("вдається", True),
    ("вдалий кидок", True), ("щось блищить вдалині", False), ("вдавлений слід", False),
])
def test_success_word_matches_only_forms_of_succeeding(text, success):
    from sft.dataset import SUCCESS_WORD

    assert bool(SUCCESS_WORD.search(text)) is success


def test_failed_success_phrase_is_not_read_as_success():
    from sft.dataset import FAILURE_WORD, SUCCESS_WORD

    line = "успіх не вдався"
    assert FAILURE_WORD.search(line)
    assert not SUCCESS_WORD.search(FAILURE_WORD.sub(" ", line))
