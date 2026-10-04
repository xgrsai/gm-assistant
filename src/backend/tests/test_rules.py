from hypothesis import given
from hypothesis import strategies as st

from keeper.game.coc7e.dice import make_rng
from keeper.game.coc7e.rules import (
    RuleViolation,
    build_investigator,
    can_push,
    lookup,
    resolve,
    sanity_check,
)

PREGEN = {
    "name": "Test", "occupation": "Tester", "age": 30,
    "characteristics": {"STR": 40, "CON": 60, "SIZ": 50, "DEX": 50, "APP": 50, "INT": 80, "POW": 60, "EDU": 70},
    "skills": {"Spot Hidden": 70, "Library Use": 60, "Cthulhu Mythos": 0},
    "luck": 55,
}


def inv():
    return build_investigator(PREGEN)


def test_derived_attributes():
    i = inv()
    assert (i["hp"], i["mp"], i["san"], i["san_max"]) == (11, 12, 60, 99)


def test_lookup_uses_sheet_values_case_insensitive():
    assert lookup(inv(), "skill", "spot hidden") == ("Spot Hidden", 70)
    assert lookup(inv(), "characteristic", "pow") == ("POW", 60)
    assert lookup(inv(), "luck", None) == ("Luck", 55)


def test_unknown_skill_becomes_engine_error():
    res = resolve(inv(), {"roll_type": "skill", "target": "Astrophysics"}, [], seed=1, turn_no=1)
    assert res.records == [] and res.errors == ["unknown_skill:Astrophysics"]


def test_skill_check_record():
    res = resolve(inv(), {"roll_type": "skill", "target": "Spot Hidden", "difficulty": "hard", "bonus_dice": 5},
                  [], seed=7, turn_no=2)
    (rec,) = res.records
    assert rec["target"] == "Spot Hidden" and rec["value"] == 70 and rec["difficulty"] == "hard"
    assert rec["bonus"] == 2 and len(rec["tens"]) == 3 and rec["turn_no"] == 2
    assert rec["passed"] == (rec["roll"] <= 35 or rec["roll"] == 1)


def test_same_seed_same_dice_across_arms():
    adj = {"roll_type": "skill", "target": "Library Use"}
    assert resolve(inv(), adj, [], seed=11, turn_no=4).records == resolve(inv(), adj, [], seed=11, turn_no=4).records


def failed(turn_no, **kw):
    return {"kind": "skill", "target": "Library Use", "difficulty": "regular", "bonus": 0, "penalty": 0,
            "passed": False, "level": "failure", "pushed": False, "turn_no": turn_no, **kw}


def test_push_rules():
    assert can_push(failed(3), 4)
    assert not can_push(failed(2), 4)                     # only the immediately previous turn
    assert not can_push(failed(3, level="fumble"), 4)
    assert not can_push(failed(3, pushed=True), 4)
    assert not can_push(failed(3, kind="sanity"), 4)
    assert not can_push(failed(3, passed=True), 4)
    res = resolve(inv(), {"roll_type": "skill", "is_push": True}, [failed(3)], seed=1, turn_no=4)
    assert res.records[0]["pushed"] and res.records[0]["target"] == "Library Use"
    res = resolve(inv(), {"roll_type": "skill", "is_push": True}, [failed(1)], seed=1, turn_no=4)
    assert res.errors == ["push_not_allowed"]


def test_sanity_event_from_scenario_overrides_llm_notation():
    res = resolve(inv(), {"roll_type": "sanity", "sanity_event": "big", "sanity_loss": "0/1"}, [], seed=3, turn_no=1,
                  sanity_events={"big": "5/5"})
    assert res.records[0]["notation"] == "5/5" and res.records[0]["san_loss"] == 5
    assert res.investigator["san"] == 55


def test_bad_sanity_notation():
    res = resolve(inv(), {"roll_type": "sanity", "sanity_loss": "lots"}, [], seed=3, turn_no=1)
    assert res.errors == ["bad_sanity_notation:lots"]


@given(st.integers(0, 5_000))
def test_int_roll_when_loss_at_least_5(seed):
    records, after = sanity_check(inv(), "5/5", make_rng(seed, "s"))
    assert after["san"] == 55
    assert records[1]["purpose"] == "temporary_insanity_check"
    assert records[0]["temporary_insanity"] == records[1]["passed"]


@given(st.integers(0, 5_000))
def test_sanity_loss_bounded(seed):
    records, after = sanity_check({**inv(), "san": 3}, "1D10/1D100", make_rng(seed, "t"))
    assert 0 <= after["san"] <= 3 and records[0]["san_loss"] <= 3


LOSE_5 = {"roll_type": "sanity", "sanity_loss": "5/5"}
NEW_DAY = {"roll_type": "none", "new_day": True}


def play(sheet, *adjs, seed=1):
    """Resolves adjudications on consecutive turns; returns the final sheet and each SAN roll's insanity flag."""
    flags = []
    for turn_no, adj in enumerate(adjs, start=1):
        res = resolve(sheet, adj, [], seed=seed, turn_no=turn_no)
        sheet = res.investigator
        flags += [r["indefinite_insanity"] for r in res.records if r["kind"] == "sanity"]
    return sheet, flags


def without_flag(records):
    return [{k: v for k, v in r.items() if k != "indefinite_insanity"} for r in records]


def test_indefinite_insanity_is_counted_per_game_day():
    # POW 60: 12+ SAN lost in one day is indefinite insanity; 3 x 5 spread over three days is not.
    sheet, flags = play(inv(), LOSE_5, NEW_DAY, LOSE_5, {**LOSE_5, "new_day": True})
    assert flags == [False, False, False]
    assert (sheet["san"], sheet["san_day_start"], sheet["san_lost_today"]) == (45, 50, 5)


def test_indefinite_insanity_flags_only_the_crossing_roll():
    _, flags = play(inv(), LOSE_5, LOSE_5, LOSE_5, LOSE_5)  # 5, 10, 15 (crosses 12), 20 in one day
    assert flags == [False, False, True, False]


def test_new_day_without_roll_resets_the_day_budget():
    tired = {**inv(), "san": 50, "san_day_start": 60, "san_lost_today": 10}
    res = resolve(tired, NEW_DAY, [], seed=1, turn_no=5)
    assert res.records == [] and res.errors == []
    assert (res.investigator["san_day_start"], res.investigator["san_lost_today"]) == (50, 0)


def test_new_day_applies_before_the_roll_and_keeps_the_dice():
    tired = {**inv(), "san": 50, "san_day_start": 60, "san_lost_today": 10}
    same_day = resolve(tired, LOSE_5, [], seed=9, turn_no=5)
    next_day = resolve(tired, {**LOSE_5, "new_day": True}, [], seed=9, turn_no=5)
    assert same_day.records[0]["indefinite_insanity"]       # 10 + 5 >= 60 // 5
    assert not next_day.records[0]["indefinite_insanity"]   # 5 < 50 // 5
    assert without_flag(same_day.records) == without_flag(next_day.records)  # the reset draws no randomness


@given(st.lists(st.integers(0, 15), max_size=10), st.integers(0, 1_000))
def test_indefinite_insanity_at_most_once_per_day(losses, seed):
    sheet, flags = play(inv(), *({"roll_type": "sanity", "sanity_loss": f"{x}/{x}"} for x in losses), seed=seed)
    assert sum(flags) <= 1
    assert any(flags) == (sheet.get("san_lost_today", 0) >= 60 // 5)


def test_rule_violation_code():
    try:
        raise RuleViolation("x")
    except RuleViolation as e:
        assert e.code == "x"
