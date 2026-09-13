"""Call of Cthulhu 7e resolution: turns an adjudication into dice records and sheet updates.

Code decides outcomes; the LLM only decides what to roll. Skill values always come from the
server-owned investigator sheet, never from model output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .dice import (
    DEFAULT_RULES,
    REQUIRED,
    Level,
    RulesConfig,
    make_rng,
    max_expr,
    parse_expr,
    roll_d100,
    roll_expr,
    success_level,
)

CHARACTERISTICS = ("STR", "CON", "SIZ", "DEX", "APP", "INT", "POW", "EDU")
PUSHABLE = {"skill", "characteristic"}  # not Luck, Sanity, combat, damage or opposed rolls
DIFFICULTIES = ("regular", "hard", "extreme")


class RuleViolation(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class Resolution:
    investigator: dict
    records: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_investigator(pregen: dict) -> dict:
    """Derive secondary attributes (QSR): HP = (CON+SIZ)//10, MP = POW//5, SAN = POW, max SAN = 99 - Mythos."""
    ch = {k: int(pregen["characteristics"][k]) for k in CHARACTERISTICS}
    skills = {k: int(v) for k, v in pregen["skills"].items()}
    hp = (ch["CON"] + ch["SIZ"]) // 10
    mp = ch["POW"] // 5
    return {
        "name": pregen["name"],
        "occupation": pregen["occupation"],
        "age": int(pregen.get("age", 30)),
        "characteristics": ch,
        "skills": skills,
        "hp": hp, "hp_max": hp,
        "mp": mp, "mp_max": mp,
        "san": ch["POW"], "san_max": 99 - skills.get("Cthulhu Mythos", 0),
        "luck": int(pregen["luck"]),
    }


def sheet_names(inv: dict) -> dict:
    """What the adjudicator may choose from: names only, values stay server-side."""
    return {"skills": sorted(inv["skills"]), "characteristics": list(CHARACTERISTICS)}


PUBLIC_KEYS = ("name", "occupation", "age", "characteristics", "skills",
               "hp", "hp_max", "mp", "mp_max", "san", "san_max", "luck")


def public_sheet(inv: dict) -> dict:
    return {k: inv[k] for k in PUBLIC_KEYS if k in inv}


def lookup(inv: dict, kind: str, target: str | None) -> tuple[str, int]:
    if kind == "luck":
        return "Luck", int(inv["luck"])
    if not target:
        raise RuleViolation("missing_target")
    if target.strip().upper() in CHARACTERISTICS:
        name = target.strip().upper()
        return name, int(inv["characteristics"][name])
    by_fold = {k.casefold(): (k, v) for k, v in inv["skills"].items()}
    key = target.strip().casefold()
    if key not in by_fold:
        raise RuleViolation(f"unknown_skill:{target}")
    name, value = by_fold[key]
    return name, int(value)


def percentile_check(inv: dict, kind: str, target: str | None, difficulty: str, bonus: int, penalty: int,
                     rng: random.Random, cfg: RulesConfig = DEFAULT_RULES, *, pushed: bool = False,
                     turn_no: int = 0) -> dict:
    if difficulty not in DIFFICULTIES:
        difficulty = "regular"
    name, value = lookup(inv, kind, target)
    if kind != "luck" and name in CHARACTERISTICS:
        kind = "characteristic"
    d = roll_d100(bonus, penalty, rng, cfg)
    level = success_level(d.result, value, difficulty, cfg)
    return {
        "kind": kind, "target": name, "value": value, "difficulty": difficulty,
        "bonus": bonus, "penalty": penalty, "tens": list(d.tens), "units": d.units, "roll": d.result,
        "level": level.name.lower(), "passed": level >= REQUIRED[difficulty], "pushed": pushed,
        "turn_no": turn_no,
    }


def can_push(prev: dict | None, turn_no: int) -> bool:
    """A push re-rolls the failed roll of the immediately previous turn, once, never after a fumble."""
    return (
        bool(prev)
        and prev["kind"] in PUSHABLE
        and not prev["passed"]
        and prev["level"] != "fumble"
        and not prev["pushed"]
        and prev.get("turn_no") == turn_no - 1
    )


def split_sanity(notation: str) -> tuple[str, str]:
    parts = [p.strip() for p in (notation or "").split("/")]
    if len(parts) != 2:
        raise RuleViolation(f"bad_sanity_notation:{notation}")
    try:
        parse_expr(parts[0])
        parse_expr(parts[1])
    except ValueError as e:
        raise RuleViolation(f"bad_sanity_notation:{notation}") from e
    return parts[0], parts[1]


def sanity_check(inv: dict, notation: str, rng: random.Random, cfg: RulesConfig = DEFAULT_RULES,
                 *, turn_no: int = 0) -> tuple[list[dict], dict]:
    on_success, on_failure = split_sanity(notation)
    san = int(inv["san"])
    d = roll_d100(rng=rng, cfg=cfg)
    level = success_level(d.result, san, "regular", cfg)
    success = level >= Level.REGULAR
    if level == Level.FUMBLE and cfg.fumbled_san_is_max_loss:
        lost = max_expr(on_failure)
    else:
        lost = roll_expr(on_success if success else on_failure, rng)
    lost = max(0, min(lost, san))
    after = san - lost
    day_start = int(inv.get("san_day_start", san))
    lost_today = int(inv.get("san_lost_today", 0)) + lost
    rec = {
        "kind": "sanity", "target": "SAN", "value": san, "difficulty": "regular", "bonus": 0, "penalty": 0,
        "tens": list(d.tens), "units": d.units, "roll": d.result, "level": level.name.lower(),
        "passed": success, "pushed": False, "turn_no": turn_no, "notation": f"{on_success}/{on_failure}",
        "san_loss": lost, "san_after": after,
        "involuntary_action": not success,                              # QSR: Keeper may take an involuntary action
        "indefinite_insanity": lost_today >= max(1, day_start // 5),    # KRB via secondary sources [medium]
        "permanent_insanity": after <= 0,
    }
    records = [rec]
    if lost >= 5 and after > 0:
        # QSR: losing 5+ SAN at once calls for an INT roll; SUCCESS means the mind grasps the horror
        # -> temporary insanity (1D10 hours).
        int_rec = percentile_check(inv, "characteristic", "INT", "regular", 0, 0, rng, cfg, turn_no=turn_no)
        int_rec["purpose"] = "temporary_insanity_check"
        rec["temporary_insanity"] = int_rec["passed"]
        records.append(int_rec)
    return records, {**inv, "san": after, "san_day_start": day_start, "san_lost_today": lost_today}


def last_player_roll(roll_log: list[dict]) -> dict | None:
    for rec in reversed(roll_log):
        if rec.get("purpose") is None:
            return rec
    return None


def resolve(inv: dict, adj: dict, roll_log: list[dict], *, seed: int | None, turn_no: int,
            sanity_events: dict[str, str] | None = None, cfg: RulesConfig = DEFAULT_RULES) -> Resolution:
    out = Resolution(investigator=inv)
    kind = adj.get("roll_type") or "none"
    rng = make_rng(seed, turn_no, 0)  # same seed + turn -> same dice in every arm (paired design)

    def clamp(x: object) -> int:
        try:
            return max(0, min(int(x or 0), cfg.max_bonus_penalty))
        except (TypeError, ValueError):
            return 0

    try:
        if kind == "sanity":
            notation = adj.get("sanity_loss") or ""
            event = adj.get("sanity_event")
            if sanity_events and event in sanity_events:  # scenario-defined magnitude wins over the LLM's guess
                notation = sanity_events[event]
            records, out.investigator = sanity_check(inv, notation, rng, cfg, turn_no=turn_no)
            out.records.extend(records)
        elif kind in ("skill", "characteristic", "luck"):
            if adj.get("is_push"):
                prev = last_player_roll(roll_log)
                if not can_push(prev, turn_no):
                    raise RuleViolation("push_not_allowed")
                rec = percentile_check(inv, prev["kind"], prev["target"], prev["difficulty"], prev["bonus"],
                                       prev["penalty"], rng, cfg, pushed=True, turn_no=turn_no)
            else:
                rec = percentile_check(inv, kind, adj.get("target"), adj.get("difficulty") or "regular",
                                       clamp(adj.get("bonus_dice")), clamp(adj.get("penalty_dice")), rng, cfg,
                                       turn_no=turn_no)
            out.records.append(rec)
    except RuleViolation as e:
        out.errors.append(e.code)  # the narrator sees the error and asks the player to clarify
    return out
