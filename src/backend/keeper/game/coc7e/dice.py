"""Call of Cthulhu 7e dice: pure functions, no LLM imports.

Rules sources and confidence:
- Quick-Start Rules (verified): Regular <= skill, Hard <= skill // 2, Extreme <= skill // 5;
  bonus/penalty dice are extra tens dice keeping the lowest/highest; one bonus cancels one penalty.
- Keeper Rulebook via secondary sources (medium confidence, hence the RulesConfig flags):
  01 is a critical; 100 is always a fumble, 96-100 is a fumble when the roll required for success is
  below 50; at most 2 bonus or penalty dice; a fumbled Sanity roll loses the maximum amount.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

Difficulty = Literal["regular", "hard", "extreme"]


class Level(IntEnum):
    FUMBLE = 0
    FAILURE = 1
    REGULAR = 2
    HARD = 3
    EXTREME = 4
    CRITICAL = 5


REQUIRED: dict[str, Level] = {"regular": Level.REGULAR, "hard": Level.HARD, "extreme": Level.EXTREME}

TARGET = {
    "regular": lambda skill: skill,
    "hard": lambda skill: skill // 2,
    "extreme": lambda skill: skill // 5,
}


@dataclass(frozen=True)
class RulesConfig:
    max_bonus_penalty: int = 2
    # KRB: "if the dice roll required for success is below 50" -> literal reading uses the
    # difficulty-adjusted target; some summaries use the base skill instead.
    fumble_threshold_on: Literal["target", "base_skill"] = "target"
    fumbled_san_is_max_loss: bool = True


DEFAULT_RULES = RulesConfig()


def make_rng(seed: int | None, *salt: object) -> random.Random:
    """Seeded RNG for reproducible experiments; OS entropy when seed is None.

    String seeds are hashed with SHA-512 by random.Random, so the stream is identical across processes
    regardless of PYTHONHASHSEED.
    """
    if seed is None:
        return random.SystemRandom()
    return random.Random(":".join(map(str, (seed, *salt))))


@dataclass(frozen=True)
class D100:
    result: int
    tens: tuple[int, ...]
    units: int


def net_dice(bonus: int, penalty: int, cfg: RulesConfig = DEFAULT_RULES) -> int:
    """Cancel bonus against penalty first, then cap: bonus 3 + penalty 1 -> net +2."""
    net = max(0, bonus) - max(0, penalty)
    return max(-cfg.max_bonus_penalty, min(net, cfg.max_bonus_penalty))


def roll_d100(bonus: int = 0, penalty: int = 0, rng: random.Random | None = None,
              cfg: RulesConfig = DEFAULT_RULES) -> D100:
    rng = rng or random.SystemRandom()
    net = net_dice(bonus, penalty, cfg)
    units = rng.randint(0, 9)
    tens = tuple(rng.randint(0, 9) for _ in range(1 + abs(net)))
    totals = [100 if (t == 0 and units == 0) else t * 10 + units for t in tens]  # "00" + "0" reads as 100
    result = min(totals) if net > 0 else max(totals) if net < 0 else totals[0]
    return D100(result, tens, units)


def success_level(roll: int, skill: int, difficulty: Difficulty = "regular",
                  cfg: RulesConfig = DEFAULT_RULES) -> Level:
    fumble_base = TARGET[difficulty](skill) if cfg.fumble_threshold_on == "target" else skill
    if roll == 1:
        return Level.CRITICAL
    if roll == 100 or (fumble_base < 50 and roll >= 96):
        return Level.FUMBLE
    if roll <= skill // 5:
        return Level.EXTREME
    if roll <= skill // 2:
        return Level.HARD
    if roll <= skill:
        return Level.REGULAR
    return Level.FAILURE


_TERM = re.compile(r"\s*([+-])?\s*(?:(\d*)\s*[dD]\s*(\d+)|(\d+))\s*")


def parse_expr(expr: str) -> list[tuple[int, int, int | None]]:
    """'1D6', '2d6+1', '1D4+1D4', '0' -> [(sign, count, sides | None)].

    An own parser keeps the RNG injectable (and therefore seedable).
    """
    s, pos, terms = expr.strip(), 0, []
    if not s:
        raise ValueError("empty dice expression")
    while pos < len(s):
        m = _TERM.match(s, pos)
        if not m or m.end() == pos or (terms and not m.group(1)):
            raise ValueError(f"bad dice expression: {expr!r}")
        sign = -1 if m.group(1) == "-" else 1
        if m.group(3):
            n, sides = int(m.group(2) or 1), int(m.group(3))
            if not (1 <= n <= 100 and 2 <= sides <= 1000):
                raise ValueError(f"dice out of range: {expr!r}")
            terms.append((sign, n, sides))
        else:
            terms.append((sign, int(m.group(4)), None))
        pos = m.end()
    return terms


def roll_expr(expr: str, rng: random.Random) -> int:
    return sum(sign * (n if sides is None else sum(rng.randint(1, sides) for _ in range(n)))
               for sign, n, sides in parse_expr(expr))


def max_expr(expr: str) -> int:
    return sum(sign * (n if sides is None else (n * sides if sign > 0 else n))
               for sign, n, sides in parse_expr(expr))
