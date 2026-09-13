from hypothesis import given
from hypothesis import strategies as st

from keeper.game.coc7e.dice import (
    Level,
    RulesConfig,
    make_rng,
    max_expr,
    net_dice,
    parse_expr,
    roll_d100,
    roll_expr,
    success_level,
)


def test_success_levels_regular():
    assert success_level(1, 10) is Level.CRITICAL
    assert success_level(12, 60) is Level.EXTREME
    assert success_level(13, 60) is Level.HARD
    assert success_level(30, 60) is Level.HARD
    assert success_level(31, 60) is Level.REGULAR
    assert success_level(60, 60) is Level.REGULAR
    assert success_level(61, 60) is Level.FAILURE


def test_fumbles():
    assert success_level(100, 99) is Level.FUMBLE
    assert success_level(96, 45) is Level.FUMBLE
    assert success_level(96, 60) is Level.FAILURE
    assert success_level(95, 45) is Level.FAILURE


def test_fumble_threshold_flag():
    target, base = RulesConfig(fumble_threshold_on="target"), RulesConfig(fumble_threshold_on="base_skill")
    assert success_level(97, 60, "hard", target) is Level.FUMBLE  # required roll 30 < 50
    assert success_level(97, 60, "hard", base) is Level.FAILURE   # base skill 60 >= 50


def test_cancel_then_cap():
    assert net_dice(3, 1) == 2
    assert net_dice(1, 1) == 0
    assert net_dice(0, 5) == -2
    assert len(roll_d100(3, 1, make_rng(1, "x")).tens) == 3


def test_double_zero_is_100():
    class Zero:
        def randint(self, a, b):
            return 0

    assert roll_d100(rng=Zero()).result == 100


@given(st.integers(0, 10_000))
def test_bonus_never_worse_penalty_never_better(seed):
    plain = roll_d100(0, 0, make_rng(seed, "x"))
    bonus = roll_d100(1, 0, make_rng(seed, "x"))
    penalty = roll_d100(0, 1, make_rng(seed, "x"))
    assert 1 <= bonus.result <= plain.result <= penalty.result <= 100


@given(st.integers(0, 10_000))
def test_seeded_rng_is_reproducible(seed):
    assert roll_d100(2, 0, make_rng(seed, 3, 0)) == roll_d100(2, 0, make_rng(seed, 3, 0))


def test_dice_expressions():
    assert parse_expr("1D6") == [(1, 1, 6)]
    assert parse_expr("2d6+1") == [(1, 2, 6), (1, 1, None)]
    assert max_expr("1D4+1D4") == 8
    assert max_expr("1D10") == 10
    assert max_expr("0") == 0
    rng = make_rng(1, "e")
    for _ in range(200):
        assert 3 <= roll_expr("1D6+2", rng) <= 8


def test_bad_expressions():
    import pytest

    for bad in ["", "D", "1D1", "abc", "1D6 2"]:
        with pytest.raises(ValueError):
            parse_expr(bad)
