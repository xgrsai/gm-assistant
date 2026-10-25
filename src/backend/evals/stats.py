"""Small, dependency-free statistics for paired evaluation designs (deterministic via fixed seeds)."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def percentile(sorted_xs: Sequence[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def bootstrap_ci(xs: Sequence[float], *, iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (items resampled with replacement)."""
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return percentile(means, alpha / 2), percentile(means, 1 - alpha / 2)


def paired_diff(a: Sequence[float], b: Sequence[float], *, iters: int = 2000, seed: int = 0) -> dict:
    """Mean of b - a over paired items, bootstrap CI and a sign-flip permutation p-value."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diffs, iters=iters, seed=seed)
    p = sign_flip_p(diffs, iters=iters, seed=seed)
    return {"n": len(diffs), "mean_diff": mean(diffs), "ci95": [lo, hi], "p": p}


def sign_flip_p(diffs: Sequence[float], *, iters: int = 5000, seed: int = 0) -> float:
    """Two-sided paired permutation test: randomly flip the sign of each difference."""
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    n = len(nonzero)
    if n <= 16:  # exact enumeration
        hits = sum(1 for mask in range(1 << n)
                   if abs(sum(d if mask >> i & 1 else -d for i, d in enumerate(nonzero))) >= observed - 1e-12)
        return hits / (1 << n)
    rng = random.Random(seed)
    hits = sum(1 for _ in range(iters)
               if abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - 1e-12)
    return (hits + 1) / (iters + 1)


def unpaired_diff(a: Sequence[float], b: Sequence[float], *, iters: int = 2000, seed: int = 0,
                  exact_limit: int = 20000) -> dict:
    """Mean of b - mean of a for two INDEPENDENT samples (different items, unlike paired_diff).

    The CI is a percentile bootstrap that resamples each sample on its own. The p-value is a two-sided permutation test
    over the sample labels: exact when there are at most `exact_limit` ways to split the pooled values, otherwise
    `iters` random splits.
    """
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    na, nb = len(a), len(b)
    observed = mean(b) - mean(a)
    rng = random.Random(seed)
    boots = sorted(sum(b[rng.randrange(nb)] for _ in range(nb)) / nb - sum(a[rng.randrange(na)] for _ in range(na)) / na
                   for _ in range(iters))
    pooled, total = [*a, *b], sum(a) + sum(b)

    def gap(sum_b: float) -> float:
        return sum_b / nb - (total - sum_b) / na

    if math.comb(na + nb, nb) <= exact_limit:
        splits = list(itertools.combinations(range(na + nb), nb))
        hits = sum(1 for idx in splits if abs(gap(sum(pooled[i] for i in idx))) >= abs(observed) - 1e-12)
        p = hits / len(splits)
    else:
        rng = random.Random(seed)
        hits = sum(1 for _ in range(iters)
                   if abs(gap(sum(pooled[i] for i in rng.sample(range(na + nb), nb)))) >= abs(observed) - 1e-12)
        p = (hits + 1) / (iters + 1)
    return {"n_a": na, "n_b": nb, "mean_diff": observed,
            "ci95": [percentile(boots, 0.025), percentile(boots, 0.975)], "p": p}


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (two-sided binomial on discordant pairs)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "p": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label sequences")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
