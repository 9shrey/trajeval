"""Statistics, written out by hand (cross-checked against scipy in tests)."""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def fisher_one_sided(a_pass: int, a_n: int, b_pass: int, b_n: int) -> float:
    """P-value of Fisher's exact test for H1: pass rate of A > pass rate of B (B regressed).

    With both margins fixed, A's pass count X is hypergeometric: drawing a_n trials from
    a_n + b_n of which a_pass + b_pass passed. p = P(X >= a_pass).
    """
    n, k = a_n + b_n, a_pass + b_pass
    denom = math.comb(n, a_n)
    hi = min(k, a_n)
    return sum(math.comb(k, x) * math.comb(n - k, a_n - x) for x in range(a_pass, hi + 1)) / denom


def benjamini_hochberg(pvals: Sequence[float]) -> list[float]:
    """BH-adjusted p-values (q-values); reject where q <= alpha to control FDR at alpha."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        q[i] = running
    return q


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = passes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(items: Sequence, stat: Callable[[Sequence], float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap. `items` are the resampling units (tasks, for clustered data)."""
    rng = random.Random(seed)
    k = len(items)
    if k == 0:
        return (float("nan"), float("nan"))
    vals = sorted(stat([items[rng.randrange(k)] for _ in range(k)]) for _ in range(n_boot))
    lo = vals[int(alpha / 2 * n_boot)]
    hi = vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa between two raters over the same items (any hashable labels)."""
    assert len(a) == len(b) and a, "need two equal-length, non-empty label lists"
    n = len(a)
    labels = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == lab for x in a) / n) * (sum(y == lab for y in b) / n) for lab in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def confusion(truth: Sequence, pred: Sequence, labels: Sequence) -> list[list[int]]:
    """rows = truth label, cols = predicted label."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, pred):
        m[idx[t]][idx[p]] += 1
    return m


def mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")
