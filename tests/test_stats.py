import random

import pytest
from scipy import stats as sps

from trajeval.stats import benjamini_hochberg, bootstrap_ci, cohen_kappa, confusion, fisher_one_sided, wilson


@pytest.mark.parametrize("a,an,b,bn", [(5, 5, 0, 5), (5, 5, 1, 5), (4, 5, 0, 5), (3, 5, 3, 5), (0, 5, 5, 5),
                                       (10, 10, 2, 10), (7, 12, 1, 9), (0, 3, 0, 3)])
def test_fisher_matches_scipy(a, an, b, bn):
    ref = sps.fisher_exact([[a, an - a], [b, bn - b]], alternative="greater").pvalue
    assert fisher_one_sided(a, an, b, bn) == pytest.approx(ref, rel=1e-9)


def test_fisher_known_values():
    assert fisher_one_sided(5, 5, 0, 5) == pytest.approx(1 / 252)  # C(10,5) = 252
    assert fisher_one_sided(3, 5, 3, 5) > 0.5


def test_bh_matches_scipy():
    rng = random.Random(0)
    p = [rng.random() ** 3 for _ in range(40)]
    ref = sps.false_discovery_control(p, method="bh")
    assert benjamini_hochberg(p) == pytest.approx(list(ref), rel=1e-9)


def test_kappa_textbook():
    # Two raters, 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po=0.7, pe=0.5, kappa=0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(a, a) == 1.0
    assert confusion(a, b, ["y", "n"]) == [[20, 5], [10, 15]]


def test_kappa_chance_level_and_degenerate():
    rng = random.Random(1)
    a = [rng.choice("pf") for _ in range(4000)]
    b = [rng.choice("pf") for _ in range(4000)]
    assert abs(cohen_kappa(a, b)) < 0.05
    assert cohen_kappa(["p"] * 5, ["p"] * 5) == 1.0


def test_wilson_and_bootstrap():
    lo, hi = wilson(0, 5)
    assert lo == 0 and 0.4 < hi < 0.5
    ci = bootstrap_ci([0.0] * 50 + [1.0] * 50, lambda xs: sum(xs) / len(xs), n_boot=1000)
    assert ci[0] < 0.5 < ci[1] and ci[1] - ci[0] < 0.3
