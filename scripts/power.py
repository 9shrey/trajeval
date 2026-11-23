"""Exact power of the per-task regression test: how many trials do you need?

For a task whose true pass rate drops from p0 (baseline) to p1 (candidate), with k trials on each
side, enumerate every outcome (a, b) ~ Binomial(k, p0) x Binomial(k, p1) and sum the probability
of those where the one-sided Fisher p-value clears the threshold. Two thresholds:
  alpha = 0.05            the uncorrected test (the "suspect" flag)
  alpha = 0.10 / 48       what one lone regression needs to pass BH at FDR 0.10 across the
                          48 gated tasks (the threshold relaxes as more tasks regress at once)

    python scripts/power.py
"""

from math import comb

from trajeval.stats import fisher_one_sided

N_TASKS, FDR = 48, 0.10  # the deterministically graded tasks a CI gate compares


def binom(k, n, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def power(p0, p1, k, alpha):
    return sum(binom(a, k, p0) * binom(b, k, p1)
               for a in range(k + 1) for b in range(k + 1) if fisher_one_sided(a, k, b, k) <= alpha)


def main():
    drops = [(1.0, 0.0), (0.9, 0.1), (0.9, 0.5), (0.8, 0.4), (0.6, 0.3)]
    for name, alpha in [("uncorrected p < 0.05", 0.05), (f"lone regression under BH (p <= {FDR}/{N_TASKS})", FDR / N_TASKS)]:
        print(f"\n{name}")
        print("| trials per side | " + " | ".join(f"{a:.0%} -> {b:.0%}" for a, b in drops) + " |")
        print("|---:|" + "---:|" * len(drops))
        for k in (3, 5, 10, 20, 40):
            print(f"| {k} | " + " | ".join(f"{power(a, b, k, alpha):.0%}" for a, b in drops) + " |")


if __name__ == "__main__":
    main()
