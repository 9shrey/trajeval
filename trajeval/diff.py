"""Compare two runs and flag per-task regressions with a significance-aware rule.

Rule (per task, k trials each side):
  p_reg  = one-sided Fisher exact p for "candidate pass rate < baseline pass rate"
  q_reg  = Benjamini-Hochberg adjusted across all tasks in the suite
  regressed  if q_reg <= alpha                 (FDR-controlled: the CI gate)
  suspect    if p_reg < 0.05 but not regressed (worth a look, does not fail CI)
  improved   symmetric
  flaky      the baseline itself passed some trials and failed others

Suite level: mean per-task pass-rate difference with a bootstrap CI that resamples *tasks*
(trials of one task are correlated, so resampling trajectories would understate the error).
The naive rule ("any task whose pass rate dropped") is reported alongside for contrast.
"""

from __future__ import annotations

from .stats import benjamini_hochberg, bootstrap_ci, fisher_one_sided, mean
from .store import per_task, summarize

METRICS = ("steps", "tool_calls", "invalid_calls", "tokens", "wall_ms")


def diff_runs(base: dict, cand: dict, alpha: float = 0.10, n_boot: int = 2000, checker_only: bool = False) -> dict:
    """checker_only: compare only deterministically graded tasks. A judge with kappa well below 1 adds
    grading noise to every open-ended task, which is no basis for failing a build."""
    A, B = per_task(base, checker_only), per_task(cand, checker_only)
    tids = [t for t in A if t in B and A[t]["n"] and B[t]["n"]]
    p_reg = [fisher_one_sided(A[t]["passes"], A[t]["n"], B[t]["passes"], B[t]["n"]) for t in tids]
    p_imp = [fisher_one_sided(B[t]["passes"], B[t]["n"], A[t]["passes"], A[t]["n"]) for t in tids]
    q_reg, q_imp = benjamini_hochberg(p_reg), benjamini_hochberg(p_imp)

    rows = []
    for i, t in enumerate(tids):
        a, b = A[t], B[t]
        delta = b["pass_rate"] - a["pass_rate"]
        if q_reg[i] <= alpha:
            status = "regressed"
        elif q_imp[i] <= alpha:
            status = "improved"
        elif p_reg[i] < 0.05:
            status = "suspect"
        elif 0 < a["passes"] < a["n"]:
            status = "flaky"
        else:
            status = "unchanged"
        rows.append({
            "task_id": t, "category": a["category"], "status": status,
            "base": f"{a['passes']}/{a['n']}", "cand": f"{b['passes']}/{b['n']}",
            "base_rate": a["pass_rate"], "cand_rate": b["pass_rate"], "delta": delta,
            "p_regress": p_reg[i], "q_regress": q_reg[i], "p_improve": p_imp[i], "q_improve": q_imp[i],
            "naive_regressed": delta < 0,
        })

    deltas = [r["delta"] for r in rows]
    suite = {"mean_delta": mean(deltas), "ci": bootstrap_ci(deltas, mean, n_boot=n_boot)}
    suite["significant"] = suite["ci"][0] > 0 or suite["ci"][1] < 0
    metric_deltas = {}
    for m in METRICS:
        d = [B[t][m] - A[t][m] for t in tids if B[t][m] == B[t][m] and A[t][m] == A[t][m]]
        metric_deltas[m] = {"mean_delta": mean(d), "ci": bootstrap_ci(d, mean, n_boot=n_boot)}
    count = lambda s: sum(r["status"] == s for r in rows)  # noqa: E731
    return {
        "base": summarize(base), "cand": summarize(cand), "alpha": alpha, "checker_only": checker_only, "tasks": rows,
        "suite": suite, "metrics": metric_deltas,
        "counts": {
            "regressed": count("regressed"), "improved": count("improved"), "suspect": count("suspect"),
            "flaky": count("flaky"), "unchanged": count("unchanged"),
            "naive_regressed": sum(r["naive_regressed"] for r in rows),
            "raw_p05_regressed": sum(r["p_regress"] < 0.05 for r in rows),
        },
    }


def format_diff(d: dict, show: str = "flagged") -> str:
    b, c = d["base"], d["cand"]
    lo, hi = d["suite"]["ci"]
    scope = "deterministically graded tasks only" if d.get("checker_only") else "all graded tasks"
    out = [f"base {b['run_id']} ({b['pass_rate']:.1%})  vs  cand {c['run_id']} ({c['pass_rate']:.1%})  [{scope}, {len(d['tasks'])} tasks]",
           f"suite: mean per-task delta {d['suite']['mean_delta']:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]"
           f"  {'SIGNIFICANT' if d['suite']['significant'] else 'not significant'}",
           "counts: " + ", ".join(f"{k}={v}" for k, v in d["counts"].items()), ""]
    keep = {"regressed", "improved", "suspect"} if show == "flagged" else None
    rows = [r for r in d["tasks"] if keep is None or r["status"] in keep]
    rows.sort(key=lambda r: (r["status"] != "regressed", r["p_regress"]))
    for r in rows:
        out.append(f"  {r['status']:<10} {r['task_id']:<36} {r['base']:>5} -> {r['cand']:<5} "
                   f"p={r['p_regress']:.4f} q={r['q_regress']:.3f}")
    return "\n".join(out)
