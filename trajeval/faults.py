"""Ground truth for the fault-injection experiment: which tasks does a fault actually touch?

A task is *affected* if replaying its gold trajectory under the fault gives any tool result that
differs from the clean world. That set is computed mechanically, before any model runs, so the
regression detector is scored against labels it could not have influenced.

Trap tasks are excluded: their correct response ("there is no order 2077") is still correct when
the database is down, so a fault does not turn them into regressions.
"""

from __future__ import annotations

from .tasks import Task
from .tools.world import make_env


def affected_tasks(tasks: list[Task], faults: list[str]) -> list[str]:
    out = []
    for t in tasks:
        if t.category == "trap":
            continue
        clean, broken = make_env(), make_env(faults)
        for i, c in enumerate(t.gold_calls):
            a = clean.call(c["tool"], c["args"], step=i)
            b = broken.call(c["tool"], c["args"], step=i)
            if (a.status, a.result) != (b.status, b.result):
                out.append(t.id)
                break
    return out


def detector_score(diff: dict, affected: list[str], min_base_rate: float = 0.6) -> dict:
    """How well did the diff's per-task flags recover the affected set?

    A task can only *detectably* regress if it passed often enough at baseline, so recall is
    reported over affected tasks with baseline pass rate >= min_base_rate. False alarms are
    flags on tasks the fault cannot have touched.
    """
    rows = {r["task_id"]: r for r in diff["tasks"]}
    aff = set(affected) & set(rows)
    detectable = {t for t in aff if rows[t]["base_rate"] >= min_base_rate}
    unaffected = set(rows) - aff

    def flagged(rule):
        if rule == "bh":
            return {t for t, r in rows.items() if r["status"] == "regressed"}
        if rule == "raw_p05":
            return {t for t, r in rows.items() if r["p_regress"] < 0.05}
        return {t for t, r in rows.items() if r["naive_regressed"]}

    out = {"affected": sorted(aff), "detectable": sorted(detectable), "min_base_rate": min_base_rate}
    for rule in ("bh", "raw_p05", "naive"):
        f = flagged(rule)
        out[rule] = {
            "flagged": len(f),
            "true_positives": len(f & aff),
            "recall_detectable": len(f & detectable) / len(detectable) if detectable else None,
            "false_alarms": len(f & unaffected),
            "false_alarm_rate": len(f & unaffected) / len(unaffected) if unaffected else None,
            "precision": len(f & aff) / len(f) if f else None,
        }
    return out
