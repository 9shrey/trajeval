"""Build the static dashboard's JSON from runs/, labels/ and experiments.yaml."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .diff import diff_runs
from .faults import detector_score
from .labels import LABELS_DIR, _read, agreement, load_labels
from .stats import bootstrap_ci
from .store import RUNS_DIR, list_runs, load_run, per_task, summarize
from .tasks import load_suite

ROOT = Path(__file__).resolve().parents[1]


def _slim(r: dict) -> dict:
    return {
        "task_id": r["task_id"], "trial": r["trial"], "final_answer": r["final_answer"], "error": r["error"],
        "checks": r["checks"], "metrics": r["metrics"],
        "tool_calls": [{"step": c["step"], "name": c["name"], "args": c["raw_args"][:400], "status": c["status"],
                        "result": c["result"][:400]} for c in r["tool_calls"]],
        "world": {"outbox": r["world"]["outbox"], "meetings": r["world"]["meetings"],
                  "files_changed": {k: v[:300] for k, v in r["world"]["files_changed"].items()}},
    }


def export(out: Path, runs_dir: Path = RUNS_DIR, judge: str = "default") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    (out / "diffs").mkdir(exist_ok=True)
    exp = yaml.safe_load((ROOT / "experiments.yaml").read_text())
    tasks = load_suite()
    runs = {}
    order = list(exp.get("runs", {}))
    rids = sorted(list_runs(runs_dir), key=lambda r: (order.index(r) if r in order else len(order), r))
    for rid in rids:
        run = load_run(rid, runs_dir, judge=judge)
        runs[rid] = run
        (out / "runs" / f"{rid}.json").write_text(_dump({
            "summary": summarize(run), "meta": run["meta"], "tasks": per_task(run),
            "trajectories": [_slim(r) for r in run["results"]],
        }, separators=(",", ":")))

    diffs = []
    for c in exp.get("comparisons", []):
        if c["base"] not in runs or c["cand"] not in runs:
            continue
        d = diff_runs(runs[c["base"]], runs[c["cand"]], checker_only=True)
        d["title"], d["kind"] = c["title"], c["kind"]
        if c["kind"] == "faults":
            from .faults import affected_tasks
            d["detector"] = detector_score(d, affected_tasks(tasks, runs[c["cand"]]["meta"]["faults"]))
        name = f"{c['base']}__{c['cand']}"
        (out / "diffs" / f"{name}.json").write_text(_dump(d, separators=(",", ":")))
        diffs.append({"id": name, "title": c["title"], "kind": c["kind"], "base": c["base"], "cand": c["cand"],
                      "counts": d["counts"], "suite": d["suite"], "detector": d.get("detector")})

    judge_data = None
    if (LABELS_DIR / "samples.jsonl").exists():
        samples = _read(LABELS_DIR / "samples.jsonl")
        labels = load_labels(LABELS_DIR, "best")
        names = sorted(p.stem for p in (LABELS_DIR / "judgments").glob("*.jsonl")) if (LABELS_DIR / "judgments").exists() else []
        verdicts = {n: {j["sample_id"]: j["verdict"] for j in _read(LABELS_DIR / "judgments" / f"{n}.jsonl")} for n in names}
        judge_data = {
            "agreement": [agreement(n) for n in names],
            "samples": [{**{k: s[k] for k in ("sample_id", "task_id", "run_id", "split", "answer")},
                         "label": labels.get(s["sample_id"], {}).get("label"),
                         "labeled_by": labels.get(s["sample_id"], {}).get("labeled_by"),
                         "rationale": labels.get(s["sample_id"], {}).get("rationale"),
                         "verdicts": {n: verdicts[n].get(s["sample_id"]) for n in names}} for s in samples],
            "label_sources": _count([labels.get(s["sample_id"], {}).get("labeled_by", "unlabelled") for s in samples]),
        }
        (out / "judge.json").write_text(_dump(judge_data, separators=(",", ":")))

    index = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": {"n_tasks": len(tasks), "by_category": _count([t.category for t in tasks]),
                  "tasks": [{"id": t.id, "category": t.category, "prompt": t.prompt} for t in tasks]},
        "runs": [summarize(runs[r]) | {"ci": _run_ci(runs[r]),
                                       "checks": {"pass_rate": summarize(_checks_only(runs[r]))["pass_rate"],
                                                  "ci": _run_ci(_checks_only(runs[r]))},
                                       "meta": {k: runs[r]["meta"].get(k) for k in ("notes", "started", "git_rev")}}
                 for r in runs],
        "diffs": diffs,
        "judge": {"agreement": judge_data["agreement"], "label_sources": judge_data["label_sources"]} if judge_data else None,
    }
    (out / "index.json").write_text(_dump(index, indent=1))
    print(f"exported {len(runs)} runs, {len(diffs)} diffs, judge={'yes' if judge_data else 'no'} -> {out}")


def _checks_only(run: dict) -> dict:
    return {"meta": run["meta"], "results": [r for r in run["results"] if r["category"] != "open_ended"]}


def _run_ci(run: dict) -> list[float]:
    """95% bootstrap CI of the pass rate, resampling tasks (trials within a task are correlated)."""
    pts = [t for t in per_task(run).values() if t["n"]]
    tot = lambda ts: sum(t["passes"] for t in ts) / max(1, sum(t["n"] for t in ts))  # noqa: E731
    return list(bootstrap_ci(pts, tot, n_boot=2000))


def _clean(x):
    """JSON for browsers: NaN/inf are not valid JSON, so they become null."""
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _dump(obj, **kw) -> str:
    return json.dumps(_clean(obj), allow_nan=False, **kw)


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d
