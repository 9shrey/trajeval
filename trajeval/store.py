"""Reading runs back: results.jsonl + optional judgments.jsonl -> per-trajectory and per-task tables."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

from .scoring import ungrounded_recipients
from .stats import mean, wilson
from .tasks import load_suite

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def read_results(run_dir: Path) -> list[dict]:
    """results.jsonl.gz (packed, committed) plus results.jsonl (appended since); either may be absent."""
    rows = []
    gz, plain = run_dir / "results.jsonl.gz", run_dir / "results.jsonl"
    if gz.exists():
        rows += [json.loads(x) for x in gzip.open(gz, "rt").read().splitlines() if x.strip()]
    if plain.exists():
        rows += [json.loads(x) for x in plain.read_text().splitlines() if x.strip()]
    return rows


def pack(run_dir: Path) -> tuple[int, int]:
    """Fold results.jsonl into results.jsonl.gz (20x smaller; what the repository stores)."""
    rows = read_results(run_dir)
    gz = run_dir / "results.jsonl.gz"
    with gzip.open(gz, "wt") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    plain = run_dir / "results.jsonl"
    before = plain.stat().st_size if plain.exists() else 0
    plain.unlink(missing_ok=True)
    return before, gz.stat().st_size


def load_run(run_id: str, runs_dir: Path = RUNS_DIR, judge: str | None = "default") -> dict:
    """Load a run. Open-ended trajectories get `correct` from the judge named `judge` if a
    judgments file exists (runs/<id>/judgments/<judge>.jsonl); otherwise they stay None."""
    d = runs_dir / run_id
    meta = json.loads((d / "meta.json").read_text())
    recs = read_results(d)
    verdicts = {}
    jp = d / "judgments" / f"{judge}.jsonl" if judge else None
    if jp and jp.exists():
        for line in jp.read_text().splitlines():
            j = json.loads(line)
            verdicts[(j["task_id"], j["trial"])] = j
    prompts = None
    for r in recs:
        m = r["metrics"]
        # Recomputed from the stored trajectory, so every run uses the current definition.
        if prompts is None:
            prompts = {t.id: t.prompt for t in load_suite()}
        m["ungrounded_recipients"], m["recipients"] = ungrounded_recipients(prompts.get(r["task_id"], ""), r["tool_calls"])
        m["graded_by"] = "checker" if m["correct"] is not None else None
        if m["correct"] is None and (r["task_id"], r["trial"]) in verdicts:
            v = verdicts[(r["task_id"], r["trial"])]
            m["correct"] = v["verdict"] == "pass" if v["verdict"] in ("pass", "fail") else None
            m["graded_by"] = f"judge:{judge}"
    recs.sort(key=lambda r: (r["task_id"], r["trial"]))
    return {"meta": meta, "results": recs}


def list_runs(runs_dir: Path = RUNS_DIR) -> list[str]:
    return sorted(p.name for p in runs_dir.iterdir() if (p / "meta.json").exists() and not p.name.startswith("_"))


def per_task(run: dict, checker_only: bool = False) -> dict[str, dict]:
    """checker_only: drop judge-graded (open-ended) tasks, e.g. to gate CI on deterministic checks only."""
    by = defaultdict(list)
    for r in run["results"]:
        if checker_only and r["category"] == "open_ended":
            continue
        by[r["task_id"]].append(r)
    out = {}
    for tid, rs in sorted(by.items()):
        graded = [r for r in rs if r["metrics"]["correct"] is not None]
        p = sum(r["metrics"]["correct"] for r in graded)
        out[tid] = {
            "task_id": tid, "category": rs[0]["category"], "n": len(graded), "passes": p,
            "pass_rate": p / len(graded) if graded else None,
            "ci": wilson(p, len(graded)) if graded else None,
            "steps": mean([r["metrics"]["steps"] for r in rs]),
            "tool_calls": mean([r["metrics"]["tool_calls"] for r in rs]),
            "invalid_calls": mean([r["metrics"]["invalid_calls"] for r in rs]),
            "efficiency": mean([r["metrics"]["efficiency"] for r in rs if r["metrics"]["correct"]]),
            "wall_ms": mean([r["metrics"]["wall_ms"] for r in rs]),
            "tokens": mean([r["metrics"]["prompt_tokens"] + r["metrics"]["completion_tokens"] for r in rs]),
        }
    return out


def summarize(run: dict) -> dict:
    rs = run["results"]
    graded = [r for r in rs if r["metrics"]["correct"] is not None]
    calls = sum(r["metrics"]["tool_calls"] for r in rs)
    invalid = sum(r["metrics"]["invalid_calls"] for r in rs)

    def frac(key):
        return sum(r["metrics"][key] for r in rs) / calls if calls else 0.0

    by_cat = defaultdict(list)
    for r in graded:
        by_cat[r["category"]].append(r["metrics"]["correct"])
    s = {
        "run_id": run["meta"]["run_id"], "model": run["meta"]["model"], "agent": run["meta"]["agent"],
        "prompt": run["meta"]["prompt"], "temperature": run["meta"]["temperature"], "trials": run["meta"]["trials"],
        "faults": run["meta"].get("faults", []),
        "trajectories": len(rs), "graded": len(graded),
        "pass_rate": mean([r["metrics"]["correct"] for r in graded]),
        "pass_rate_by_category": {c: mean(v) for c, v in sorted(by_cat.items())},
        "tool_calls_per_task": calls / len(rs) if rs else 0,
        "invalid_call_rate": invalid / calls if calls else 0.0,
        "unknown_tool_rate": frac("unknown_tool"), "schema_error_rate": frac("schema_error"),
        "malformed_rate": frac("malformed_args"), "tool_error_rate": frac("tool_error"),
        "redundant_call_rate": frac("redundant_calls"),
        "loop_rate": mean([r["metrics"]["looped"] for r in rs]),
        "runaway_rate": mean([r["metrics"].get("runaway_turns", 0) > 0 for r in rs]),
        "max_steps_rate": mean([r["metrics"]["stop_reason"] == "max_steps" for r in rs]),
        "error_rate": mean([r["metrics"]["stop_reason"] == "error" for r in rs]),
        "unrequested_side_effects": sum(r["metrics"]["unrequested_side_effects"] for r in rs),
        "ungrounded_recipients": sum(r["metrics"].get("ungrounded_recipients", 0) for r in rs),
        "recipients": sum(r["metrics"].get("recipients", 0) for r in rs),
        "ungrounded_recipient_rate": (sum(r["metrics"].get("ungrounded_recipients", 0) for r in rs) /
                                      max(1, sum(r["metrics"].get("recipients", 0) for r in rs))),
        "efficiency": mean([r["metrics"]["efficiency"] for r in rs if r["metrics"]["correct"]]),
        "steps": mean([r["metrics"]["steps"] for r in rs]),
        "tokens_per_task": mean([r["metrics"]["prompt_tokens"] + r["metrics"]["completion_tokens"] for r in rs]),
        "wall_s_p50": _pct([r["metrics"]["wall_ms"] / 1000 for r in rs], 50),
        "wall_s_p95": _pct([r["metrics"]["wall_ms"] / 1000 for r in rs], 95),
    }
    if any("cost_usd" in r["metrics"] for r in rs):
        s["cost_usd_per_task"] = mean([r["metrics"].get("cost_usd", 0) for r in rs])
    return s


def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)
