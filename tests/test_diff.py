"""Regression detector on synthetic runs where the truth is known."""

import json
import random

from trajeval.diff import diff_runs
from trajeval.faults import affected_tasks, detector_score
from trajeval.store import load_run, summarize
from trajeval.tasks import load_suite


def fake_run(run_id, rates, trials=5, seed=0):
    """rates: {task_id: true pass probability}. Builds a run dict shaped like store.load_run."""
    rng = random.Random(seed)
    results = []
    for tid, p in rates.items():
        for k in range(trials):
            ok = rng.random() < p
            results.append({"task_id": tid, "category": "lookup", "trial": k, "final_answer": "x", "error": None,
                            "checks": [], "tool_calls": [],
                            "world": {"outbox": [], "meetings": [], "files_changed": {}},
                            "metrics": {"correct": ok, "steps": 2, "tool_calls": 1, "invalid_calls": 0,
                                        "unknown_tool": 0, "schema_error": 0, "malformed_args": 0, "tool_error": 0,
                                        "redundant_calls": 0, "looped": False, "efficiency": 1.0,
                                        "unrequested_side_effects": 0, "prompt_tokens": 100, "completion_tokens": 10,
                                        "wall_ms": 1000.0, "stop_reason": "answered"}})
    meta = {"run_id": run_id, "model": "m", "agent": "native", "prompt": "v1", "temperature": 0.7, "trials": trials,
            "faults": []}
    return {"meta": meta, "results": results}


def test_aa_runs_flag_nothing_but_naive_rule_flags_many():
    rng = random.Random(42)
    rates = {f"t{i}": rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) for i in range(60)}
    a, b = fake_run("a", rates, seed=1), fake_run("b", rates, seed=2)
    d = diff_runs(a, b)
    assert d["counts"]["regressed"] == 0
    assert d["counts"]["naive_regressed"] >= 10       # noise looks like regressions to a naive rule
    assert not d["suite"]["significant"]


def test_real_regressions_found():
    rates = {f"t{i}": 0.9 for i in range(60)}
    broken = dict(rates, **{f"t{i}": 1.0 for i in range(10)})
    after = dict(rates, **{f"t{i}": 0.0 for i in range(10)})
    d = diff_runs(fake_run("a", broken, seed=3), fake_run("b", after, seed=4))
    flagged = {r["task_id"] for r in d["tasks"] if r["status"] == "regressed"}
    assert flagged == {f"t{i}" for i in range(10)}      # every 5/5 -> 0/5 task, and nothing else
    assert d["suite"]["significant"] and d["suite"]["mean_delta"] < 0


def test_power_limit_with_five_trials():
    """Documented limit: one task dropping 4/5 -> 0/5 has Fisher p = 6/252 = 0.024, which does not
    survive BH across 60 tasks. Five trials only detect near-total collapses of a single task."""
    rates = {f"t{i}": 1.0 for i in range(60)}
    a = fake_run("a", rates)
    b = fake_run("b", dict(rates, t0=0.0))
    a["results"][0]["metrics"]["correct"] = False          # t0 baseline 4/5
    row = {r["task_id"]: r for r in diff_runs(a, b)["tasks"]}["t0"]
    assert abs(row["p_regress"] - 6 / 252) < 1e-12 and row["status"] == "suspect"


def test_detector_score_counts():
    rates = {f"t{i}": 1.0 for i in range(20)}
    broken = dict(rates, t0=0.0, t1=0.0)
    d = diff_runs(fake_run("a", rates), fake_run("b", broken))
    s = detector_score(d, ["t0", "t1", "t2"])
    assert s["bh"]["true_positives"] == 2 and s["bh"]["false_alarms"] == 0
    assert s["bh"]["recall_detectable"] == 2 / 3


def test_affected_tasks_from_gold_replay():
    tasks = load_suite()
    aff = affected_tasks(tasks, ["down:techtrail.example"])
    assert set(aff) == {"multi-x1-range-gap", "multi-weight-difference", "open-x1-review-summary"}
    assert affected_tasks(tasks, []) == []


def test_store_roundtrip_with_judgments(tmp_path):
    run = fake_run("r1", {"t0": 1.0}, trials=2)
    run["results"].append({**run["results"][0], "task_id": "open-x", "trial": 0, "category": "open_ended",
                           "metrics": {**run["results"][0]["metrics"], "correct": None}})
    d = tmp_path / "r1"
    (d / "judgments").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(run["meta"]))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run["results"]))
    assert summarize(load_run("r1", tmp_path, judge=None))["graded"] == 2
    (d / "judgments" / "default.jsonl").write_text(json.dumps({"task_id": "open-x", "trial": 0, "verdict": "fail"}) + "\n")
    loaded = load_run("r1", tmp_path)
    s = summarize(loaded)
    assert s["graded"] == 3 and abs(s["pass_rate"] - 2 / 3) < 1e-9
    assert [r for r in loaded["results"] if r["task_id"] == "open-x"][0]["metrics"]["graded_by"] == "judge:default"


def test_tool_outage_fault():
    from trajeval.tools.world import make_env
    env = make_env(["break:web_search"])
    rec = env.call("web_search", {"query": "Voltcell"})
    assert rec.status == "tool_error" and "503" in rec.result
    assert env.call("fetch_page", {"url": "https://voltcell.example/about"}).status == "ok"  # other tools fine
    aff = affected_tasks(load_suite(), ["break:convert_currency"])
    assert aff == ["lookup-fx-usd", "multi-unpaid-invoices-inr"]


def test_pack_roundtrip_and_resume_reads_both(tmp_path):
    from trajeval.store import pack, read_results
    d = tmp_path / "r"
    d.mkdir()
    rows = [{"task_id": "t", "trial": k, "x": "é"} for k in range(3)]
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pack(d)
    assert not (d / "results.jsonl").exists() and read_results(d) == rows
    (d / "results.jsonl").write_text(json.dumps({"task_id": "t", "trial": 3}) + "\n")  # appended after packing
    assert [r["trial"] for r in read_results(d)] == [0, 1, 2, 3]


def test_checker_only_ignores_judge_graded_tasks():
    rates = {f"t{i}": 1.0 for i in range(10)}
    a, b = fake_run("a", rates), fake_run("b", dict(rates, t0=0.0, t1=0.0, t2=0.0))
    for run in (a, b):
        for r in run["results"]:
            if r["task_id"] in ("t0", "t1", "t2"):
                r["category"] = "open_ended"      # pretend these three were judge-graded
    assert diff_runs(a, b)["counts"]["regressed"] == 3
    d = diff_runs(a, b, checker_only=True)
    assert d["counts"]["regressed"] == 0 and len(d["tasks"]) == 7
