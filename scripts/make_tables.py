"""Regenerate the README's result tables from runs/ and labels/ (between <!-- table:x --> markers).

    python scripts/make_tables.py            # rewrite README.md in place
    python scripts/make_tables.py --print    # print the tables only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from trajeval.diff import diff_runs
from trajeval.export import _run_ci
from trajeval.faults import affected_tasks, detector_score
from trajeval.labels import LABELS_DIR, agreement
from trajeval.store import list_runs, load_run, summarize
from trajeval.tasks import load_suite

ROOT = Path(__file__).resolve().parents[1]
EXP = yaml.safe_load((ROOT / "experiments.yaml").read_text())
SERVED = {v["served"]: k for k, v in EXP["servers"].items()}
MODEL_NAME = {"qwen3b-awq": "Qwen2.5-3B (AWQ)", "qwen1.5b-awq": "Qwen2.5-1.5B (AWQ)", "qwen0.5b": "Qwen2.5-0.5B"}
SERVER_NAME = {"vllm": "vLLM", "nanoserve": "nanoserve"}


def pct(x, d=0):
    return "–" if x is None or x != x else f"{x * 100:.{d}f}%"


def checks_only(run: dict) -> dict:
    return {"meta": run["meta"], "results": [r for r in run["results"] if r["category"] != "open_ended"]}


def runs_table(runs: dict) -> str:
    rows = ["| run | model | server, tool parsing | prompt | trials | pass rate, checks only: 48 tasks (95% CI) | with judge: 68 tasks | lookup | multi-step | action | trap | open-ended (judge) | invalid calls | tokens / task | time p50 |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rid, run in runs.items():
        s = summarize(run)
        det = checks_only(run)
        sd = summarize(det)
        srv = EXP["servers"][EXP["runs"][rid]["server"]]
        lo, hi = _run_ci(det)
        c = s["pass_rate_by_category"]
        gold = s["agent"] == "gold-path"
        parse = {"text": "trajeval", "native": "server (hermes)", "gold-path": "gold path (diagnostic)"}[s["agent"]]
        faults = " + faults" if s["faults"] else ""
        rows.append(f"| `{rid}` | {MODEL_NAME[EXP['runs'][rid]['server']]} | {SERVER_NAME[srv['kind']]}, {parse} | {s['prompt']}{faults} "
                    f"| {s['trials']} | **{pct(sd['pass_rate'], 1)}** ({pct(lo)}–{pct(hi)}) | {'–' if gold else pct(s['pass_rate'], 1)} | "
                    f"{pct(c.get('lookup'))} | {pct(c.get('multi_step'))} | {pct(c.get('action'))} | {pct(c.get('trap'))} | "
                    f"{'–' if gold else pct(c.get('open_ended'))} | {'–' if gold else pct(s['invalid_call_rate'], 1)} | "
                    f"{s['tokens_per_task']:,.0f} | {s['wall_s_p50']:.0f} s |")
    return "\n".join(rows)


def validity_table(runs: dict) -> str:
    rows = ["| run | tool calls / task | hallucinated tool | schema error | malformed | tool error (valid call) | invented recipient addresses | ran past its turn | loops | hit 8-turn limit | unrequested side effects |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rid, run in runs.items():
        s = summarize(run)
        if s["agent"] == "gold-path":
            continue  # the harness made the calls; validity is not the model's
        rows.append(f"| `{rid}` | {s['tool_calls_per_task']:.2f} | {pct(s['unknown_tool_rate'], 1)} | {pct(s['schema_error_rate'], 1)} | "
                    f"{pct(s['malformed_rate'], 1)} | {pct(s['tool_error_rate'], 1)} | {s['ungrounded_recipients']} of {s['recipients']} | {pct(s['runaway_rate'], 1)} | {pct(s['loop_rate'], 1)} | "
                    f"{pct(s['max_steps_rate'], 1)} | {s['unrequested_side_effects']} |")
    return "\n".join(rows)


def diffs_table(runs: dict) -> tuple[str, str]:
    rows = ["| comparison | suite change in pass rate (95% CI) | regressed (Fisher + BH) | improved | suspect (p < 0.05) | naive: any drop |",
            "|---|---|---:|---:|---:|---:|"]
    det_rows = ""
    for c in EXP["comparisons"]:
        if c["base"] not in runs or c["cand"] not in runs:
            continue
        d = diff_runs(runs[c["base"]], runs[c["cand"]], checker_only=True)
        s, n = d["suite"], d["counts"]
        sig = "" if s["significant"] else ", n.s."
        rows.append(f"| {c['title']} (`{c['base']}` → `{c['cand']}`) | {s['mean_delta'] * 100:+.1f} pts "
                    f"({s['ci'][0] * 100:+.1f} to {s['ci'][1] * 100:+.1f}{sig}) | {n['regressed']} | {n['improved']} | "
                    f"{n['suspect']} | {n['naive_regressed']} |")
        if c["kind"] == "faults":
            det = detector_score(d, affected_tasks(load_suite(), runs[c["cand"]]["meta"]["faults"]))
            dr = ["| rule | flagged | caught (of detectable) | false alarms (untouched tasks) | precision |",
                  "|---|---:|---:|---:|---:|"]
            for k, name in [("bh", "Fisher + BH (q ≤ 0.10)"), ("raw_p05", "Fisher, uncorrected p < 0.05"),
                            ("naive", "any drop in pass rate")]:
                x = det[k]
                caught = "–" if x["recall_detectable"] is None else f"{round(x['recall_detectable'] * len(det['detectable']))} of {len(det['detectable'])}"
                untouched = len(d["tasks"]) - len(det["affected"])
                dr.append(f"| {name} | {x['flagged']} | {caught} | {x['false_alarms']} of {untouched} | {pct(x['precision'])} |")
            det_rows = (f"Faults touch {len(det['affected'])} tasks (gold-trajectory replay); {len(det['detectable'])} of them "
                        f"passed at least {det['min_base_rate']:.0%} of baseline trials, so a drop there is detectable.\n\n" + "\n".join(dr))
    return "\n".join(rows), det_rows


def judge_table() -> str:
    names = [j["name"] for j in EXP.get("judges", []) if (LABELS_DIR / "judgments" / f"{j['name']}.jsonl").exists()]
    if not names:
        return "_No judge results yet._"
    rows = ["| judge | sees | dev kappa (95% CI) | test kappa (95% CI) | test accuracy | test confusion [label pass: judge pass, fail / label fail: judge pass, fail] | judge pass rate vs label (test) |",
            "|---|---|---|---|---:|---|---|"]
    cfgs = {j["name"]: j for j in EXP["judges"]}
    for n in names:
        a = agreement(n)
        sees = {"rubric": "rubric", "rubric+ref": "rubric + reference", "rubric+ref-v2": "rubric + reference, v2 prompt (evidence first)",
                "majority3": "rubric + reference, 3 votes"}[cfgs[n]["config"]]
        d, t = a.get("dev"), a.get("test")
        f = lambda x: f"{x['kappa']:.2f} ({x['kappa_ci'][0]:.2f} to {x['kappa_ci'][1]:.2f})" if x else "–"  # noqa: E731
        cm = t["confusion"]["rows_truth_cols_judge"] if t else None
        rows.append(f"| {MODEL_NAME[cfgs[n]['server']]} | {sees} | {f(d)} | **{f(t)}** | {pct(t['accuracy']) if t else '–'} | "
                    f"{cm[0][0]}, {cm[0][1]} / {cm[1][0]}, {cm[1][1]} | {pct(t['judge_pass_rate'])} vs {pct(t['label_pass_rate'])} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    have = set(list_runs())
    runs = {rid: load_run(rid) for rid in EXP["runs"] if rid in have}
    diffs, detector = diffs_table(runs)
    tables = {"runs": runs_table(runs), "validity": validity_table(runs), "diffs": diffs, "detector": detector,
              "judge": judge_table()}
    if a.print:
        for k, v in tables.items():
            print(f"## {k}\n{v}\n")
        return
    readme = (ROOT / "README.md").read_text()
    for k, v in tables.items():
        readme, n = re.subn(rf"(<!-- table:{k} -->\n)(?:.*?\n)?(<!-- /table:{k} -->)",
                            lambda m, v=v: m.group(1) + v + "\n" + m.group(2), readme, flags=re.S)
        if not n:
            print(f"warning: no <!-- table:{k} --> marker in README")
    (ROOT / "README.md").write_text(readme)
    print("README tables updated")


if __name__ == "__main__":
    main()
