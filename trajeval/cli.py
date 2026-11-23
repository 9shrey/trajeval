"""trajeval command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .store import RUNS_DIR, list_runs, load_run, summarize


def _agent(a):
    from .agents.loop import GoldPathAgent, NativeToolsAgent, TextToolsAgent
    from .llm import ChatClient
    from .tasks import load_suite
    client = ChatClient(a.base_url, a.model)
    kw = dict(prompt=a.prompt, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_tokens)
    if a.agent == "gold-path":
        return GoldPathAgent(client, gold_calls={t.prompt: t.gold_calls for t in load_suite(a.suite)}, **kw), client
    cls = {"native": NativeToolsAgent, "text": TextToolsAgent}[a.agent]
    return cls(client, **kw), client


def cmd_run(a):
    from .runner import RunConfig, run_suite
    from .tasks import load_suite
    tasks = load_suite(a.suite)
    if a.tasks:
        want = set(a.tasks.split(","))
        tasks = [t for t in tasks if t.id in want or t.category in want]
    price = json.loads(a.price) if a.price else None
    cfg = RunConfig(run_id=a.run_id, agent=a.agent, model=a.model, base_url=a.base_url, prompt=a.prompt,
                    temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, trials=a.trials, max_steps=a.max_steps, max_tokens=a.max_tokens,
                    concurrency=a.concurrency, seed=a.seed, faults=a.fault or [], tasks=a.tasks and a.tasks.split(","),
                    price=price, notes=a.notes)
    agent, client = _agent(a)

    async def go():
        try:
            return await run_suite(agent, tasks, cfg, Path(a.runs_dir), progress=lambda s: print(s, flush=True))
        finally:
            await client.aclose()

    out = asyncio.run(go())
    print_summary(summarize(load_run(a.run_id, Path(a.runs_dir), judge=None)))
    print(f"results: {out}/results.jsonl")


def print_summary(s: dict):
    pr = s["pass_rate"]
    print(f"\n{s['run_id']}: {s['model']} ({s['agent']}, prompt {s['prompt']}, T={s['temperature']}, "
          f"{s['trials']} trial(s), faults={s['faults'] or 'none'})")
    print(f"  pass rate {pr:.1%} over {s['graded']}/{s['trajectories']} graded trajectories")
    print("  by category: " + ", ".join(f"{k} {v:.0%}" for k, v in s["pass_rate_by_category"].items()))
    print(f"  tool calls/task {s['tool_calls_per_task']:.2f}  invalid {s['invalid_call_rate']:.1%} "
          f"(unknown {s['unknown_tool_rate']:.1%}, schema {s['schema_error_rate']:.1%}, "
          f"malformed {s['malformed_rate']:.1%})  redundant {s['redundant_call_rate']:.1%}  "
          f"loops {s['loop_rate']:.1%}  hit max steps {s['max_steps_rate']:.1%}")
    print(f"  efficiency {s['efficiency']:.2f}  steps {s['steps']:.2f}  tokens/task {s['tokens_per_task']:.0f}  "
          f"wall p50 {s['wall_s_p50']:.1f}s p95 {s['wall_s_p95']:.1f}s  "
          f"unrequested side effects {s['unrequested_side_effects']}  "
          f"invented recipient addresses {s['ungrounded_recipient_rate']:.0%}")


def cmd_report(a):
    ids = a.run_ids or list_runs(Path(a.runs_dir))
    for rid in ids:
        print_summary(summarize(load_run(rid, Path(a.runs_dir), judge=a.judge)))


def cmd_diff(a):
    from .diff import diff_runs, format_diff
    base = load_run(a.base, Path(a.runs_dir), judge=a.judge)
    cand = load_run(a.cand, Path(a.runs_dir), judge=a.judge)
    d = diff_runs(base, cand, alpha=a.alpha, checker_only=a.checker_only)
    print(format_diff(d, show="all" if a.all else "flagged"))
    if a.json:
        Path(a.json).write_text(json.dumps(d, indent=1))
    if a.fail_on_regression and d["counts"]["regressed"]:
        sys.exit(1)


def _client(a):
    from .llm import ChatClient
    return ChatClient(a.base_url, a.model)


async def _judge_items(client, cfg, items, concurrency):
    from .judge import judge_many
    try:
        return await judge_many(client, cfg, items, concurrency=concurrency)
    finally:
        await client.aclose()


def cmd_judge(a):
    """Judge the open-ended trajectories of runs -> runs/<id>/judgments/<name>.jsonl.

    Identical (task, answer) pairs are judged once, and --reuse-labels seeds the cache with the
    verdicts the same judge config already gave on the labelled samples (a subset of these answers).
    """
    from .judge import JUDGE_CONFIGS
    from .labels import LABELS_DIR, _read
    from .tasks import load_suite
    tasks = {t.id: t for t in load_suite() if t.is_judged}
    cfg = JUDGE_CONFIGS[a.config]
    cache: dict[tuple[str, str], dict] = {}
    if a.reuse_labels:
        samples = {x["sample_id"]: x for x in _read(LABELS_DIR / "samples.jsonl")}
        for j in _read(LABELS_DIR / "judgments" / f"{a.reuse_labels}.jsonl"):
            s = samples[j["sample_id"]]
            cache[(s["task_id"], s["answer"].strip())] = {k: v for k, v in j.items() if k != "sample_id"}
    runs = {rid: load_run(rid, Path(a.runs_dir), judge=None) for rid in a.run_ids}
    todo: dict[tuple[str, str], dict] = {}
    for run in runs.values():
        for r in run["results"]:
            if r["task_id"] in tasks and r["final_answer"] and r["final_answer"].strip():
                key = (r["task_id"], r["final_answer"].strip())
                if key not in cache and key not in todo:
                    t = tasks[r["task_id"]]
                    todo[key] = {"task_id": t.id, "prompt": t.prompt, "rubric": t.rubric, "reference": t.reference,
                                 "answer": r["final_answer"]}
    n_total = sum(1 for run in runs.values() for r in run["results"] if r["task_id"] in tasks)
    print(f"{n_total} open-ended trajectories; {len(cache)} verdicts reused, {len(todo)} unique answers to judge")
    for key, j in zip(todo, asyncio.run(_judge_items(_client(a), cfg, list(todo.values()), a.concurrency))):
        cache[key] = j
    for rid, run in runs.items():
        out = []
        for r in run["results"]:
            if r["task_id"] not in tasks:
                continue
            ans = (r["final_answer"] or "").strip()
            j = cache[(r["task_id"], ans)] if ans else {"verdict": "fail", "rationale": "no final answer"}
            out.append({"run_id": rid, "task_id": r["task_id"], "trial": r["trial"], "judge": cfg.name,
                        "judge_model": a.model, "verdict": j["verdict"], "rationale": j.get("rationale", "")})
        d = Path(a.runs_dir) / rid / "judgments"
        d.mkdir(exist_ok=True)
        (d / f"{a.name}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in out))
        print(f"{rid}: {sum(j['verdict'] == 'pass' for j in out)}/{len(out)} open-ended trajectories pass")


def cmd_sample_labels(a):
    from .labels import LABELS_DIR, build_samples
    samples = build_samples(a.run_ids, n=a.n, seed=a.seed, runs_dir=Path(a.runs_dir),
                            gold_path_run_ids=a.gold_path_runs)
    LABELS_DIR.mkdir(exist_ok=True)
    p = LABELS_DIR / "samples.jsonl"
    if p.exists() and not a.force:
        sys.exit(f"{p} exists; pass --force to overwrite (labels refer to its sample ids)")
    p.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    by = {}
    for s in samples:
        by[s["run_id"]] = by.get(s["run_id"], 0) + 1
    print(f"wrote {len(samples)} samples to {p}: {by}; "
          f"dev {sum(s['split'] == 'dev' for s in samples)}, test {sum(s['split'] == 'test' for s in samples)}")


def cmd_judge_labels(a):
    """Run a judge config over the labelled samples -> labels/judgments/<name>.jsonl"""
    from .judge import JUDGE_CONFIGS
    from .labels import LABELS_DIR, _read
    samples = _read(LABELS_DIR / "samples.jsonl")
    cfg = JUDGE_CONFIGS[a.config]
    out = asyncio.run(_judge_items(_client(a), cfg, samples, a.concurrency))
    d = LABELS_DIR / "judgments"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.name}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in out))
    print(f"judged {len(out)} samples -> {d / (a.name + '.jsonl')}")


def cmd_agree(a):
    from .labels import LABELS_DIR, agreement
    names = a.names or sorted(p.stem for p in (LABELS_DIR / "judgments").glob("*.jsonl"))
    print(f"label source: {a.source}")
    print(f"{'judge':<28} {'split':<5} {'n':>4} {'kappa':>6} {'95% CI':>15} {'acc':>6} {'judge pass':>10} "
          f"{'label pass':>10}  confusion [[TP-pass..]]")
    for n in names:
        r = agreement(n, source=a.source)
        for split in ("dev", "test", "all"):
            if split not in r:
                continue
            x = r[split]
            print(f"{n:<28} {split:<5} {x['n']:>4} {x['kappa']:>6.3f} [{x['kappa_ci'][0]:>5.2f}, {x['kappa_ci'][1]:>5.2f}] "
                  f"{x['accuracy']:>6.1%} {x['judge_pass_rate']:>10.1%} {x['label_pass_rate']:>10.1%}  "
                  f"{x['confusion']['rows_truth_cols_judge']}")


def cmd_label(a):
    from .labels import review
    review(reviewer=a.reviewer)


def cmd_affected(a):
    from .faults import affected_tasks
    from .tasks import load_suite
    aff = affected_tasks(load_suite(), a.fault)
    print(f"{len(aff)} tasks touched by {a.fault}:")
    for t in aff:
        print("  " + t)


def cmd_pack(a):
    from .store import pack
    for rid in a.run_ids or list_runs(Path(a.runs_dir)):
        before, after = pack(Path(a.runs_dir) / rid)
        print(f"{rid}: {before / 1e6:.1f} MB -> {after / 1e6:.2f} MB")


def cmd_export(a):
    from .export import export
    export(Path(a.out), runs_dir=Path(a.runs_dir))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trajeval")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an agent over the task suite")
    r.add_argument("--run-id", required=True)
    r.add_argument("--agent", choices=["native", "text", "gold-path"], default="text",
                   help="gold-path: the harness makes the gold tool calls first, then the model answers (diagnostic)")
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    r.add_argument("--prompt", default="v1")
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--top-p", type=float, default=0.8)
    r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--trials", type=int, default=5)
    r.add_argument("--max-steps", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--fault", action="append", help="e.g. deny:/incidents/ or down:techtrail.example")
    r.add_argument("--tasks", help="comma-separated task ids or categories")
    r.add_argument("--suite", default=None)
    r.add_argument("--price", help='{"input": usd_per_mtok, "output": usd_per_mtok}')
    r.add_argument("--notes", default="")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="summarise runs")
    p.add_argument("run_ids", nargs="*")
    p.add_argument("--judge", default="default")
    p.set_defaults(fn=cmd_report)

    d = sub.add_parser("diff", help="compare two runs, flag significant per-task regressions")
    d.add_argument("base")
    d.add_argument("cand")
    d.add_argument("--alpha", type=float, default=0.10, help="FDR level for the BH-adjusted Fisher test")
    d.add_argument("--judge", default="default")
    d.add_argument("--all", action="store_true")
    d.add_argument("--json")
    d.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any task regressed (for CI)")
    d.add_argument("--checker-only", action="store_true", help="only deterministically graded tasks (no judge noise)")
    d.set_defaults(fn=cmd_diff)

    def llm_args(p):
        p.add_argument("--model", required=True)
        p.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
        p.add_argument("--concurrency", type=int, default=12)

    j = sub.add_parser("judge", help="LLM-judge the open-ended trajectories of runs")
    j.add_argument("run_ids", nargs="+")
    j.add_argument("--config", default="rubric+ref")
    j.add_argument("--name", default="default", help="judgments file name; 'default' is what reports use")
    j.add_argument("--reuse-labels", help="seed with this label-judgment file's verdicts (same config and model)")
    llm_args(j)
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("sample-labels", help="draw open-ended answers from runs for labelling")
    s.add_argument("run_ids", nargs="+")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--gold-path-runs", nargs="*", default=None, help="gold-path runs supplying half of each task's quota")
    s.set_defaults(fn=cmd_sample_labels)

    jl = sub.add_parser("judge-labels", help="run a judge config over the labelled samples")
    jl.add_argument("--config", default="rubric+ref")
    jl.add_argument("--name", required=True)
    llm_args(jl)
    jl.set_defaults(fn=cmd_judge_labels)

    ag = sub.add_parser("agree", help="judge-vs-label agreement: Cohen's kappa, confusion matrix")
    ag.add_argument("names", nargs="*")
    ag.add_argument("--source", default="best", choices=["best", "claude", "human"])
    ag.set_defaults(fn=cmd_agree)

    lb = sub.add_parser("label", help="review the labelled samples in the terminal (records human labels)")
    lb.add_argument("--reviewer", default="human")
    lb.set_defaults(fn=cmd_label)

    af = sub.add_parser("affected", help="tasks whose gold trajectory a fault touches")
    af.add_argument("--fault", action="append", required=True)
    af.set_defaults(fn=cmd_affected)

    pk = sub.add_parser("pack", help="gzip finished runs' results (what the repository stores)")
    pk.add_argument("run_ids", nargs="*")
    pk.set_defaults(fn=cmd_pack)

    ex = sub.add_parser("export", help="build the static dashboard's data files")
    ex.add_argument("--out", default="dashboard/data")
    ex.set_defaults(fn=cmd_export)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
