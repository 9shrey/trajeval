"""Labelled samples for judge calibration, and judge-vs-label agreement.

Files (labels/):
  samples.jsonl          one open-ended answer per line: sample_id, task, rubric, reference, answer, split
  labels_claude.jsonl    pass/fail labels written by Claude (an LLM) as a first pass, with rationales
  labels_human.jsonl     labels from a human reviewer (`trajeval label`); these take precedence
  judgments/<name>.jsonl the judge's verdicts on every sample, per judge config

Nothing here relabels Claude's labels as human: `labeled_by` is carried through to every report.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .stats import bootstrap_ci, cohen_kappa, confusion
from .store import RUNS_DIR, load_run
from .tasks import load_suite

LABELS_DIR = Path(__file__).resolve().parents[1] / "labels"


def _read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def build_samples(run_ids: list[str], n: int = 200, seed: int = 0, runs_dir: Path = RUNS_DIR,
                  dev_frac: float = 0.5, gold_path_run_ids: list[str] | None = None) -> list[dict]:
    """Draw n open-ended answers: an equal quota per task, split evenly between the agent runs and
    (if given) gold-path runs, alternating runs within each pool. Small agents mostly fail
    open-ended tasks, so gold-path answers are what puts enough passes in the set to measure a
    judge at all (with one class near-empty, kappa is undefined in practice). The dev/test split
    is by *task*, so a judge prompt tuned on dev never saw the test tasks' rubrics."""
    tasks = {t.id: t for t in load_suite() if t.is_judged}
    pools = [("agent", run_ids)] + ([("gold-path", gold_path_run_ids)] if gold_path_run_ids else [])
    answers: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for source, rids in pools:
        for rid in rids:
            for r in load_run(rid, runs_dir, judge=None)["results"]:
                if r["task_id"] in tasks and r["final_answer"]:
                    answers[(r["task_id"], source)][rid].append(r)
    rng = random.Random(seed)
    tids = sorted(tasks)
    dev = set(rng.sample(tids, round(len(tids) * dev_frac)))
    per_task = n // len(tids)
    out = []
    for tid in tids:
        # Gold-path answers get up to half the quota; whatever a pool cannot supply comes from the other.
        remaining = per_task
        for p_i, (source, rids) in enumerate(pools):
            last = p_i == len(pools) - 1
            want = remaining if last else min(remaining, per_task // len(pools))
            if not last and p_i == 0 and len(pools) > 1:
                gold_have = sum(len(v) for v in answers[(tid, pools[1][0])].values())
                want = max(want, remaining - min(gold_have, per_task // 2))
            picked = _round_robin(answers[(tid, source)], rids, want, rng)
            for r, rid in picked:
                t = tasks[tid]
                out.append({"sample_id": f"s{len(out):03d}", "run_id": rid, "source": source, "task_id": tid,
                            "trial": r["trial"], "split": "dev" if tid in dev else "test", "prompt": t.prompt,
                            "rubric": t.rubric, "reference": t.reference, "answer": r["final_answer"]})
            remaining -= len(picked)
    return out


def _round_robin(by_run: dict[str, list], rids: list[str], want: int, rng: random.Random) -> list[tuple[dict, str]]:
    for rid in sorted(by_run):
        rng.shuffle(by_run[rid])
    order = [rid for rid in rids if by_run.get(rid)]
    picked, k = [], 0
    while len(picked) < want and any(by_run[r] for r in order):
        rid = order[k % len(order)]
        k += 1
        if by_run[rid]:
            picked.append((by_run[rid].pop(), rid))
    return picked


def load_labels(labels_dir: Path = LABELS_DIR, source: str = "best") -> dict[str, dict]:
    """source: 'claude', 'human', or 'best' (human where reviewed, else claude)."""
    claude = {x["sample_id"]: x for x in _read(labels_dir / "labels_claude.jsonl")}
    human = {x["sample_id"]: x for x in _read(labels_dir / "labels_human.jsonl")}
    if source == "claude":
        return claude
    if source == "human":
        return human
    return {**claude, **human}


def agreement(judge_name: str, labels_dir: Path = LABELS_DIR, source: str = "best") -> dict:
    samples = {s["sample_id"]: s for s in _read(labels_dir / "samples.jsonl")}
    labels = load_labels(labels_dir, source)
    judg = {j["sample_id"]: j for j in _read(labels_dir / "judgments" / f"{judge_name}.jsonl")}
    out = {"judge": judge_name, "label_source": source}
    for split in ("dev", "test", "all"):
        ids = [i for i in samples if i in labels and i in judg and (split == "all" or samples[i]["split"] == split)]
        if not ids:
            continue
        truth = [labels[i]["label"] for i in ids]
        pred = [judg[i]["verdict"] if judg[i]["verdict"] in ("pass", "fail") else "fail" for i in ids]
        pairs = list(zip(truth, pred))
        k = cohen_kappa(truth, pred)
        lo, hi = bootstrap_ci(pairs, lambda ps: cohen_kappa([a for a, _ in ps], [b for _, b in ps]), n_boot=2000)
        cm = confusion(truth, pred, ["pass", "fail"])
        out[split] = {
            "n": len(ids), "kappa": k, "kappa_ci": [lo, hi],
            "accuracy": sum(a == b for a, b in pairs) / len(ids),
            "confusion": {"labels": ["pass", "fail"], "rows_truth_cols_judge": cm},
            "label_pass_rate": truth.count("pass") / len(ids), "judge_pass_rate": pred.count("pass") / len(ids),
            "unparsed": sum(judg[i]["verdict"] == "unparsed" for i in ids),
            "labeled_by": dict(sorted(_count(labels[i].get("labeled_by", "?") for i in ids).items())),
        }
    return out


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


def review(labels_dir: Path = LABELS_DIR, reviewer: str = "human", blind: bool = True) -> None:
    """Terminal review of the labelled samples. Every decision is saved immediately as a human label.

    blind (default): you label each answer without seeing Claude's label; it is revealed after you
    decide, and your first decision is what gets recorded. That keeps your labels independent, so
    Claude-vs-human agreement is a real inter-annotator number. --show-claude shows it up front
    (faster, but anchors you toward agreeing).
    """
    samples = _read(labels_dir / "samples.jsonl")
    claude = load_labels(labels_dir, "claude")
    human = load_labels(labels_dir, "human")
    todo = [s for s in samples if s["sample_id"] not in human]
    keys = "p=pass, x=fail, s=skip, q=quit" if blind else "a=agree, f=flip, p=pass, x=fail, s=skip, q=quit"
    print(f"{len(human)} reviewed, {len(todo)} to go ({'blind' if blind else 'Claude label shown'}). Keys: {keys}")
    print("The labelling rule is in labels/README.md. Progress is saved after every answer.\n")
    out = (labels_dir / "labels_human.jsonl").open("a")
    done, agreed = len(human), 0
    for n, s in enumerate(todo, 1):
        c = claude.get(s["sample_id"], {})
        print("=" * 100)
        print(f"[{done + n}/{len(samples)}] {s['sample_id']}  task {s['task_id']}  ({s['split']})\n")
        print(f"TASK: {s['prompt']}\n\nRUBRIC (pass only if every point holds):")
        for i, r in enumerate(s["rubric"], 1):
            print(f"  {i}. {r}")
        print(f"\nREFERENCE: {s['reference']}\n\nANSWER:\n{s['answer']}\n")
        if c and not blind:
            print(f"Claude's label: {c['label'].upper()}  ({c.get('rationale', '')})")
        allowed = ("p", "x", "s", "q") if blind else ("a", "f", "p", "x", "s", "q")
        while True:
            k = input(f"[{'/'.join(allowed)}] > ").strip().lower()
            if k in ("a", "f") and not c:
                print("no Claude label to agree with or flip; use p or x")
                continue
            if k in allowed:
                break
        if k == "q":
            break
        if k == "s":
            continue
        label = {"a": c.get("label"), "f": {"pass": "fail", "fail": "pass"}.get(c.get("label")),
                 "p": "pass", "x": "fail"}[k]
        out.write(json.dumps({"sample_id": s["sample_id"], "label": label, "labeled_by": reviewer,
                              "blind": blind, "claude_label": c.get("label")}) + "\n")
        out.flush()
        if blind and c:
            same = c["label"] == label
            agreed += same
            print(f"-> saved {label.upper()}. Claude said {c['label'].upper()}"
                  f"{'' if same else '  <-- disagreement: ' + c.get('rationale', '')}")
    out.close()
    if blind and n:
        print(f"\nThis session: you and Claude agreed on {agreed} answers. `trajeval agree` reports kappa.")


def claude_vs_human(labels_dir: Path = LABELS_DIR) -> dict | None:
    """Inter-annotator agreement between Claude's labels and the human review (blind rows only, if any)."""
    claude, human = load_labels(labels_dir, "claude"), load_labels(labels_dir, "human")
    rows = [h for h in human.values() if h["sample_id"] in claude]
    blind_rows = [h for h in rows if h.get("blind")]
    rows = blind_rows or rows
    if not rows:
        return None
    a = [claude[h["sample_id"]]["label"] for h in rows]
    b = [h["label"] for h in rows]
    return {"n": len(rows), "blind": bool(blind_rows), "kappa": cohen_kappa(a, b),
            "agreement": sum(x == y for x, y in zip(a, b)) / len(rows),
            "confusion": confusion(a, b, ["pass", "fail"])}
