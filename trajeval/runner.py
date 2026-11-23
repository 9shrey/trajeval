"""Run an agent over a suite: tasks x trials, a fresh world per trajectory, bounded concurrency,
results appended to runs/<run_id>/results.jsonl as they finish (a killed run resumes)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .scoring import score
from .tasks import Task
from .tools.world import make_env


@dataclass
class RunConfig:
    run_id: str
    agent: str                    # "native" | "text"
    model: str
    base_url: str
    prompt: str = "v1"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    trials: int = 1
    max_steps: int = 8
    max_tokens: int = 512
    concurrency: int = 16
    seed: int = 0
    faults: list[str] = field(default_factory=list)
    tasks: list[str] | None = None      # subset of task ids; None = all
    price: dict | None = None           # {"input": $/Mtok, "output": $/Mtok}
    notes: str = ""


def trajectory_seed(run_seed: int, task_id: str, trial: int) -> int:
    h = hashlib.sha256(f"{run_seed}:{task_id}:{trial}".encode()).hexdigest()
    return int(h[:8], 16)


def _git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:
        return None


async def run_trajectory(agent, task: Task, trial: int, cfg: RunConfig) -> dict:
    env = make_env(cfg.faults)
    seed = trajectory_seed(cfg.seed, task.id, trial)
    t0 = time.perf_counter()
    result = await agent.run(task.prompt, env, max_steps=cfg.max_steps, seed=seed)
    wall = (time.perf_counter() - t0) * 1000
    scored = score(task, result, env, wall, cfg.price)
    return {
        "run_id": cfg.run_id, "task_id": task.id, "category": task.category, "trial": trial, "seed": seed,
        "final_answer": result.final_answer, "error": result.error,
        **scored,
        "tool_calls": [c.to_dict() for c in env.calls],
        "world": env.world.snapshot(),
        "messages": result.messages, "llm_calls": result.llm_calls,
    }


async def run_suite(agent, tasks: list[Task], cfg: RunConfig, runs_dir: Path, progress=print) -> Path:
    out_dir = runs_dir / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    from .store import read_results
    done = {(r["task_id"], r["trial"]) for r in read_results(out_dir)}
    meta = {**cfg.__dict__, "agent_name": agent.name, "git_rev": _git_rev(), "host": platform.node(),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_tasks": len(tasks)}
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta["started"] = json.loads(meta_path.read_text()).get("started", meta["started"])
    meta_path.write_text(json.dumps(meta, indent=2))

    jobs = [(t, k) for k in range(cfg.trials) for t in tasks if (t.id, k) not in done]
    sem = asyncio.Semaphore(cfg.concurrency)
    lock = asyncio.Lock()
    n_done, n_correct, t_start = 0, 0, time.perf_counter()

    async def one(task: Task, trial: int):
        nonlocal n_done, n_correct
        async with sem:
            rec = await run_trajectory(agent, task, trial, cfg)
        async with lock:
            with results_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            n_correct += rec["metrics"]["correct"] is True
            if n_done % 10 == 0 or n_done == len(jobs):
                el = time.perf_counter() - t_start
                progress(f"[{cfg.run_id}] {n_done}/{len(jobs)} trajectories, {el:.0f}s, "
                         f"{n_correct} correct so far (judged tasks pending)")

    await asyncio.gather(*(one(t, k) for t, k in jobs))
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=2))
    return out_dir
