"""Reproduce every number in the README: runs, judge calibration, judging, export.

    python scripts/reproduce.py            # everything (hours on a 4 GB GTX 1650)
    python scripts/reproduce.py --step runs
    python scripts/reproduce.py --step judges
    python scripts/reproduce.py --step export

Reads experiments.yaml. Brings up one model server at a time (the GPU holds one), waits for it,
runs what needs it, and shuts it down. Finished (task, trial) pairs are skipped, so an interrupted
reproduce resumes where it stopped. Needs: vLLM in a separate venv (VLLM_BIN, see
scripts/serve_vllm.sh) and nanoserve (NANOSERVE_DIR) for the 0.5B runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = yaml.safe_load((ROOT / "experiments.yaml").read_text())
TRAJEVAL = [sys.executable, "-m", "trajeval.cli"]
LOGS = ROOT / "logs"


def start_server(name: str) -> subprocess.Popen:
    s = EXP["servers"][name]
    LOGS.mkdir(exist_ok=True)
    log = open(LOGS / f"server-{name}.log", "w")
    if s["kind"] == "vllm":
        env = dict(os.environ, **{k: str(v) for k, v in s.get("env", {}).items()})
        cmd = [str(ROOT / "scripts" / "serve_vllm.sh"), s["model"], s["served"], str(s["port"])]
    else:
        nd = Path(os.environ.get("NANOSERVE_DIR", ROOT.parent / "nanoserve"))
        env = dict(os.environ)
        cmd = [str(nd / ".venv" / "bin" / "python"), "-m", "nanoserve.server", "--model", s["model"],
               "--port", str(s["port"]), *s.get("args", [])]
    print(f"starting {name}: {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                         cwd=str(ROOT))
    url = f"http://127.0.0.1:{s['port']}/v1/models"
    for _ in range(600):
        if p.poll() is not None:
            raise RuntimeError(f"{name} exited; see logs/server-{name}.log")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                print(f"{name} ready", flush=True)
                return p
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError(f"{name} did not come up")


def stop_server(p: subprocess.Popen) -> None:
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    time.sleep(5)  # let the driver release GPU memory


def base_url(server: str) -> str:
    return f"http://127.0.0.1:{EXP['servers'][server]['port']}/v1"


def run_cmds(server: str, only: set[str] | None = None) -> list[list[str]]:
    cmds = []
    srv = EXP["servers"][server]
    for rid, r in EXP["runs"].items():
        if r["server"] != server or (only and rid not in only):
            continue
        parse = "trajeval (text mode)" if r["agent"] == "text" else "the server's parser (native mode)"
        notes = f"{srv['model']} on {srv['kind']}; tool calls parsed by {parse}"
        c = TRAJEVAL + ["run", "--run-id", rid, "--agent", r["agent"], "--model", EXP["servers"][server]["served"],
                        "--base-url", base_url(server), "--prompt", r["prompt"], "--seed", str(r["seed"]),
                        "--trials", str(r.get("trials", EXP.get("trials", 5))),
                        "--concurrency", str(r.get("concurrency", srv.get("env", {}).get("MAX_SEQS", 16))),
                        "--notes", notes]
        for f in r.get("faults", []):
            c += ["--fault", f]
        cmds.append(c)
    return cmds


def judge_cmds(server: str) -> list[list[str]]:
    cmds = []
    for j in EXP.get("judges", []):
        if j["server"] == server:
            cmds.append(TRAJEVAL + ["judge-labels", "--config", j["config"], "--name", j["name"],
                                    "--model", EXP["servers"][server]["served"],
                                    "--base-url", base_url(server)])
    dj = EXP.get("default_judge")
    if dj and dj["server"] == server:
        reuse = next((j["name"] for j in EXP.get("judges", []) if j["server"] == server and j["config"] == dj["config"]), None)
        judged = [rid for rid, r in EXP["runs"].items() if r["agent"] != "gold-path"]  # diagnostics not judged
        cmds.append(TRAJEVAL + ["judge", *judged, "--config", dj["config"], "--name", "default",
                                *(["--reuse-labels", reuse] if reuse else []),
                                "--model", EXP["servers"][server]["served"], "--base-url", base_url(server)])
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["runs", "judges", "export", "all"], default="all")
    ap.add_argument("--only", help="comma-separated run ids (runs step only)")
    a = ap.parse_args()
    # Two passes: every run first (the labelled samples and the default judging read all runs),
    # then every judge. Each pass brings each server up once.
    plan = []
    if a.step in ("runs", "all"):
        only = set(a.only.split(",")) if a.only else None
        plan += [(srv, run_cmds(srv, only)) for srv in EXP["servers"]]
    if a.step in ("judges", "all"):
        plan += [(srv, judge_cmds(srv)) for srv in EXP["servers"]]
    plan = [(srv, cmds) for srv, cmds in plan if cmds]
    for server, cmds in plan:
        p = start_server(server)
        try:
            for c in cmds:
                print("$", " ".join(c[2:]), flush=True)
                subprocess.run(c, check=True, cwd=str(ROOT))
        finally:
            stop_server(p)
    if a.step in ("export", "all"):
        subprocess.run(TRAJEVAL + ["export"], check=True, cwd=str(ROOT))
    print(json.dumps({"done": a.step}))


if __name__ == "__main__":
    main()
