import json

import pytest

from trajeval import export as export_mod
from trajeval import labels as labels_mod
from trajeval.judge import JUDGE_CONFIGS, build_prompt, judge_one, parse_verdict


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def labdir(tmp_path):
    samples = [{"sample_id": f"s{i:03d}", "task_id": f"open-{i % 4}", "run_id": "r", "trial": 0,
                "split": "dev" if i % 4 < 2 else "test", "prompt": "p", "rubric": ["r1"], "reference": "ref",
                "answer": f"answer {i}"} for i in range(40)]
    truth = ["pass" if i % 3 else "fail" for i in range(40)]
    write_jsonl(tmp_path / "samples.jsonl", samples)
    write_jsonl(tmp_path / "labels_claude.jsonl",
                [{"sample_id": s["sample_id"], "label": t, "labeled_by": "claude-opus-5.5"} for s, t in zip(samples, truth)])
    # the human flips two of Claude's labels
    write_jsonl(tmp_path / "labels_human.jsonl",
                [{"sample_id": "s000", "label": "pass", "labeled_by": "human"},
                 {"sample_id": "s001", "label": "fail", "labeled_by": "human"}])
    perfect = [{"sample_id": s["sample_id"], "verdict": t} for s, t in zip(samples, truth)]
    write_jsonl(tmp_path / "judgments" / "perfect.jsonl", perfect)
    write_jsonl(tmp_path / "judgments" / "yes-man.jsonl", [{"sample_id": s["sample_id"], "verdict": "pass"} for s in samples])
    return tmp_path


def test_agreement_sources_and_degenerate_judge(labdir):
    a = labels_mod.agreement("perfect", labdir, source="claude")
    assert a["all"]["kappa"] == 1.0 and a["dev"]["n"] + a["test"]["n"] == 40
    assert a["all"]["labeled_by"] == {"claude-opus-5.5": 40}
    b = labels_mod.agreement("perfect", labdir, source="best")       # human labels override 2 of them
    assert b["all"]["accuracy"] == 38 / 40 and b["all"]["labeled_by"] == {"claude-opus-5.5": 38, "human": 2}
    h = labels_mod.agreement("perfect", labdir, source="human")
    assert h["all"]["n"] == 2 and h["all"]["accuracy"] == 0.0
    y = labels_mod.agreement("yes-man", labdir, source="claude")
    # a judge that always says pass gets 2/3 accuracy here but zero kappa: why kappa is reported
    assert abs(y["all"]["accuracy"] - 26 / 40) < 1e-9 and y["all"]["kappa"] == 0.0
    assert y["all"]["confusion"]["rows_truth_cols_judge"] == [[26, 0], [14, 0]]


def test_verdict_parsing_and_prompt():
    assert parse_verdict("1. ok\n2. missing\nVERDICT: FAIL") == "fail"
    assert parse_verdict("**Verdict:** PASS") == "pass"   # markdown bold is common
    assert parse_verdict("I think it passes.") == "unparsed"  # no verdict line: not guessed at
    assert parse_verdict("VERDICT: PASS ... actually VERDICT: FAIL") == "fail"   # the last one wins
    assert parse_verdict("verdict - pass") == "pass"
    p = build_prompt("task", ["a", "b"], "the ref", "ans", use_reference=False)
    assert "the ref" not in p and "1. a\n2. b" in p
    assert "the ref" in build_prompt("task", ["a"], "the ref", "ans", use_reference=True)


def test_majority_vote_and_empty_answer():
    import asyncio

    from trajeval.llm import LLMResponse

    class C:
        model = "fake"

        def __init__(self, outs):
            self.outs = list(outs)

        async def chat(self, *a, **k):
            return LLMResponse({"role": "assistant", "content": self.outs.pop(0)}, "stop", 10, 5, 1.0)

    cfg = JUDGE_CONFIGS["majority3"]
    r = asyncio.run(judge_one(C(["VERDICT: PASS", "VERDICT: FAIL", "VERDICT: PASS"]), cfg, "p", ["r"], "ref", "ans"))
    assert r["verdict"] == "pass" and r["votes"] == ["pass", "fail", "pass"]
    r = asyncio.run(judge_one(C(["VERDICT: PASS", "no idea", "VERDICT: FAIL"]), cfg, "p", ["r"], "ref", "ans"))
    assert r["verdict"] == "fail"  # 1-1 tie with one unparsed: ties fail
    r = asyncio.run(judge_one(C([]), cfg, "p", ["r"], "ref", "   "))
    assert r["verdict"] == "fail" and r["judge_tokens"] == 0


def test_export_writes_browser_valid_json(tmp_path, monkeypatch, labdir):
    from tests.test_diff import fake_run
    runs = tmp_path / "runs"
    for rid, p in [("qwen1.5b-v2", 0.8), ("qwen1.5b-v2-rerun", 0.8)]:
        run = fake_run(rid, {f"lookup-t{i}": p for i in range(10)}, seed=hash(rid) % 100)
        d = runs / rid
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps(run["meta"] | {"notes": "", "started": "x"}))
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run["results"]))
    monkeypatch.setattr(export_mod, "LABELS_DIR", labdir)
    out = tmp_path / "data"
    export_mod.export(out, runs_dir=runs)
    idx = json.loads((out / "index.json").read_text())
    assert [r["run_id"] for r in idx["runs"]] == ["qwen1.5b-v2", "qwen1.5b-v2-rerun"]  # experiments.yaml order
    assert idx["diffs"][0]["kind"] == "aa"
    for p in out.rglob("*.json"):
        assert "NaN" not in p.read_text(), p  # JSON.parse in a browser rejects NaN


def test_build_samples_quota_split_and_sources(tmp_path):
    from tests.test_diff import fake_run
    from trajeval.tasks import load_suite
    open_ids = [t.id for t in load_suite() if t.is_judged]
    for rid in ("agentA", "agentB", "goldA"):
        run = fake_run(rid, {t: 0.5 for t in open_ids}, trials=5)
        for r in run["results"]:
            r["metrics"]["correct"] = None
            r["category"] = "open_ended"
        d = tmp_path / rid
        d.mkdir()
        (d / "meta.json").write_text(json.dumps(run["meta"]))
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run["results"]))
    s = labels_mod.build_samples(["agentA", "agentB"], n=200, runs_dir=tmp_path, gold_path_run_ids=["goldA"])
    assert len(s) == 200 and len({x["sample_id"] for x in s}) == 200
    per_task = {}
    for x in s:
        per_task.setdefault(x["task_id"], []).append(x)
    assert all(len(v) == 10 for v in per_task.values())
    assert all(sum(x["source"] == "gold-path" for x in v) == 5 for v in per_task.values())
    assert {x["run_id"] for x in s if x["source"] == "agent"} == {"agentA", "agentB"}   # both agent runs used
    for v in per_task.values():
        assert len({x["split"] for x in v}) == 1                                          # split by task
    assert labels_mod.build_samples(["agentA", "agentB"], n=200, runs_dir=tmp_path, gold_path_run_ids=["goldA"]) == s


def test_judge_cli_dedupes_and_reuses_label_verdicts(tmp_path, monkeypatch, labdir):
    """Identical answers are judged once; answers already judged in the label set are not re-sent."""
    from tests.test_diff import fake_run
    from trajeval import cli
    from trajeval.tasks import load_suite
    tid = [t.id for t in load_suite() if t.is_judged][0]
    run = fake_run("r1", {tid: 0.5}, trials=4)
    for k, r in enumerate(run["results"]):
        r["final_answer"] = ["same answer", "same answer", "answer 0", ""][k]
        r["metrics"]["correct"] = None
    d = tmp_path / "runs" / "r1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(run["meta"]))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run["results"]))
    samples = [json.loads(x) for x in (labdir / "samples.jsonl").read_text().splitlines()]
    samples[0]["task_id"], samples[0]["answer"] = tid, "answer 0"
    (labdir / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    monkeypatch.setattr("trajeval.labels.LABELS_DIR", labdir)
    sent = []

    async def fake_judge_items(client, cfg, items, concurrency):
        sent.extend(items)
        return [{"verdict": "pass", "rationale": "ok"} for _ in items]

    monkeypatch.setattr(cli, "_judge_items", fake_judge_items)
    monkeypatch.setattr(cli, "_client", lambda a: None)
    cli.main(["--runs-dir", str(tmp_path / "runs"), "judge", "r1", "--model", "m", "--reuse-labels", "perfect"])
    assert [x["answer"] for x in sent] == ["same answer"]          # deduped, and "answer 0" reused
    out = [json.loads(x) for x in (d / "judgments" / "default.jsonl").read_text().splitlines()]
    assert [o["verdict"] for o in out] == ["pass", "pass", "fail", "fail"]   # reused s000 verdict (fail); empty fails


def test_blind_review_records_first_decision_and_claude_vs_human(labdir, monkeypatch):
    (labdir / "labels_human.jsonl").unlink()
    answers = iter(["p", "x", "q"])                       # label s000 pass, s001 fail, then quit
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    labels_mod.review(labdir, reviewer="tester", blind=True)
    rows = [json.loads(x) for x in (labdir / "labels_human.jsonl").read_text().splitlines()]
    assert [(r["sample_id"], r["label"], r["blind"]) for r in rows] == [("s000", "pass", True), ("s001", "fail", True)]
    c = labels_mod.claude_vs_human(labdir)                # Claude: s000 fail, s001 pass -> both disagree
    assert c["n"] == 2 and c["blind"] and c["agreement"] == 0.0
