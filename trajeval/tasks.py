"""Task suite loading and final-answer / world-state checkers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SUITE_DIR = Path(__file__).resolve().parents[1] / "tasks"
CATEGORIES = ("lookup", "multi_step", "action", "trap", "open_ended")


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    check: list[dict]
    gold_calls: list[dict]
    gold_answer: str | None = None
    rubric: list[str] = field(default_factory=list)
    reference: str | None = None

    @property
    def is_judged(self) -> bool:
        return any(c["type"] == "judge" for c in self.check)

    @property
    def expects_side_effect(self) -> bool:
        return any(c["tool"] in SIDE_EFFECT_TOOLS for c in self.gold_calls)


SIDE_EFFECT_TOOLS = {"write_file", "send_email", "schedule_meeting"}


def load_suite(path: str | Path | None = None) -> list[Task]:
    path = Path(path) if path else SUITE_DIR / "kestrel.yaml"
    raw = yaml.safe_load(path.read_text())
    tasks, seen = [], set()
    for t in raw:
        if t["id"] in seen:
            raise ValueError(f"duplicate task id {t['id']}")
        seen.add(t["id"])
        if t["category"] not in CATEGORIES:
            raise ValueError(f"{t['id']}: unknown category {t['category']}")
        tasks.append(Task(id=t["id"], category=t["category"], prompt=t["prompt"], check=t["check"],
                          gold_calls=t["gold"]["calls"], gold_answer=t["gold"].get("answer"),
                          rubric=t.get("rubric", []), reference=t.get("reference")))
    return tasks


# ---------------------------------------------------------------------- checkers
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?\b|\b\d{1,2}:\d{2}\b")
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
REFUSAL_PHRASES = (
    "not found", "no such", "does not exist", "doesn't exist", "do not exist", "don't exist", "no record",
    "couldn't find", "could not find", "can't find", "cannot find", "unable to find", "not able to find",
    "no order", "no customer", "there is no", "there are no", "no results", "no review", "not exist",
    "wasn't able to find", "was not able to find", "did not find", "didn't find", "no information",
    "not in our", "isn't in our", "is not in our", "no matching", "not available",
)


def extract_numbers(text: str) -> list[float]:
    """Numbers in free text: Indian/western digit grouping, currency signs, dates and times ignored."""
    text = _DATE.sub(" ", text)
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _has(text: str, item) -> bool:
    """item is a string, or a list meaning any-of."""
    low = text.lower()
    if isinstance(item, list):
        return any(str(i).lower() in low for i in item)
    return str(item).lower() in low


def run_check(check: dict, answer: str | None, world) -> tuple[bool | None, str]:
    """Return (passed, detail). passed is None for checks the harness cannot grade (judge)."""
    answer = answer or ""
    kind = check["type"]
    if kind == "judge":
        return None, "needs judge"
    if kind == "number":
        target, tol = float(check["value"]), float(check.get("tol", 0.01))
        nums = extract_numbers(answer)
        ok = any(abs(n - target) <= tol for n in nums)
        return ok, f"want {check['value']}, found {nums[:8]}"
    if kind == "contains":
        missing = [i for i in check["items"] if not _has(answer, i)]
        return not missing, f"missing {missing}" if missing else "all present"
    if kind == "not_contains":
        present = [i for i in check["items"] if _has(answer, i)]
        return not present, f"forbidden present {present}" if present else "none present"
    if kind == "refusal":
        ok = _has(answer, list(REFUSAL_PHRASES))
        return ok, "says not found" if ok else "does not say the item was not found"
    if kind == "file":
        path = check["path"]
        content = world.fs.get(path)
        if content is None:
            return False, f"{path} not written"
        missing = [i for i in check.get("contains", []) if not _has(content, i)]
        bad = [i for i in check.get("not_contains", []) if _has(content, i)]
        ok = not missing and not bad
        return ok, f"missing {missing}, forbidden {bad}" if not ok else "file ok"
    if kind == "email":
        to = check["to"].lower()
        cands = [e for e in world.outbox if e["to"] == to]
        if not cands:
            return False, f"no email to {to} (sent to {[e['to'] for e in world.outbox]})"
        for e in cands:
            if all(_has(e["subject"], i) for i in check.get("subject_contains", [])) and \
               all(_has(e["body"], i) for i in check.get("body_contains", [])):
                return True, "email ok"
        return False, "email to the right person, wrong subject/body"
    if kind == "no_email":
        return not world.outbox, f"{len(world.outbox)} emails sent"
    if kind == "meeting":
        want = sorted(a.lower() for a in check["attendees"])
        for m in world.meetings:
            if m["date"] == check["date"] and m["attendees"] == want:
                return True, "meeting ok"
        return False, f"no meeting on {check['date']} with {want}; got {world.meetings}"
    raise ValueError(f"unknown check type {kind}")


def check_task(task: Task, answer: str | None, world) -> tuple[bool | None, list[dict]]:
    """All checks must pass. Returns (correct, per-check details); correct is None for judged tasks."""
    results = []
    for c in task.check:
        ok, detail = run_check(c, answer, world)
        results.append({"type": c["type"], "passed": ok, "detail": detail})
    graded = [r["passed"] for r in results if r["passed"] is not None]
    if task.is_judged:
        return None, results
    return all(graded), results
