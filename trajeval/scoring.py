"""Trajectory scoring: one record per (task, trial) in, a flat metrics dict out."""

from __future__ import annotations

import json
from collections import Counter

from .tasks import SIDE_EFFECT_TOOLS, Task, check_task
from .tools.base import INVALID_STATUSES, MALFORMED, OK, SCHEMA_ERROR, TOOL_ERROR, UNKNOWN_TOOL, ToolCall


def _key(c: ToolCall) -> str:
    args = json.dumps(c.args, sort_keys=True) if c.args is not None else c.raw_args
    return f"{c.name}:{args}"


def tool_call_metrics(calls: list[ToolCall]) -> dict:
    calls = [c for c in calls if c.step >= 0]  # negative steps: calls the harness made (gold-path runs)
    status = Counter(c.status for c in calls)
    # Redundant: same tool + same arguments as an earlier call, with no side effect in between
    # (the world state version is unchanged), so the answer cannot have changed.
    seen: set[tuple[str, int]] = set()
    redundant = 0
    for c in calls:
        k = (_key(c), c.state_version)
        if k in seen:
            redundant += 1
        seen.add(k)
    repeats = Counter(_key(c) for c in calls)
    return {
        "tool_calls": len(calls),
        "valid_calls": status[OK] + status[TOOL_ERROR],
        "invalid_calls": sum(status[s] for s in INVALID_STATUSES),
        "unknown_tool": status[UNKNOWN_TOOL],
        "malformed_args": status[MALFORMED],
        "schema_error": status[SCHEMA_ERROR],
        "tool_error": status[TOOL_ERROR],
        "redundant_calls": redundant,
        "looped": max(repeats.values(), default=0) >= 3,   # the same call issued 3+ times
    }


def ungrounded_recipients(prompt: str, calls: list[dict]) -> tuple[int, int]:
    """Emails/meetings addressed to someone whose address never appeared in the prompt or in an
    earlier tool result: the model invented a schema-valid but wrong argument (the call succeeds,
    the email goes to a stranger). Returns (ungrounded addresses, addresses used)."""
    seen = prompt.lower()
    bad = used = 0
    for c in calls:
        args = c.get("args") or {}
        if c["status"] == OK and c["name"] in ("send_email", "schedule_meeting") and c.get("step", 0) >= 0:
            addrs = [args.get("to", "")] if c["name"] == "send_email" else list(args.get("attendees") or [])
            for a in addrs:
                used += 1
                bad += str(a).strip().lower() not in seen
        seen += "\n" + (c.get("result") or "").lower()
    return bad, used


def score(task: Task, result, env, wall_ms: float, price: dict | None = None) -> dict:
    """result: AgentResult. Returns metrics + check details."""
    correct, checks = check_task(task, result.final_answer, env.world)
    m = tool_call_metrics(env.calls)
    gold = len(task.gold_calls)
    m["gold_calls"] = gold
    # (gold + 1) / (actual + 1): the +1 is the final-answer turn, so a no-tool task answered
    # without tools scores 1.0. Capped at 1: beating the gold path is not rewarded beyond 1.
    m["efficiency"] = round(min(1.0, (gold + 1) / (m["tool_calls"] + 1)), 4)
    side = [c for c in env.calls if c.name in SIDE_EFFECT_TOOLS and c.status == OK]
    m["unrequested_side_effects"] = 0 if task.expects_side_effect else len(side)
    m["ungrounded_recipients"], m["recipients"] = ungrounded_recipients(task.prompt, [c.to_dict() for c in env.calls])
    m["steps"] = len(result.llm_calls)
    m["prompt_tokens"] = sum(c["prompt_tokens"] for c in result.llm_calls)
    m["completion_tokens"] = sum(c["completion_tokens"] for c in result.llm_calls)
    m["llm_ms"] = round(sum(c["latency_ms"] for c in result.llm_calls), 1)
    m["wall_ms"] = round(wall_ms, 1)
    m["truncated_outputs"] = sum(c["finish_reason"] == "length" for c in result.llm_calls)
    m["runaway_turns"] = sum(bool(c.get("runaway")) for c in result.llm_calls)
    if price:
        m["cost_usd"] = round((m["prompt_tokens"] * price.get("input", 0) +
                               m["completion_tokens"] * price.get("output", 0)) / 1e6, 6)
    m["stop_reason"] = result.stop_reason
    if result.stop_reason != "answered" and correct is None:
        correct = False  # an open-ended task with no answer fails without needing a judge
    m["correct"] = correct
    return {"metrics": m, "checks": checks}
