"""Tools, the environment that owns them, and the record of every call.

The environment, not the agent, is the source of truth for a trajectory: every tool call
goes through `ToolEnv.call`, which validates it, executes it and records the outcome. An
agent built on any framework (raw function calling, LangGraph, ...) is scored identically as
long as its tool calls are routed through the env.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import jsonschema

# Outcome classes for a single tool call. Only "ok" and "tool_error" are valid calls;
# the rest are the model's fault.
OK = "ok"
TOOL_ERROR = "tool_error"          # valid call, the tool itself reported an error (file not found, SQL error)
UNKNOWN_TOOL = "unknown_tool"      # hallucinated tool name
MALFORMED = "malformed_args"       # arguments were not a JSON object
SCHEMA_ERROR = "schema_error"      # missing/extra/wrong-typed arguments
INVALID_STATUSES = (UNKNOWN_TOOL, MALFORMED, SCHEMA_ERROR)

MAX_RESULT_CHARS = 2000


class ToolFailure(Exception):
    """Raised by a tool for an expected failure; the message is returned to the agent."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    fn: Callable[..., Any]
    side_effect: bool = False  # mutates the world (write_file, send_email, ...)

    def spec(self) -> dict:
        """OpenAI function-calling tool spec."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class ToolCall:
    step: int                 # index of the LLM turn that issued the call
    name: str
    raw_args: str             # exactly what the model produced
    args: dict | None         # parsed arguments, None if unparseable
    status: str
    result: str
    latency_ms: float
    state_version: int        # world-state version *before* the call; used for redundancy detection

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolEnv:
    tools: dict[str, Tool]
    world: Any = None
    calls: list[ToolCall] = field(default_factory=list)
    state_version: int = 0
    max_result_chars: int = MAX_RESULT_CHARS

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, raw_args: str | dict | None, step: int = 0) -> ToolCall:
        t0 = time.perf_counter()
        version = self.state_version
        args: dict | None = None
        if isinstance(raw_args, dict):
            args, raw = raw_args, json.dumps(raw_args)
        else:
            raw = raw_args if raw_args is not None else ""
            try:
                parsed = json.loads(raw) if raw.strip() else {}
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                pass

        tool = self.tools.get(name)
        if tool is None:
            status, result = UNKNOWN_TOOL, f"Error: no tool named {name!r}. Available tools: {', '.join(self.tools)}."
        elif args is None:
            status, result = MALFORMED, "Error: arguments must be a JSON object."
        else:
            err = _schema_error(tool.parameters, args)
            if err:
                status, result = SCHEMA_ERROR, f"Error: invalid arguments for {name}: {err}"
            else:
                try:
                    out = tool.fn(**args)
                    status, result = OK, out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                    if tool.side_effect:
                        self.state_version += 1
                except ToolFailure as e:
                    status, result = TOOL_ERROR, f"Error: {e}"
                except Exception as e:  # a tool bug should not crash a run; it is recorded
                    status, result = TOOL_ERROR, f"Error: {type(e).__name__}: {e}"
        if len(result) > self.max_result_chars:
            result = result[: self.max_result_chars] + f"\n... [truncated, {len(result)} chars total]"
        rec = ToolCall(step, name, raw, args, status, result, (time.perf_counter() - t0) * 1000, version)
        self.calls.append(rec)
        return rec


    def record_malformed(self, raw: str, step: int, name: str | None = None,
                         message: str = "Error: could not parse the tool call.") -> ToolCall:
        """Record an attempted call that never reached a tool (broken envelope / format violation)."""
        rec = ToolCall(step, name or "<unparsed>", raw, None, MALFORMED, message, 0.0, self.state_version)
        self.calls.append(rec)
        return rec


def _schema_error(schema: dict, args: dict) -> str | None:
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as e:
        where = "/".join(str(p) for p in e.absolute_path)
        return f"{where + ': ' if where else ''}{e.message}"
    return None


def params(required: dict[str, tuple[str, str]], optional: dict[str, tuple[str, str]] | None = None) -> dict:
    """Build a strict JSON schema: {arg: (json_type, description)}; no extra arguments allowed."""
    optional = optional or {}
    props = {k: {"type": t, "description": d} for k, (t, d) in {**required, **optional}.items()}
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}
