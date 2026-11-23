"""Agent interface. An agent gets a task prompt and a ToolEnv and must route every tool call
through `env.call`; the env keeps the authoritative trajectory."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from ..tools.base import ToolEnv

SYSTEM_PROMPTS = {
    "v1": (
        "You are an assistant for Kestrel Cycles, an e-bike company in Bengaluru, India. Today is 2025-10-24. "
        "Use the tools to look up facts instead of guessing, and only take actions (emails, files, meetings) "
        "that the user asked for. When you have what you need, reply to the user with the answer and no tool call."
    ),
    # v2 says where data lives and how to work; v1 -> v2 is tracked as a config change like any other.
    "v2": (
        "You are an assistant for Kestrel Cycles, an e-bike company in Bengaluru, India. Today is 2025-10-24.\n"
        "Where things are:\n"
        "- Customers, orders, products, employees (with emails and managers) and support tickets are in the SQL "
        "database. Call get_db_schema before your first SQL query so you use the real column names.\n"
        "- Policies, pricing plans, incident reports, logs, invoices, vendor profiles, meeting notes and inventory are "
        "files: use list_dir to see what exists, then read_file.\n"
        "- News, reviews, investors and public policies are on the web: web_search, then fetch_page for details.\n"
        "Rules:\n"
        "1. Never guess a number, name or email address: look it up.\n"
        "2. If one search finds nothing, try another tool or location before concluding something does not exist. "
        "If it really does not exist, say so plainly.\n"
        "3. Use calculate for arithmetic.\n"
        "4. Only send emails, write files or schedule meetings when the user asks you to.\n"
        "5. When you are done, reply to the user with a short, direct answer and no tool call."
    ),
}


@dataclass
class AgentResult:
    final_answer: str | None
    stop_reason: str                      # answered | unparsed_call | max_steps | error
    messages: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Agent(Protocol):
    name: str

    async def run(self, prompt: str, env: ToolEnv, *, max_steps: int, seed: int | None) -> AgentResult: ...


FORMAT_ERROR = ("Error: that looked like a tool call but was not in the required format, so it was not run. "
                "Wrap each call in <tool_call></tool_call> tags with valid JSON, or reply to the user without a tool call.")
_TAGGED = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.S)


# A small model sometimes keeps writing past the end of its turn: it emits <|im_start|> (hidden by
# skip_special_tokens, leaving a bare role line) and then invents the user's reply or the tool's
# result. Everything from the first such boundary on is the model's fiction, not its turn.
_TURN_BOUNDARY = re.compile(r"(?:^|\n)(?:user|assistant|system)\n|<tool_response>")


def cut_runaway(content: str) -> tuple[str, bool]:
    """Return (the model's own turn, whether it ran past the end of it)."""
    m = _TURN_BOUNDARY.search(content)
    if not m:
        return content, False
    return content[: m.start()].rstrip(), True


@dataclass
class ParsedCall:
    name: str | None
    args: dict | str      # dict if parsed, raw text otherwise


def parse_text_tool_calls(content: str) -> tuple[list[ParsedCall], str]:
    """Parse Hermes/Qwen-style tool calls from model text.

    Returns (calls, kind) where kind is "tagged" (<tool_call> blocks found), "untagged" (a bare
    JSON object with a "name" key: small models often drop the tags; recorded as a format
    violation and not executed) or "none" (a plain answer).
    """
    bodies = [m.strip() for m in _TAGGED.findall(content)]
    kind = "tagged"
    if not bodies:
        stripped = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if stripped.startswith("{") and '"name"' in stripped:
            bodies, kind = [stripped], "untagged"
        else:
            return [], "none"
    calls = []
    for body in bodies:
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', body)
            calls.append(ParsedCall(m.group(1) if m else None, body))
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            calls.append(ParsedCall(None, body))
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append(ParsedCall(obj["name"], args if isinstance(args, dict) else json.dumps(args)))
    return calls, kind


def execute(env: ToolEnv, call: ParsedCall, step: int) -> str:
    """Run one parsed call through the env (which classifies it) and return the tool result text."""
    if isinstance(call.args, dict):
        return env.call(call.name or "<unparsed>", call.args, step=step).result
    if call.name in env.tools:
        return env.call(call.name, call.args, step=step).result  # unparseable args -> malformed_args
    return env.record_malformed(call.args, step, call.name).result
