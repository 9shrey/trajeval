"""Two function-calling agents over an OpenAI-compatible chat endpoint.

- `NativeToolsAgent`: passes `tools=[...]` and reads `message.tool_calls` (vLLM with a tool
  parser, hosted APIs). The server does the parsing.
- `TextToolsAgent`: renders the tool list into the system prompt in Qwen2.5's own chat-template
  format and parses `<tool_call>` blocks itself. For servers with no `tools` parameter (nanoserve).
  The prompt is byte-identical to what Qwen's Jinja template produces, so the model sees the format
  it was trained on; only the parser differs.

A call the server cannot parse ends a native trajectory (an OpenAI client sees plain text, the
final answer); the attempt is recorded as malformed. The text agent owns its parser, so it tells
the model about a format violation and lets it retry, as a hand-rolled agent loop would.
"""

from __future__ import annotations

import json

from ..llm import ChatClient, LLMRequestError
from ..tools.base import ToolEnv
from .base import FORMAT_ERROR, SYSTEM_PROMPTS, AgentResult, cut_runaway, execute, parse_text_tool_calls


class _Base:
    def __init__(self, client: ChatClient, *, prompt: str = "v1", temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, max_tokens: int = 768, name: str | None = None):
        self.client = client
        self.prompt_version = prompt
        self.system = SYSTEM_PROMPTS[prompt]
        self.temperature = temperature
        self.sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
        self.max_tokens = max_tokens
        self.name = name or f"{type(self).__name__}:{client.model}"

    def _log(self, calls: list, step: int, r, runaway: bool = False) -> None:
        calls.append({"step": step, "latency_ms": round(r.latency_ms, 1), "prompt_tokens": r.prompt_tokens,
                      "completion_tokens": r.completion_tokens, "finish_reason": r.finish_reason, "runaway": runaway})


class NativeToolsAgent(_Base):
    async def run(self, prompt: str, env: ToolEnv, *, max_steps: int = 8, seed: int | None = None) -> AgentResult:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]
        llm_calls: list[dict] = []
        for step in range(max_steps):
            try:
                r = await self.client.chat(messages, tools=env.specs(), **self.sampling,
                                           max_tokens=self.max_tokens, seed=seed)
            except LLMRequestError as e:
                return AgentResult(None, "error", messages, llm_calls, str(e))
            self._log(llm_calls, step, r)
            msg = r.message
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            out = {"role": "assistant", "content": content}
            if tool_calls:
                out["tool_calls"] = [{"id": tc.get("id") or f"call_{step}_{i}", "type": "function",
                                      "function": {"name": tc["function"]["name"],
                                                   "arguments": tc["function"].get("arguments") or "{}"}}
                                     for i, tc in enumerate(tool_calls)]
            messages.append(out)
            if tool_calls:
                for tc in out["tool_calls"]:
                    rec = env.call(tc["function"]["name"], tc["function"]["arguments"], step=step)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": rec.result})
                continue
            # The server's parser found no call, so for an OpenAI-style client this *is* the final answer,
            # whatever it contains. If the text still looks like a tool call, the model tried to call a tool
            # and the server could not parse it: record that attempt as malformed, then stop, exactly as a
            # production client would (no retry the real stack would not give).
            calls, kind = parse_text_tool_calls(content)
            for c in calls:
                env.record_malformed(c.args if isinstance(c.args, str) else json.dumps(c.args), step, c.name,
                                     message="server returned this tool call as text: its parser rejected it")
            return AgentResult(content.strip(), "answered" if kind == "none" else "unparsed_call", messages, llm_calls)
        return AgentResult(None, "max_steps", messages, llm_calls)


TOOLS_HEADER = ("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
TOOLS_FOOTER = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
                "\"arguments\": <args-json-object>}\n</tool_call>")


def qwen_tools_system(system: str, specs: list[dict]) -> str:
    """Qwen2.5's chat-template rendering of a system prompt plus tools."""
    return system + TOOLS_HEADER + "".join("\n" + json.dumps(s, ensure_ascii=False) for s in specs) + TOOLS_FOOTER


class TextToolsAgent(_Base):
    async def run(self, prompt: str, env: ToolEnv, *, max_steps: int = 8, seed: int | None = None) -> AgentResult:
        messages = [{"role": "system", "content": qwen_tools_system(self.system, env.specs())},
                    {"role": "user", "content": prompt}]
        return await self._continue(messages, env, max_steps, seed)

    async def _continue(self, messages: list[dict], env: ToolEnv, max_steps: int, seed: int | None) -> AgentResult:
        llm_calls: list[dict] = []
        for step in range(max_steps):
            try:
                r = await self.client.chat(messages, **self.sampling, max_tokens=self.max_tokens,
                                           seed=seed)
            except LLMRequestError as e:
                return AgentResult(None, "error", messages, llm_calls, str(e))
            # Keep only the model's own turn; an invented user reply or tool result is cut (like a
            # ReAct "Observation:" stop sequence) and counted, never fed back as if it were real.
            content, runaway = cut_runaway(r.message.get("content") or "")
            self._log(llm_calls, step, r, runaway)
            messages.append({"role": "assistant", "content": content})
            calls, kind = parse_text_tool_calls(content)
            if kind == "none":
                return AgentResult(content.strip(), "answered", messages, llm_calls)
            if kind == "untagged":
                for c in calls:
                    env.record_malformed(c.args if isinstance(c.args, str) else json.dumps(c.args), step, c.name,
                                         message=FORMAT_ERROR)
                messages.append({"role": "user", "content": FORMAT_ERROR})
                continue
            results = [execute(env, c, step) for c in calls]
            messages.append({"role": "user", "content": "\n".join(
                f"<tool_response>\n{res}\n</tool_response>" for res in results)})
        return AgentResult(None, "max_steps", messages, llm_calls)


class GoldPathAgent(TextToolsAgent):
    """Diagnostic agent: the harness first makes the task's gold tool calls (through the env, so they
    are recorded and their results are real), as if the model had made them, then the model takes
    over. It answers "given perfect tool use, can the model write the answer?", separating
    tool-use failures from answer-writing failures. Its validity metrics are not the model's."""

    def __init__(self, *a, gold_calls: dict[str, list[dict]] | None = None, **kw):
        super().__init__(*a, **kw)
        self.gold_calls = gold_calls or {}

    async def run(self, prompt: str, env: ToolEnv, *, max_steps: int = 8, seed: int | None = None) -> AgentResult:
        messages = [{"role": "system", "content": qwen_tools_system(self.system, env.specs())},
                    {"role": "user", "content": prompt}]
        for i, c in enumerate(self.gold_calls.get(prompt, [])):
            body = json.dumps({"name": c["tool"], "arguments": c["args"]}, ensure_ascii=False)
            rec = env.call(c["tool"], c["args"], step=-1 - i)
            messages.append({"role": "assistant", "content": f"<tool_call>\n{body}\n</tool_call>"})
            messages.append({"role": "user", "content": f"<tool_response>\n{rec.result}\n</tool_response>"})
        return await self._continue(messages, env, max_steps, seed)
