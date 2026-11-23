"""Agent loops against a scripted fake LLM: parsing, invalid-call classification, stop reasons."""

import asyncio
import json
from pathlib import Path

from trajeval.agents.base import SYSTEM_PROMPTS, parse_text_tool_calls
from trajeval.agents.loop import NativeToolsAgent, TextToolsAgent, qwen_tools_system
from trajeval.llm import LLMRequestError, LLMResponse
from trajeval.scoring import score
from trajeval.tasks import load_suite
from trajeval.tools.world import make_env

TASKS = {t.id: t for t in load_suite()}


class FakeClient:
    """Returns scripted assistant messages in order and records what it was sent."""

    model = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    async def chat(self, messages, **kw):
        self.sent.append((json.loads(json.dumps(messages)), kw))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        msg = item if isinstance(item, dict) else {"role": "assistant", "content": item}
        return LLMResponse(msg, "stop", 100, 10, 5.0)


def native_call(name, args, i=0):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def tagged(name, args):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


def run(agent, prompt, env, max_steps=8):
    return asyncio.run(agent.run(prompt, env, max_steps=max_steps, seed=1))


def test_text_template_matches_qwen_jinja():
    """The text agent's prompt equals what Qwen2.5's own chat template renders (golden file made
    with transformers.apply_chat_template), so the model sees its trained tool format."""
    golden = (Path(__file__).parent / "qwen_template_golden.txt").read_text()
    sys_block = qwen_tools_system(SYSTEM_PROMPTS["v1"], make_env().specs())
    assert golden.startswith(f"<|im_start|>system\n{sys_block}<|im_end|>\n")


def test_text_agent_solves_task_and_scores():
    task = TASKS["lookup-customers-bengaluru"]
    client = FakeClient([tagged("sql_query", {"query": "SELECT COUNT(*) FROM customers WHERE city='Bengaluru'"}),
                         "There are 12 customers in Bengaluru."])
    env = make_env()
    res = run(TextToolsAgent(client), task.prompt, env)
    assert res.stop_reason == "answered" and res.final_answer.endswith("Bengaluru.")
    tool_msg = client.sent[1][0][-1]
    assert tool_msg["role"] == "user" and tool_msg["content"].startswith("<tool_response>\nCOUNT(*)\n12")
    s = score(task, res, env, 10.0)["metrics"]
    assert s["correct"] and s["tool_calls"] == 1 and s["invalid_calls"] == 0 and s["efficiency"] == 1.0


def test_invalid_calls_are_classified_not_crashing():
    client = FakeClient([
        tagged("lookup_customer", {"name": "x"}),                      # hallucinated tool
        tagged("read_file", {"file": "/docs/handbook.md"}),            # schema error (wrong arg name)
        '<tool_call>\n{"name": "sql_query", "arguments": {"query": "SELECT 1"\n</tool_call>',  # broken JSON
        '{"name": "sql_query", "arguments": {"query": "SELECT 1"}}',  # untagged: format violation
        tagged("read_file", {"path": "/nope.txt"}),                     # valid call, tool error
        "I could not find it.",
    ])
    env = make_env()
    res = run(TextToolsAgent(client), "q", env)
    assert [c.status for c in env.calls] == ["unknown_tool", "schema_error", "malformed_args", "malformed_args",
                                             "tool_error"]
    assert res.stop_reason == "answered"
    m = score(TASKS["trap-missing-order"], res, env, 1.0)["metrics"]
    assert m["invalid_calls"] == 4 and m["tool_error"] == 1 and m["unknown_tool"] == 1


def test_redundant_and_loop_detection():
    q = {"query": "SELECT status FROM orders WHERE id = 1044"}
    client = FakeClient([tagged("sql_query", q)] * 3 + ["shipped"])
    env = make_env()
    res = run(TextToolsAgent(client), "q", env)
    m = score(TASKS["lookup-order-status"], res, env, 1.0)["metrics"]
    assert m["redundant_calls"] == 2 and m["looped"] and m["correct"] and m["efficiency"] == 0.5


def test_repeat_after_side_effect_is_not_redundant():
    read = {"path": "/reports/x.txt"}
    client = FakeClient([tagged("read_file", read), tagged("write_file", {"path": "/reports/x.txt", "content": "1"}),
                         tagged("read_file", read), "done"])
    env = make_env()
    run(TextToolsAgent(client), "q", env)
    m = score(TASKS["action-write-helmet-total"], run_result(env), env, 1.0)["metrics"]
    assert m["redundant_calls"] == 0


def run_result(env):
    from trajeval.agents.base import AgentResult
    return AgentResult("done", "answered", [], [])


def test_max_steps_and_server_error():
    env = make_env()
    res = run(TextToolsAgent(FakeClient([tagged("list_dir", {})] * 3)), "q", env, max_steps=3)
    assert res.stop_reason == "max_steps" and res.final_answer is None
    m = score(TASKS["open-billing-bug"], res, env, 1.0)["metrics"]
    assert m["correct"] is False  # an open-ended task with no answer fails without a judge
    res = run(TextToolsAgent(FakeClient([LLMRequestError("400: context length")])), "q", make_env())
    assert res.stop_reason == "error" and "context" in res.error


def test_native_agent_solves_task():
    task = TASKS["action-email-order-status"]
    client = FakeClient([
        native_call("sql_query", {"query": "SELECT status FROM orders WHERE id=1044"}),
        native_call("send_email", {"to": "nikhil.joshi@mailbox.example", "subject": "Order 1044",
                                   "body": "Your order has shipped."}, 1),
        "Done, I emailed Nikhil.",
    ])
    env = make_env()
    res = run(NativeToolsAgent(client), task.prompt, env)
    assert client.sent[0][1]["tools"][0]["function"]["name"] == "list_dir"
    assert client.sent[1][0][-1] == {"role": "tool", "tool_call_id": "c0", "content": "status\nshipped"}
    m = score(task, res, env, 1.0)["metrics"]
    assert m["correct"] and m["invalid_calls"] == 0 and m["unrequested_side_effects"] == 0


def test_native_unparsed_call_ends_trajectory_like_production():
    """vLLM's hermes parser rejects a call with no "arguments" key and returns it as text. An OpenAI
    client sees a final answer; the harness records the attempted call as malformed and stops."""
    client = FakeClient([{"role": "assistant", "content": '<tool_call>\n{"name": "get_db_schema"}'}, "unreachable"])
    env = make_env()
    res = run(NativeToolsAgent(client), "q", env)
    assert res.stop_reason == "unparsed_call" and res.final_answer.startswith("<tool_call>")
    assert [c.status for c in env.calls] == ["malformed_args"] and len(client.script) == 1
    # the text agent, which owns its parser, accepts the same call (missing arguments -> {})
    env = make_env()
    res = run(TextToolsAgent(FakeClient(['<tool_call>\n{"name": "get_db_schema"}', "done"])), "q", env)
    assert [c.status for c in env.calls] == ["ok"] and res.final_answer == "done"


def test_unrequested_side_effect_counted():
    task = TASKS["trap-do-not-email"]
    client = FakeClient([tagged("send_email", {"to": "mira@kestrelcycles.example", "subject": "bug", "body": "hi"}),
                         "Talk to Mira Fernandes."])
    env = make_env()
    res = run(TextToolsAgent(client), task.prompt, env)
    out = score(task, res, env, 1.0)
    assert out["metrics"]["unrequested_side_effects"] == 1 and out["metrics"]["correct"] is False


def test_parse_variants():
    calls, kind = parse_text_tool_calls('thinking...\n<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>'
                                        '\n<tool_call>\n{"name": "b", "arguments": "{\\"y\\": 2}"}\n</tool_call>')
    assert kind == "tagged" and [(c.name, c.args) for c in calls] == [("a", {"x": 1}), ("b", {"y": 2})]
    calls, kind = parse_text_tool_calls('<tool_call>\n{"name": "a", "arguments": {}}')  # unclosed tag
    assert kind == "tagged" and calls[0].args == {}
    assert parse_text_tool_calls("The answer is 12.")[1] == "none"
    assert parse_text_tool_calls('```json\n{"name": "a", "arguments": {}}\n```')[1] == "untagged"


def test_invented_recipient_is_counted_even_though_the_call_is_valid():
    task = TASKS["action-email-escalation"]
    client = FakeClient([tagged("send_email", {"to": "head.of.support@example.com", "subject": "Ticket 18 escalation",
                                               "body": "ticket 18"}), "sent"])
    env = make_env()
    m = score(task, run(TextToolsAgent(client), task.prompt, env), env, 1.0)["metrics"]
    assert m["invalid_calls"] == 0 and (m["ungrounded_recipients"], m["recipients"]) == (1, 1) and not m["correct"]
    client = FakeClient([tagged("sql_query", {"query": "SELECT email FROM employees WHERE title='Head of Support'"}),
                         tagged("send_email", {"to": "leela@kestrelcycles.example", "subject": "Ticket 18 escalation",
                                               "body": "ticket 18"}), "sent"])
    env = make_env()
    m = score(task, run(TextToolsAgent(client), task.prompt, env), env, 1.0)["metrics"]
    assert (m["ungrounded_recipients"], m["recipients"]) == (0, 1) and m["correct"]


def test_runaway_turn_is_cut_and_counted():
    """The model calls a tool, then (skipping <|im_end|>) writes a fake user turn with an invented
    tool result. Only its own turn is kept; the invented result never reaches the history."""
    content = ('<tool_call>\n{"name": "sql_query", "arguments": {"query": "SELECT COUNT(*) FROM tickets"}}\n</tool_call>'
               "\nuser\n<tool_response>\n15\n</tool_response>")
    client = FakeClient([content, "There are 20 tickets."])
    env = make_env()
    res = run(TextToolsAgent(client), "q", env)
    assert res.messages[2]["content"].endswith("</tool_call>")
    assert "<tool_response>\nCOUNT(*)\n20" in client.sent[1][0][-1]["content"]   # the real result, not 15
    assert [c["runaway"] for c in res.llm_calls] == [True, False]
    m = score(TASKS["lookup-cancelled-orders"], res, env, 1.0)["metrics"]
    assert m["runaway_turns"] == 1 and m["tool_calls"] == 1


def test_cut_runaway_leaves_normal_text_alone():
    from trajeval.agents.base import cut_runaway
    assert cut_runaway("The user asked for 12 customers.\nThat is all.") == ("The user asked for 12 customers.\nThat is all.", False)
    assert cut_runaway("Done.\nassistant\nMore") == ("Done.", True)


def test_gold_path_agent_prefills_real_results_and_excludes_them_from_validity():
    from trajeval.agents.loop import GoldPathAgent
    task = TASKS["open-billing-bug"]
    client = FakeClient(["The tax is computed on the subtotal, not the discounted price."])
    env = make_env()
    res = run(GoldPathAgent(client, gold_calls={task.prompt: task.gold_calls}), task.prompt, env)
    sent = client.sent[0][0]
    assert sent[2]["content"].startswith("<tool_call>") and "def order_total" in sent[3]["content"]
    assert [c.step for c in env.calls] == [-1]
    m = score(task, res, env, 1.0)["metrics"]
    assert m["tool_calls"] == 0 and m["steps"] == 1
