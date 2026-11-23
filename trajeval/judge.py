"""LLM-as-judge for open-ended tasks: pass/fail against a per-task rubric.

Judge configs differ in what the judge sees and how many samples it takes:
  rubric       task + rubric + answer
  rubric+ref   ... + a reference answer
  majority     rubric+ref, 3 samples at temperature 0.7, majority vote
Each config's agreement with labelled samples is measured in agreement.py.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass

from .llm import ChatClient, LLMRequestError

JUDGE_SYSTEM = ("You are a strict, careful grader. You check an assistant's answer against a rubric and decide "
                "whether it passes. You do not reward length or politeness, only correctness and completeness.")

JUDGE_TEMPLATE = """Grade the assistant's answer to a task given at a company called Kestrel Cycles.

## Task given to the assistant
{prompt}

## Rubric (the answer PASSES only if it satisfies every point)
{rubric}
{reference_block}
## Assistant's answer
<answer>
{answer}
</answer>

Check each rubric point in one short line, then finish with exactly one line: "VERDICT: PASS" or "VERDICT: FAIL"."""

REFERENCE_BLOCK = """
## Reference answer (correct; the assistant does not need to match its wording, but facts that contradict it are errors)
{reference}
"""

# v2: written after reading the dev-split disagreements of the v1 judges (they passed raw tool
# calls, "I have sent the email" non-answers and answers with invented facts, usually with no
# reasoning at all). It makes the judge quote evidence for every point before deciding, and lists
# the automatic failures explicitly. Test-split results were not looked at while writing it.
JUDGE_TEMPLATE_V2 = """Grade the assistant's answer to a task given at a company called Kestrel Cycles.

## Task given to the assistant
{prompt}

## Rubric (the answer PASSES only if it satisfies every point)
{rubric}
{reference_block}
## Assistant's answer
<answer>
{answer}
</answer>

The answer FAILS automatically if any of these is true:
- it is a tool call, code, or a plan for what to do next instead of the requested answer;
- it claims it did something (sent an email, wrote a file) instead of giving the answer;
- it says the information could not be found, or asks the user for details;
- it states any fact, number, name or date that contradicts the reference answer.

Work in this order:
1. For each rubric point, quote the exact words from the answer that satisfy it, or write "none".
2. List any statement in the answer that contradicts the reference, or write "none".
3. Finish with exactly one line: "VERDICT: PASS" or "VERDICT: FAIL"."""

_VERDICT = re.compile(r"VERDICT\s*[:\-]?\s*\**\s*(PASS|FAIL)", re.I)


@dataclass
class JudgeConfig:
    name: str
    use_reference: bool = True
    samples: int = 1
    temperature: float = 0.0
    max_tokens: int = 512
    template: str = "v1"


JUDGE_CONFIGS = {
    "rubric": JudgeConfig("rubric", use_reference=False),
    "rubric+ref": JudgeConfig("rubric+ref", use_reference=True),
    "majority3": JudgeConfig("majority3", use_reference=True, samples=3, temperature=0.7),
    "rubric+ref-v2": JudgeConfig("rubric+ref-v2", use_reference=True, template="v2", max_tokens=768),
    "majority3-v2": JudgeConfig("majority3-v2", use_reference=True, samples=3, temperature=0.7, template="v2",
                                max_tokens=768),
}


def build_prompt(prompt: str, rubric: list[str], reference: str | None, answer: str, use_reference: bool,
                 template: str = "v1") -> str:
    rub = "\n".join(f"{i}. {r}" for i, r in enumerate(rubric, 1))
    ref = REFERENCE_BLOCK.format(reference=reference) if use_reference and reference else ""
    t = {"v1": JUDGE_TEMPLATE, "v2": JUDGE_TEMPLATE_V2}[template]
    return t.format(prompt=prompt, rubric=rub, reference_block=ref, answer=answer.strip())


def parse_verdict(text: str) -> str:
    m = _VERDICT.findall(text or "")
    return m[-1].lower() if m else "unparsed"


async def judge_one(client: ChatClient, cfg: JudgeConfig, prompt: str, rubric: list[str], reference: str | None,
                    answer: str | None, seed: int = 0) -> dict:
    if not answer or not answer.strip():
        return {"verdict": "fail", "votes": [], "rationale": "no final answer", "judge_tokens": 0, "latency_ms": 0}
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_prompt(prompt, rubric, reference, answer, cfg.use_reference, cfg.template)}]
    outs = []
    for k in range(cfg.samples):
        try:
            r = await client.chat(msgs, temperature=cfg.temperature, max_tokens=cfg.max_tokens, seed=seed + k)
            outs.append(r)
        except LLMRequestError as e:
            outs.append(e)
    votes = [parse_verdict(o.message.get("content", "")) if not isinstance(o, Exception) else "unparsed" for o in outs]
    counted = Counter(v for v in votes if v != "unparsed")
    if not counted:
        verdict = "unparsed"
    elif cfg.samples == 1:
        verdict = votes[0]
    else:
        verdict = "pass" if counted["pass"] > counted["fail"] else "fail"  # ties fail: be strict
    first = next((o for o in outs if not isinstance(o, Exception)), None)
    return {
        "verdict": verdict, "votes": votes,
        "rationale": first.message.get("content", "") if first else str(outs[0]),
        "judge_tokens": sum(o.prompt_tokens + o.completion_tokens for o in outs if not isinstance(o, Exception)),
        "latency_ms": sum(o.latency_ms for o in outs if not isinstance(o, Exception)),
    }


async def judge_many(client: ChatClient, cfg: JudgeConfig, items: list[dict], concurrency: int = 16,
                     progress=print) -> list[dict]:
    """items: dicts with prompt, rubric, reference, answer (+ any ids, passed through)."""
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(it):
        nonlocal done
        async with sem:
            j = await judge_one(client, cfg, it["prompt"], it["rubric"], it.get("reference"), it.get("answer"),
                                seed=it.get("seed", 0))
        done += 1
        if done % 25 == 0 or done == len(items):
            progress(f"[judge {cfg.name}] {done}/{len(items)}")
        return {**{k: v for k, v in it.items() if k in ("sample_id", "task_id", "trial", "run_id")},
                "judge": cfg.name, "judge_model": client.model, **j}

    return await asyncio.gather(*(one(it) for it in items))
