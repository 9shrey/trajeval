# Labelled samples for judge calibration

`samples.jsonl` holds 200 open-ended answers drawn from the runs by `trajeval sample-labels`
(round-robin over the 20 open-ended tasks, alternating runs so every model is represented). The
split is by task: dev tasks were used to iterate on the judge prompt, test tasks were not.

## Who labelled them

| file | labeller | status |
|---|---|---|
| `labels_claude.jsonl` | Claude (Anthropic's LLM), first pass, one-line rationale per label | **not human labels** |
| `labels_human.jsonl` | a person, via `trajeval label` | takes precedence where present |

Every report says which source a number comes from (`labeled_by`). A kappa computed against
`labels_claude.jsonl` measures agreement between two LLMs, not between a judge and a human, and is
labelled that way in the README and the dashboard.

To review: `trajeval label` walks through every sample, shows the task, rubric, reference, answer and
Claude's label, and records your decision (agree / flip / pass / fail). It saves after every
answer, so it can be stopped and resumed. `trajeval agree --source human` then reports agreement
on the reviewed subset, and `--source best` uses human labels wherever they exist.

## The labelling rule

A sample is **pass** only if the answer satisfies every rubric point for its task:

1. Required facts must be present and correct. "At least one/two of ..." is satisfied by the minimum.
   A rubric point with several parts is met when its core claim is there (the firmware update made
   the range estimate read 0 km); a missing supporting detail ("at about 20%") does not fail it,
   but any part that is stated wrongly does.
2. Any statement that contradicts the world (wrong number, wrong person, wrong status, wrong
   eligibility) fails the sample, even if every required fact is also there.
3. Invented specifics that the tools could not have returned (tracking numbers, delivery dates,
   fees, names) fail the sample. Generic courtesy ("sorry for the wait", "let me know") does not.
4. Length and format requests in the prompt ("two sentences") are not graded unless the rubric says so.
5. An answer that says it could not find information that the tools do provide is a fail. So is an
   answer that is a raw tool call, or that asks the user to do the agent's job.
6. Rounding: amounts within the tolerance the rubric states ("within Rs 1", "about") pass.
