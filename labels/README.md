# Labelled samples for judge calibration

`samples.jsonl` holds 200 open-ended answers drawn from the runs by `trajeval sample-labels`
(round-robin over the 20 open-ended tasks, alternating runs so every model is represented). The
split is by task: dev tasks were used to iterate on the judge prompt, test tasks were not.

## Who labelled them

| file | labeller | status |
|---|---|---|
| `labels_claude.jsonl` | Claude (Anthropic's LLM), first draft, one-line rationale per label | 200 drafted |
| `labels_human.jsonl` | the author, reviewing the drafts | 200 reviewed, 0 changed, **not blind** |

The author's review (2026-11-23) saw Claude's label next to each answer and accepted all 200. That
confirms the drafts against the rule below; it is not an independent second annotation, so there is
no human-human agreement figure for this set. Each row of `labels_human.jsonl` records `blind`
and the draft label it was compared with.

For an independent annotation run `trajeval label`. It is blind by default: it shows the task,
rubric, reference and answer, records your pass/fail, and only then reveals the draft label
(`--show-claude` shows it up front instead). It saves after every answer, so it can be stopped and
resumed. `trajeval agree` then reports Claude-vs-human kappa on the blind rows, and every judge's
agreement against the labels (`--source human`, `claude` or `best`).

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
