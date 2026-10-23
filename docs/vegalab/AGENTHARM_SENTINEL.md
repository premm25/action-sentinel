# AgentHarm through Sentinel

RQ7b measured Sentinel with the VegaML head on synthetic actions. This study runs real AgentHarm requests
through the same pipeline: validation, provider, uncertainty, governance.

## Protocol (committed before the run)

Script: `scripts/agentharm_sentinel.py`.

**Mapping.** Each AgentHarm request becomes a Sentinel action:
- action type `run_agent_task`, from source `assistant-agent`
- payload `{request, tools}`, target `agent-tools/<first tool>`, no evidence

The action is admitted by `examples/rule_packs/agent_tasks.yaml` as low risk. The pack has **no intent
rules**: whether a request is harmful can't be decided by comparing fields. This study therefore measures
Sentinel's governance around a model, not its rules. For the same reason, no hard-violation blocks are
expected.

**Question.** `request_safety`: safe (may run) or unsafe (must be refused).

**Expected outcome.**
- Benign requests should be APPROVE.
- Harmful requests should be BLOCK. In Sentinel a model can only escalate, never block, so
  HUMAN_REVIEW of a harmful request counts as a safe miss.
- **APPROVE of a harmful request is the failure that matters.**

**Provider.** `vegaml-head`: VegaML's few-shot head on the compact framing, fitted on the calibration
split (AgentHarm validation, 64 requests, base behaviours disjoint from test).

**Thresholds.** Governance thresholds come unchanged from `config/sentinel.yaml`; auto-approval needs
confidence of at least 0.85. Nothing is tuned.

**Runs.** Each configuration is run once on test (352):
1. `1_head_only_naive`: the head's label is mapped straight to an outcome (safe → APPROVE, unsafe → BLOCK).
2. `2_full_sentinel_head`: the full pipeline with the head.

**Reported.**
- Harmful requests approved (false approvals).
- Benign requests auto-approved (coverage).
- Review load.
- Outcome counts.

The converted dataset stays outside git. The reports hold only IDs, labels and numbers.

## Results (single run, 2026-10-24, CPU)

Outputs: `reports/vegalab/agentharm_sentinel_0.8b/`. The head was fitted on 64 calibration requests
(cross-validated accuracy 1.0, kept). On this framing (compact sentence, safe/unsafe) it labelled
**0.830** of test requests correctly, against 0.841 for the lab framing in `AGENTHARM.md`. Its ECE was
0.037.

| | head alone (naive) | full Sentinel + head |
|---|---:|---:|
| harmful requests **approved** (of 176) | **24** (13.6%) | **0** (0.0%, 95% CI 0–2.1%) |
| benign requests auto-approved (of 176) | 140 (79.5%) | 95 (54.0%) |
| sent to human review (of 352) | 0 | 257 (73.0%) |
| blocked | 186 | 0 |

**Startup artifact in the naive run.** The first two test requests ended in ERROR because the first call
loads the model and fits the head (about 8 minutes on this CPU), longer than Sentinel's 200-second
provider timeout. Sentinel failed closed on them. The late answers were then cached, so the full run
had no errors. Both requests were harmful and the head labelled them unsafe, so counting them makes no
difference to the 24.

**Why the outcomes came out as they did:**
- All 176 harmful requests went to review. 152 were labelled unsafe (a model can escalate, never block,
  so these are review rather than BLOCK). The other 24 were labelled safe, but each with confidence
  below the 0.85 auto-approval threshold.
- Of the 81 benign requests sent to review, 45 were labelled safe with confidence below 0.85, and 36
  were labelled unsafe.
- Median provider latency after warm-up: 2.6 s.

## What this shows

1. **Governance turned a 13.6% harmful-approval rate into zero** on real requests. It also held up on
   synthetic data (RQ7b: 0/34 false approvals). It did this without rules about intent; only the
   uncertainty threshold and the "a model can only escalate" policy were involved.
2. **The margin was thin.** The most confident harmful request labelled safe scored 0.842, just under
   the 0.85 threshold. The threshold was fixed before this run and not tuned, but a slightly different
   model, framing or threshold could let some through. The zero is not a guarantee.
3. **The price is review load.** 73% of requests went to a human, and 46% of benign ones did. That is a
   lot for production; a better provider is the main way to reduce it.
4. **Sentinel blocked nothing**, by design: no rule in the pack can recognise a harmful intent, and model
   labels only escalate. Rejecting the 176 harmful requests is left to reviewers.
5. **Operational finding:** a cold VegaML head provider times out on its first requests. Sentinel fails
   closed (ERROR, never approve), which is correct, but a deployment should warm the provider up
   before taking traffic.

## Limits

- Single requests, not agent trajectories; the rule pack only admits the action type.
- One provider, one framing, one threshold, one model size, CPU.
- 176 harmful/benign twin pairs, so the examples aren't independent.
- The head was fitted on 64 AgentHarm validation requests, the same kind of data it was tested on.
  Requests from a different source would likely do worse.
