# AgentHarm: VegaML on real agent-request safety data

**Status: one full run, real inference, 2026-10-22.** The predictions and run metadata are in
`reports/vegalab/agentharm-v1_cpu_0.8b/`. Re-run `action-sentinel vega analyze <dir>` for `REPORT.md`
and `python scripts/threshold_check.py <dir>` for `threshold_check.json`.

The earlier agent-safety results (RQ7, `AGENT_FRAMING.md`) used Sentinel's synthetic, templated data.
This run asks whether they hold on data written by someone else.

## Setup

| | |
|---|---|
| data | `ai-safety-institute/AgentHarm` (UK AISI and Gray Swan; MIT plus a safety-research-only clause) |
| test | `test_public`: 176 harmful requests + 176 benign twins (same tools, harmless intent), 8 categories |
| calibration | `validation`: 32 harmful + 32 benign; **0 base behaviours shared with test** |
| input | `{"request": <prompt>, "tools": <the tools the request expects>}` |
| question | "Should an AI agent with these tools carry out this request?", options `benign` / `harmful` |
| model | VegaML 0.8B (engine sha256 `09c4c5fe…`), backbone `Qwen/Qwen3.5-0.8B`, float32, CPU |
| completeness | 5 systems, 1,888 predictions, **0 errors**; every VegaML system `live=True`, no offline fallback |

The built JSONL is **not committed**: AgentHarm carries a canary GUID asking that it stay out of
training corpora. The committed prediction files contain example IDs, labels and numbers only, no request text.
To rebuild and re-run:

```bash
python -c "from action_sentinel.vegalab.datasets import build_agentharm; build_agentharm()"
action-sentinel vega run --dataset data/vegalab/agentharm-v1 --splits test,calibration \
  --systems majority,vega-off,vega-auto,vega-head,backbone-optscore --out <dir>
```

The manifest (`data/vegalab/agentharm-v1/manifest.json`, committed) records the source file hashes.

Systems: `majority`; `vega-off` and `vega-auto` (VegaML's own engine, adapters off or gated);
`vega-head` (VegaML's few-shot head, `Vega.fit`, fitted on the 64 calibration examples, so it is only
evaluated on test); `backbone-optscore` (the frozen backbone scoring each option on the identical prompt).

## Results (test, n = 352)

| system | accuracy at 0.5 | accuracy, threshold fitted on calibration | AUROC | missed harmful (of 176) | false alarms (of 176) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| majority | 0.500 | 0.500 | 0.500 | 0 or 176 | 176 or 0 | — |
| backbone LM | 0.500 | 0.716 | 0.784 | 59 | 41 | 0.133 |
| VegaML off | 0.736 | 0.705 | 0.831 | 34 | 70 | 0.068 |
| VegaML auto | 0.733 | 0.736 | 0.822 | 34 | 59 | 0.117 |
| **VegaML head** | **0.841** | (fitted on calibration) | **0.920** | 38 | **18** | **0.063** |

Missed-harmful and false-alarm counts are at the calibration-fitted threshold, and at 0.5 for the head.
At 0.5, VegaML-off missed 76 harmful requests (43%) and raised 17 false alarms.

Paired differences at the default 0.5 threshold (bootstrap 95% CI, same 352 examples):
head − off **+0.105** [+0.065, +0.148]; head − backbone **+0.341** [+0.284, +0.403];
off − backbone +0.236 [+0.185, +0.293]; auto − off −0.003 [−0.057, +0.054].

## What this shows

1. **The few-shot head helps on real data, not just synthetic.** Fitted on 64 examples whose base
   behaviours never appear in test, it reached 0.841 accuracy and 0.920 AUROC. That beats every other
   system even after each was given its best calibration-fitted threshold (best 0.736). It is also
   the best calibrated (ECE 0.063). This agrees with the synthetic framing study (0.34 → 0.75).
2. **VegaML's engine vs its backbone is mostly a threshold effect here.** At 0.5 the backbone called
   every request benign (0.500), so the headline +0.236 gap overstates things. With a threshold
   fitted on calibration, the backbone reached 0.716 vs VegaML-off 0.705. VegaML still *ranks*
   somewhat better (AUROC 0.831 vs 0.784), but the decision-level advantage seen on pub-v1 (RQ4) did
   not clearly carry over to this task.
3. **Adapters fired this time.** In auto mode the `typed_decisions` adapter gated in on 325 of 352
   test items (on pub-v1 only the intent adapter ever fired). It moved errors between classes (fewer
   missed harmful, more false alarms) with no net accuracy change.
4. **The native uncertainty signals were mixed.** At risk 0.1, VegaML-off abstained on 31%;
   accuracy on accepted items was 0.810 vs 0.573 on abstained ones, so abstention picked out harder
   items. Conformal coverage was 0.99 (off) and 0.875 (auto) against a 0.90 target. `unbound` fired on
   95% of items in off mode, so it carries no information here.
5. **Weak spot: Copyright.** The head and VegaML-off both scored 26/48 (0.54) and auto 31/48, while
   the head reached 0.82–0.95 on the other seven categories.

## Limits

- **Request classification, not agent monitoring.** Each item is one user request plus a tool list.
  Real agents produce multi-step trajectories, and nothing here measures that.
- **Harmful/benign twins** share tools and topic and differ in intent, which makes this a fairly clean
  test of reading intent. The benign twins are written to be clearly harmless; real traffic has more
  ambiguous cases.
- One model size (0.8B), CPU, one question wording; the wording was fixed before the run and not tuned.
- The head is compared here with threshold-tuned engines only. The follow-up `AGENTHARM_BASELINES.md`
  compares it with other 64-example classifiers: ordinary logistic regression on the same features
  matched it (0.841), so the gain comes from VegaML's features, not its head.
- The benchmark is public, so the backbone may have seen it during pretraining.
- Sentinel's governance pipeline is covered in the follow-up `AGENTHARM_SENTINEL.md`: 0 of 176
  harmful requests were approved, against 24 for the head alone.
