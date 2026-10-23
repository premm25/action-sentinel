# Running Action Sentinel with VegaML

[VegaML](https://github.com/NandhaKishorM/vegaml) is the risk model behind Action Sentinel.
- **VegaML** reads each proposed agent action and judges it: safe, needs review, or unsafe.
- **Sentinel's rules and governance** decide what may actually run. A model can only escalate: it can
  send an action to review, but never approve or block it on its own say-so.

## What VegaML contributes, measured

VegaML offers two readouts over the same frozen Qwen3.5 backbone: its trained physics **engine**, and a
**few-shot head** fitted on a handful of labelled examples. On agent safety the head is the one that
works, so it is what Sentinel uses:

| result | where |
|---|---|
| VegaML's engine on Sentinel's synthetic action question: near chance; it abstained on every action | `docs/vegalab/RESULTS.md` (RQ7) |
| Head fitted on 28 synthetic actions: 0.75 test accuracy; behind Sentinel, 8 of 10 safe actions auto-approved and 0 of 34 false approvals | `docs/vegalab/AGENT_FRAMING.md`, RQ7b |
| Head on AgentHarm (real requests, 64 fitting examples): 0.841 accuracy, 0.920 AUROC | `docs/vegalab/AGENTHARM.md` |
| The head's gain comes from VegaML's prompt-conditioned features: plain logistic regression on them matches it, a plain backbone probe does worse | `docs/vegalab/AGENTHARM_BASELINES.md` |
| Behind full Sentinel on AgentHarm: 0 of 176 harmful requests approved (24 with the head alone) | `docs/vegalab/AGENTHARM_SENTINEL.md` |

These are single CPU runs with the 0.8B checkpoint. The limits are in each document.

## Install

VegaML needs torch, about 1.8 GB of weights and about 3.5 GB of RAM for the 0.8B checkpoint on CPU.

```bash
pip install "action-sentinel[api,vegaml] @ git+https://github.com/premm25/action-sentinel.git"
# or, from a clone:  pip install -e ".[api,vegaml]"
action-sentinel vega check --load        # downloads the weights on first use and loads the model once
```

The results in this repository were measured with vegaml 0.9.0, transformers 5.19.0 and torch 2.14 (CPU).

## Verified end to end (2026-10-24, CPU)

`action-sentinel vega fit` on the bundled synthetic dataset gave cross-validated accuracy 0.800 on 28
calibration actions, and 33/44 (0.750) on the test split. That matches `AGENT_FRAMING.md`. A fresh process
then loaded the saved head through this profile in 36 s and evaluated six safe test actions:
- four were auto-approved (head confidence 0.86–0.97)
- two went to review (head said safe, at confidence 0.83, below the 0.85 auto-approval threshold)

## Run Sentinel with VegaML

`config/sentinel-vegaml.yaml` is the VegaML profile:
- the `rules` and `vegaml-head` providers judge every action side by side
- `warm_up: true` loads the model and the head when the API starts

```bash
action-sentinel --config config/sentinel-vegaml.yaml serve
```

```python
from action_sentinel.config import load_settings
from action_sentinel.service import Sentinel

sentinel = Sentinel(load_settings("config/sentinel-vegaml.yaml"))
sentinel.warm_up()                       # the API does this itself when warm_up is true
result = sentinel.evaluate(action)       # VegaML's verdict is in result.decision.provider_results
```

Out of the box the head is fitted at startup on the calibration split of the bundled synthetic dataset.
That is enough to try it, but **fit it on your own actions before relying on it**.

## Fit the head on your own actions

Label some of your agent's actions in the Confidence Lab dataset format
(`examples/evaluation_datasets/sentinel_synthetic_v1.jsonl` shows it):
- the action itself
- the question: by default `disposition`, with safe, needs_review or unsafe
- the expected label
- a split: `calibration` to fit on, `test` to measure on

Then:

```bash
action-sentinel vega fit --dataset my_actions.jsonl --out heads/my_actions.npz --evaluate test
```

- It fits on the `calibration` split (or `dev`) and **refuses `test`**.
- It prints the cross-validated accuracy. VegaML discards a head that does no better than chance.
- With `--evaluate test` it also reports accuracy on the test split, as a measurement only.
- The head file records the question, framing, checkpoint and engine it was fitted for. A provider
  refuses to serve it for anything else (`TTT_HEAD_INVALID`, `TTT_HEAD_QUESTION_MISMATCH`), because its
  features wouldn't match.

To serve the saved head, set `ttt_head_path` in the `vegaml-head` provider's config. It replaces
`ttt_fit_dataset`, and startup no longer has to fit:

```yaml
  - name: vegaml-head
    type: vegaml
    config: {checkpoint: "0.8b", device: cpu, adapter_mode: "off", framing: compact, readout: ttt,
             ttt_head_path: heads/my_actions.npz}
```

## Try it on real traffic first

Put the agent's tools behind `@guard.tool` in shadow mode (`docs/GUARD.md`), with the VegaML profile.
Every call runs as before, and Sentinel records what VegaML and the rules would have done. Label those
records, fit the head on them, measure it, and only then switch the guard to `enforce`.

## Operating notes

- **Warm-up:** a cold provider loads the model (and fits a head from a dataset) on its first decision.
  That can take minutes on CPU, longer than the 200-second provider timeout, so those first decisions
  end in ERROR. Sentinel fails closed and nothing is approved, but warm up before taking traffic.
  The API does this when `warm_up` is true; in-process code calls `sentinel.warm_up()`. A saved head
  avoids the fit entirely.
- **One model in memory:** every VegaML provider with the same checkpoint and device shares one loaded
  model.
- **Latency (0.8B, laptop CPU, after warm-up):** median 2.6 s for the head alone on AgentHarm. In a full
  `evaluate` alongside the rules, 6–10 s per action was seen on a busy laptop. A GPU is much faster.
- **Startup:** loading a saved head took 36 s in total (model load included). Fitting the bundled head at
  startup instead takes a few minutes on CPU.
- **No GPU or no VegaML installed:** the VegaML providers report themselves unavailable and decisions
  that need them fail closed. The default `config/sentinel.yaml` uses the rules only and works offline.
