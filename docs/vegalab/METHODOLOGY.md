# Methodology

## Research questions (investigated, not presumed)

| RQ | how the lab addresses it |
|---|---|
| RQ1 accuracy among predefined choices | accuracy, balanced accuracy, macro F1, confusion matrices per task (test split) |
| RQ2 do probabilities match correctness | ECE (10 equal-width bins), Brier, log loss, reliability bins; plus a one-parameter temperature fitted on the calibration split and applied to test |
| RQ3 does abstention reduce error, and at what coverage | VegaML's native `abstain` flag at the checkpoint's default risk (0.1): coverage, selective accuracy, accuracy on abstained items; risk–coverage curve and AURC from confidence; conformal set coverage and size |
| RQ4 VegaML vs its frozen backbone | `vega-off` vs `backbone-optscore` on identical items, paired bootstrap 95% CI |
| RQ5 ambiguity, option count, imbalance, shift | `dbpedia14_k2/k4/k8/14` on the same items; slices by difficulty (adversarial ANLI), input length and label-space size; balanced accuracy for imbalanced tasks |
| RQ6 latency, memory, throughput by size | per-example wall clock, peak working set and throughput in run metadata; 0.8B on CPU locally, 0.8B and 4B on GPU through the Kaggle notebook |
| RQ7 governance independent of model quality | Sentinel's Confidence Lab on the synthetic agent-action dataset with VegaML as the provider, naive vs governed |

## Systems and conditions

| system | what it is | input |
|---|---|---|
| `vega-off` | VegaML engine readout, adapters off (pure base engine) | VegaML's own prompt |
| `vega-auto` | VegaML's default: learned gates decide per question whether `public_intents` or `typed_decisions` apply | same |
| `backbone-optscore` | the same frozen backbone as a plain causal LM: prompt token ids from `vegaml._common.build_prompt` with the checkpoint's settings (identical to what VegaML reads); each option scored by mean token log-prob of `" <option text>"` after `Outcome:`; softmax, temperature 1 | identical token ids |
| `majority` | most frequent label in the calibration and dev examples (training splits); Laplace-smoothed prior as probabilities | none |

Shared conditions: same examples, options and question wording; `views=1`; `alpha=0.1`; default risk;
greedy and deterministic (no sampling anywhere). Backbone precision follows VegaML's policy (float32
on CPU; fp16 or bf16 on CUDA). Batched option scoring for the backbone uses a repeated KV/linear-
attention cache, verified equal to one-at-a-time scoring within 1e-5.

**What the comparison can and can't show.** `vega-off` vs `backbone-optscore` differ in the *decision
mechanism* (trained engine vs zero-shot LM option likelihood) on the same frozen features and prompt.
A difference is evidence about the trained engine relative to **this** baseline method. It doesn't
show that the physics formulation itself is responsible: a different trained head on the same
features might do as well. The comparison that isolates that is not run here.

## Metric definitions

See the docstring of `action_sentinel/vegalab/analysis.py`. Errors count as incorrect and are reported
separately. Probabilistic metrics are computed only when every evaluated example has a complete
distribution. Confidence intervals: Wilson for proportions, paired bootstrap (2,000 resamples, seed 0)
for accuracy differences.

## Reproducibility

Every run folder contains `experiment.json` (config), `<system>.jsonl` (raw predictions with native
output), `<system>.run.json` (versions, hardware, timing, peak memory, dataset hashes, prediction-file
hash), `metrics.json`, `summary.csv` and `REPORT.md`. Re-running `action-sentinel vega analyze <dir>`
recomputes every number from the raw predictions.
