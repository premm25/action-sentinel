# AgentHarm: does VegaML's head beat ordinary classifiers?

The AgentHarm run (`AGENTHARM.md`) found VegaML's few-shot head at 0.841 test accuracy, well ahead of
VegaML's engine and of the threshold-tuned backbone. It left open whether *any* classifier fitted on the
same 64 examples would do as well. This study answers that.

## Protocol (committed before the run)

Script: `scripts/agentharm_baselines.py`. Data: `agentharm-v1`, the same split as `AGENTHARM.md`.
Every system is fitted on the 64 calibration examples only and evaluated once on the 352 test examples.

| system | features | classifier | what it isolates |
|---|---|---|---|
| `vega-head` | VegaML pooled features (prompt with question and options) | VegaML's `TTTHead` (PCA 128 + logistic) | the reference; must reproduce the main run's predictions |
| `vegafeat-lr` | same VegaML features | ordinary L2 logistic regression, no PCA | VegaML's head vs a textbook head |
| `vegafeat-centroid` | same VegaML features | nearest class centroid, cosine | how separable the features are with nothing fitted |
| `probe-lr` | backbone hidden states of the request alone, mean-pooled, layers 13 and 19 | L2 logistic regression | VegaML's prompt and pooling vs a plain linear probe |
| `tfidf-lr` | TF-IDF word uni+bigrams of request and tool names | L2 logistic regression | whether a neural model is needed at all |

- **Regularisation:** λ is chosen from {0.001, 0.01, 0.1, 1, 10} by 5-fold stratified cross-validation
  inside the calibration split. Ties go to the larger λ, and the seed is fixed (20261024). Nothing is
  tuned on test.
- **Threshold:** 0.5 for every system.
- **Reported:** accuracy, AUROC, missed harmful requests, false alarms, ECE, and a paired bootstrap 95%
  interval (2,000 resamples) for the head minus each baseline.
- **Sanity check:** `vega-head` must match the main run's `vega-head.jsonl` probabilities. If it doesn't,
  the feature extraction differs and the comparison is void.

**How the result is read:**
- If `vegafeat-lr` matches the head, VegaML's head offers no advantage over standard logistic
  regression; the value is in the features.
- If `probe-lr` matches it too, the value is in the frozen backbone, not in VegaML.
- If `tfidf-lr` matches it, the task is solvable from surface words and says little about any model.

Features are cached outside the repo. The committed outputs hold only example IDs, labels and probabilities.

## Results (single run, 2026-10-24, CPU)

Outputs: `reports/vegalab/agentharm_baselines_0.8b/` (`results.json`, and `predictions.jsonl` with
per-example probabilities keyed by ID). VegaML ran live with no offline fallback. Feature sizes:
VegaML 6,144, probe 2,048 (two layers × 1,024).

**Sanity check: passed.** The reproduced `vega-head` gives the same decision as the main run on all 352
test examples (0 flips, correlation 1.0000). The median and 95th-percentile absolute differences are
0.0000; the maximum, on a few examples, is 0.058, which looks like floating-point noise in feature
extraction. Accuracy, AUROC and ECE agree with `AGENTHARM.md` to within 0.001.

| system | accuracy | AUROC | missed harmful (of 176) | false alarms (of 176) | ECE | λ (CV) |
|---|---:|---:|---:|---:|---:|---:|
| `vega-head` | **0.841** | **0.921** | 38 | **18** | 0.063 | (VegaML default) |
| `vegafeat-lr` | **0.841** | 0.887 | 32 | 24 | 0.076 | 10 |
| `vegafeat-centroid` | 0.832 | 0.870 | **31** | 28 | n/a | n/a |
| `probe-lr` | 0.781 | 0.870 | 33 | 44 | **0.058** | 1 |
| `tfidf-lr` | 0.611 | 0.672 | 56 | 81 | 0.111 | 10 |

Paired accuracy differences, head minus baseline (bootstrap 95% CI, same 352 examples):

| comparison | difference | 95% CI |
|---|---:|---|
| vega-head − vegafeat-lr | +0.000 | [−0.023, +0.023] |
| vega-head − vegafeat-centroid | +0.009 | [−0.020, +0.037] |
| vega-head − probe-lr | **+0.060** | [+0.020, +0.099] |
| vega-head − tfidf-lr | **+0.230** | [+0.173, +0.290] |

## What this shows

1. **VegaML's head adds nothing over a textbook classifier at the decision level.** Ordinary logistic
   regression on the same features reached the same accuracy (0.841), and a nearest-centroid rule with
   no fitted weights reached 0.832. Neither difference is distinguishable from zero. The head does rank
   better (AUROC 0.921 vs 0.887) and makes fewer false alarms but more misses, so it trades errors
   rather than avoiding them.
2. **VegaML's features do add something over a plain probe of the same backbone:** +0.060 [+0.020,
   +0.099]. VegaML reads the request together with the question and its options, and pools three summaries
   per layer over spans of that prompt (2 layers × 3 × 1,024 = 6,144). A mean-pooled read of the request alone separates the classes less well (44 false alarms
   vs 18). This is the part of VegaML that helps here.
3. **The task needs a neural model.** TF-IDF on 64 examples reached only 0.611: harmful requests and
   their benign twins share most of their words and tools, so surface vocabulary barely separates them.
4. In short: on this task the value is in **VegaML's prompt-conditioned features**, not in its head and
   not in the physics engine (the engine scored 0.736, see `AGENTHARM.md`).

## Limits

- **λ selection was uninformative:** cross-validated calibration accuracy was saturated (0.98 for
  every λ on VegaML features, 0.97 for the probe), so the tie rule chose the largest λ. With 64
  examples the calibration split can't tell these settings apart.
- One probe design, pre-declared: mean pooling over layers 13 and 19. Other pooling (last token,
  other layers) might narrow the gap to VegaML's features, and none was tried.
- One dataset, one model size (0.8B), CPU. The 352 test items are 176 harmful/benign twin pairs, so
  the examples aren't independent and the intervals are somewhat optimistic.
