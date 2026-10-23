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

## Results

Pending: filled in after the single run.
