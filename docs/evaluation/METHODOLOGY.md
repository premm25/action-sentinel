# Confidence Lab methodology

## Question

Can an agent platform make more reliable decisions by combining deterministic policies,
specialized decision models, measured uncertainty and controlled human intervention?

Confidence Lab measures two things separately:

1. **Model prediction quality.** How well each provider answers the typed `disposition` question
   (`safe` / `needs_review` / `unsafe`) compared with `expected_label`.
2. **Governance correctness.** How well the final Sentinel outcome (`APPROVE` / `HUMAN_REVIEW` /
   `BLOCK` / `ABSTAIN` / `ERROR`) matches `expected_outcome`, with special attention to false
   approvals.

A provider can be inaccurate while governance stays safe, and the reverse. The reports keep the two apart.

## Dataset

`examples/evaluation_datasets/sentinel_synthetic_v1.jsonl` is generated deterministically by
`scripts/build_dataset.py` (seed `20261014`). The file's sha256 starts with `3eafcbd3f6bd`, and the
hash is recorded in every report.

- 86 examples, **all synthetic** (`synthetic: true`). Splits: dev 14, calibration 28, test 44,
  assigned by seeded shuffle.
- Tags cover clear low-risk actions, production changes, medium- and high-impact actions, hard
  policy violations, unsupported action types, missing mandatory fields, contradictory evidence,
  ambiguous or unverified evidence, insufficient evidence, change freeze, and provider timeouts.
- `expected_label` and `expected_outcome` were assigned by the dataset author from the template,
  according to the reference rule pack `core@1.0.0` and the category definitions.
- `mock_script` contains outputs for two **simulated** providers. `mock-a` is about 85% correct and
  confidently wrong on about a third of hard violations; `mock-b` is about 70% correct and abstains
  on some ambiguous cases. They are drawn with seeded noise around the true label and only exercise
  the pipeline.

## Configurations (`action-sentinel lab suite`)

| name | providers | validation | disagreement | uncertainty | review | mode |
|---|---|---|---|---|---|---|
| 1_rules_only | rules | ✓ | ✓ | ✓ | ✓ | sentinel |
| 2_<model>_only_naive | model | ✗ | – | – | – | naive: label to outcome |
| 3_rules_only_naive | rules | ✗ | – | – | – | naive |
| 4_validation_plus_<model> | model | ✓ | ✗ | ✗ | ✗ | sentinel |
| 5_rules_multi_disagreement | rules + model (+ second) | ✓ | ✓ | ✗ | ✗ | sentinel |
| 6_full_sentinel | rules + model (+ second) | ✓ | ✓ | ✓ | ✓ | sentinel |
| ablation_no_X | same as 6 | one of validation / disagreement / uncertainty / review switched off ||||

When review is switched off, a case that would be escalated falls back to the first provider's
label. Only an undisputed permissive label approves; anything else blocks. This models "automation
without humans".

Runs are **paired**. Every configuration uses the same examples and, through per-example caching,
identical provider outputs. Differences between configurations are due to governance, not to model
nondeterminism. Ollama runs at temperature 0 with seed 0.

## Metrics

Provider metrics: accuracy, coverage, selective accuracy, per-class precision/recall/F1 (undefined
when the denominator is 0, reported as n/a), macro F1 over defined classes, confusion matrix,
top-label ECE (10 equal-width bins) with reliability bins, risk–coverage curve with AURC, multiclass
Brier and log loss (only if every answered example has a full distribution over all categories),
abstention, error and timeout rates, latency percentiles, and errors made with confidence ≥ 0.9.

Governance metrics: outcome accuracy; **false approval rate** (APPROVE when another outcome was
expected); hard-violation block rate; policy-violation detection (BLOCK among expected BLOCK); false
block rate; missed escalation (APPROVE when HUMAN_REVIEW was expected); over-escalation and
auto-approval rate among expected APPROVE; escalation, abstention, error and disagreement rates;
pipeline latency. Each rate is stored with k, n and a 95% Wilson interval.

Cost is reported as "not available" because no cost model is configured. No value is invented.

## Rules that protect validity

- Thresholds (`approve_min_confidence = 0.85`, `abstain_below = 0.5`, `max_auto_approve_risk = low`)
  were set before any evaluation and **were not tuned on any split**. In particular, nothing was
  changed after looking at test results.
- The LLM prompt was not changed after the test-split run.
- `action-sentinel lab calibrate` measures ECE on the calibration split only. Calibration profiles are not
  applied automatically.

## Known threats to validity

- **Circularity.** Expected outcomes encode the same reference policy the rule pack implements, so
  `1_rules_only` matching them 44/44 shows internal consistency, not real-world correctness.
- **Small n.** 44 test examples give wide intervals; one example is 2.3 percentage points.
- **Synthetic distribution.** Real agent traffic has different proportions and much more ambiguity.
- **Simulated providers** carry no information about real models.
- **One local model**, one prompt, on CPU.
