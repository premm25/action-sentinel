# RQ7: Sentinel governance with VegaML 0.8B (synthetic agent actions, test split)

Dataset `sentinel-synthetic-v1` (sha256 `3eafcbd3f6bdâ€¦`), splits test, 44 examples per configuration.
Policy `gov-1.0.0+core@1.0.0`. Sentinel 0.1.0, Python 3.14.6.

Values are `rate [k/n]`. Lower is better where marked â†“. 95% Wilson intervals are in the JSON.

## Governance outcomes

| metric | 1_rules_only | 2_vegaml-off_only_naive | 3_rules_only_naive | 3_vegaml-auto_only_naive | 4_validation_plus_vegaml-off | 5_rules_multi_disagreement | 6_full_sentinel | ablation_no_validation | ablation_no_disagreement | ablation_no_uncertainty | ablation_no_review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| outcome accuracy | 1.000 [44/44] | 0.000 [0/44] | 0.727 [32/44] | 0.000 [0/44] | 0.318 [14/44] | 0.545 [24/44] | 0.773 [34/44] | 0.455 [20/44] | 0.773 [34/44] | 1.000 [44/44] | 0.545 [24/44] |
| false approval rate â†“ | 0.000 [0/34] | 0.000 [0/34] | 0.088 [3/34] | 0.000 [0/34] | 0.000 [0/34] | 0.029 [1/34] | 0.000 [0/34] | 0.000 [0/34] | 0.000 [0/34] | 0.000 [0/34] | 0.029 [1/34] |
| hard violations blocked | 1.000 [10/10] | 0.000 [0/10] | 0.500 [5/10] | 0.000 [0/10] | 1.000 [10/10] | 1.000 [10/10] | 1.000 [10/10] | 0.000 [0/10] | 1.000 [10/10] | 1.000 [10/10] | 1.000 [10/10] |
| policy-violation detection (expected BLOCK) | 1.000 [14/14] | 0.000 [0/14] | 0.357 [5/14] | 0.000 [0/14] | 1.000 [14/14] | 1.000 [14/14] | 1.000 [14/14] | 0.000 [0/14] | 1.000 [14/14] | 1.000 [14/14] | 1.000 [14/14] |
| false block rate â†“ | 0.000 [0/30] | 0.000 [0/30] | 0.067 [2/30] | 0.000 [0/30] | 0.000 [0/30] | 0.633 [19/30] | 0.000 [0/30] | 0.000 [0/30] | 0.000 [0/30] | 0.000 [0/30] | 0.633 [19/30] |
| missed escalation rate â†“ | 0.000 [0/20] | 0.000 [0/20] | 0.050 [1/20] | 0.000 [0/20] | 0.000 [0/20] | 0.050 [1/20] | 0.000 [0/20] | 0.000 [0/20] | 0.000 [0/20] | 0.000 [0/20] | 0.050 [1/20] |
| over-escalation of safe cases | 0.000 [0/10] | 1.000 [10/10] | 0.000 [0/10] | 1.000 [10/10] | 1.000 [10/10] | 0.000 [0/10] | 1.000 [10/10] | 1.000 [10/10] | 1.000 [10/10] | 0.000 [0/10] | 0.000 [0/10] |
| auto-approval of safe cases | 1.000 [10/10] | 0.000 [0/10] | 1.000 [10/10] | 0.000 [0/10] | 0.000 [0/10] | 1.000 [10/10] | 0.000 [0/10] | 0.000 [0/10] | 0.000 [0/10] | 1.000 [10/10] | 1.000 [10/10] |
| human escalation rate | 0.455 [20/44] | 0.000 [0/44] | 0.545 [24/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.682 [30/44] | 1.000 [44/44] | 0.682 [30/44] | 0.455 [20/44] | 0.000 [0/44] |
| abstention rate | 0.000 [0/44] | 1.000 [44/44] | 0.000 [0/44] | 1.000 [44/44] | 0.682 [30/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] |
| error rate | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] |
| disagreement rate | 0.045 [2/44] | 0.000 [0/44] | 0.045 [2/44] | 0.000 [0/44] | 0.000 [0/44] | 0.045 [2/44] | 0.045 [2/44] | 0.045 [2/44] | 0.045 [2/44] | 0.045 [2/44] | 0.045 [2/44] |
| latency p50 ms (pipeline) | 0.6 | 2477.8 | 0.6 | 2621.9 | 0.7 | 0.8 | 0.9 | 0.8 | 0.8 | 0.8 | 0.9 |

Configurations:

- **1_rules_only**: providers ['rules']; validation=True, disagreement=True, uncertainty=True, review=True, mode=sentinel
- **2_vegaml-off_only_naive**: providers ['vegaml-off']; validation=False, disagreement=True, uncertainty=True, review=True, mode=naive
- **3_rules_only_naive**: providers ['rules']; validation=False, disagreement=True, uncertainty=True, review=True, mode=naive
- **3_vegaml-auto_only_naive**: providers ['vegaml-auto']; validation=False, disagreement=True, uncertainty=True, review=True, mode=naive
- **4_validation_plus_vegaml-off**: providers ['vegaml-off']; validation=True, disagreement=False, uncertainty=False, review=False, mode=sentinel
- **5_rules_multi_disagreement**: providers ['rules', 'vegaml-off', 'vegaml-auto']; validation=True, disagreement=True, uncertainty=False, review=False, mode=sentinel
- **6_full_sentinel**: providers ['rules', 'vegaml-off', 'vegaml-auto']; validation=True, disagreement=True, uncertainty=True, review=True, mode=sentinel
- **ablation_no_validation**: providers ['rules', 'vegaml-off', 'vegaml-auto']; validation=False, disagreement=True, uncertainty=True, review=True, mode=sentinel
- **ablation_no_disagreement**: providers ['rules', 'vegaml-off', 'vegaml-auto']; validation=True, disagreement=False, uncertainty=True, review=True, mode=sentinel
- **ablation_no_uncertainty**: providers ['rules', 'vegaml-off', 'vegaml-auto']; validation=True, disagreement=True, uncertainty=False, review=True, mode=sentinel
- **ablation_no_review**: providers ['rules', 'vegaml-off', 'vegaml-auto']; validation=True, disagreement=True, uncertainty=True, review=False, mode=sentinel

## Provider predictions (model accuracy, independent of governance)

| provider | kind | accuracy | coverage | macro F1 | ECE | Brier | AURC | confâ‰¥0.9 errors | abstain | errors+timeouts | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | deterministic | 0.727 [32/44] | 1.000 | 0.706 | n/a | n/a | n/a | 0 | 0.000 [0/44] | 0 | 0.0 |
| vegaml-off | statistical | 0.000 [0/44] | 0.000 | n/a | n/a | n/a | n/a | 0 | 1.000 [44/44] | 0 | 2476.0 |
| vegaml-auto | statistical | 0.000 [0/44] | 0.000 | n/a | n/a | n/a | n/a | 0 | 1.000 [44/44] | 0 | 2620.2 |

## Error analysis: 6_full_sentinel

- False approvals: none
- False blocks: none
- All outcome mismatches (expectedâ†’actual): APPROVEâ†’HUMAN_REVIEW Ã—10

## Notes and limitations

- All examples are synthetic; results describe behaviour on this dataset only.
- n=44: confidence intervals are wide; differences of a few examples are not significant.
- Expected outcomes encode the reference policy's intent, written by the dataset author. Agreement with them shows consistency with that policy, not real-world correctness.
