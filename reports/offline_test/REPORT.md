# Confidence Lab: offline (rules + simulated providers), test split

Dataset `sentinel-synthetic-v1` (sha256 `3eafcbd3f6bdâ€¦`), splits test, 44 examples per configuration.
Policy `gov-1.0.0+core@1.0.0`. Sentinel 0.1.0, Python 3.14.6.

Values are `rate [k/n]`. Lower is better where marked â†“. 95% Wilson intervals are in the JSON.

## Governance outcomes

| metric | 1_rules_only | 2_mock-a_only_naive | 3_rules_only_naive | 4_validation_plus_mock-a | 5_rules_multi_disagreement | 6_full_sentinel | ablation_no_validation | ablation_no_disagreement | ablation_no_uncertainty | ablation_no_review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| outcome accuracy | 1.000 [44/44] | 0.750 [33/44] | 0.727 [32/44] | 0.500 [22/44] | 0.455 [20/44] | 0.795 [35/44] | 0.477 [21/44] | 0.818 [36/44] | 0.909 [40/44] | 0.455 [20/44] |
| false approval rate â†“ | 0.000 [0/34] | 0.147 [5/34] | 0.088 [3/34] | 0.029 [1/34] | 0.000 [0/34] | 0.000 [0/34] | 0.000 [0/34] | 0.000 [0/34] | 0.000 [0/34] | 0.000 [0/34] |
| hard violations blocked | 1.000 [10/10] | 0.400 [4/10] | 0.500 [5/10] | 1.000 [10/10] | 1.000 [10/10] | 1.000 [10/10] | 0.000 [0/10] | 1.000 [10/10] | 1.000 [10/10] | 1.000 [10/10] |
| policy-violation detection (expected BLOCK) | 1.000 [14/14] | 0.571 [8/14] | 0.357 [5/14] | 1.000 [14/14] | 1.000 [14/14] | 1.000 [14/14] | 0.000 [0/14] | 1.000 [14/14] | 1.000 [14/14] | 1.000 [14/14] |
| false block rate â†“ | 0.000 [0/30] | 0.100 [3/30] | 0.067 [2/30] | 0.700 [21/30] | 0.800 [24/30] | 0.000 [0/30] | 0.000 [0/30] | 0.000 [0/30] | 0.000 [0/30] | 0.800 [24/30] |
| missed escalation rate â†“ | 0.000 [0/20] | 0.050 [1/20] | 0.050 [1/20] | 0.050 [1/20] | 0.000 [0/20] | 0.000 [0/20] | 0.000 [0/20] | 0.000 [0/20] | 0.000 [0/20] | 0.000 [0/20] |
| over-escalation of safe cases | 0.000 [0/10] | 0.100 [1/10] | 0.000 [0/10] | 0.000 [0/10] | 0.000 [0/10] | 0.900 [9/10] | 0.900 [9/10] | 0.800 [8/10] | 0.400 [4/10] | 0.000 [0/10] |
| auto-approval of safe cases | 1.000 [10/10] | 0.800 [8/10] | 1.000 [10/10] | 0.800 [8/10] | 0.600 [6/10] | 0.100 [1/10] | 0.100 [1/10] | 0.200 [2/10] | 0.600 [6/10] | 0.600 [6/10] |
| human escalation rate | 0.455 [20/44] | 0.455 [20/44] | 0.545 [24/44] | 0.000 [0/44] | 0.000 [0/44] | 0.659 [29/44] | 0.977 [43/44] | 0.636 [28/44] | 0.545 [24/44] | 0.000 [0/44] |
| abstention rate | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] |
| error rate | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] | 0.000 [0/44] |
| disagreement rate | 0.045 [2/44] | 0.159 [7/44] | 0.045 [2/44] | 0.159 [7/44] | 0.614 [27/44] | 0.614 [27/44] | 0.614 [27/44] | 0.614 [27/44] | 0.614 [27/44] | 0.614 [27/44] |
| latency p50 ms (pipeline) | 1.4 | 2.1 | 1.5 | 1.7 | 1.9 | 2.4 | 2.9 | 2.9 | 5.4 | 6.4 |

Configurations:

- **1_rules_only**: providers ['rules']; validation=True, disagreement=True, uncertainty=True, review=True, mode=sentinel
- **2_mock-a_only_naive**: providers ['mock-a']; validation=False, disagreement=True, uncertainty=True, review=True, mode=naive
- **3_rules_only_naive**: providers ['rules']; validation=False, disagreement=True, uncertainty=True, review=True, mode=naive
- **4_validation_plus_mock-a**: providers ['mock-a']; validation=True, disagreement=False, uncertainty=False, review=False, mode=sentinel
- **5_rules_multi_disagreement**: providers ['rules', 'mock-a', 'mock-b']; validation=True, disagreement=True, uncertainty=False, review=False, mode=sentinel
- **6_full_sentinel**: providers ['rules', 'mock-a', 'mock-b']; validation=True, disagreement=True, uncertainty=True, review=True, mode=sentinel
- **ablation_no_validation**: providers ['rules', 'mock-a', 'mock-b']; validation=False, disagreement=True, uncertainty=True, review=True, mode=sentinel
- **ablation_no_disagreement**: providers ['rules', 'mock-a', 'mock-b']; validation=True, disagreement=False, uncertainty=True, review=True, mode=sentinel
- **ablation_no_uncertainty**: providers ['rules', 'mock-a', 'mock-b']; validation=True, disagreement=True, uncertainty=False, review=True, mode=sentinel
- **ablation_no_review**: providers ['rules', 'mock-a', 'mock-b']; validation=True, disagreement=True, uncertainty=True, review=False, mode=sentinel

## Provider predictions (model accuracy, independent of governance)

| provider | kind | accuracy | coverage | macro F1 | ECE | Brier | AURC | confâ‰¥0.9 errors | abstain | errors+timeouts | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | deterministic | 0.727 [32/44] | 1.000 | 0.706 | n/a | n/a | n/a | 0 | 0.000 [0/44] | 0 | 0.1 |
| mock-a | test_double | 0.750 [33/44] | 1.000 | 0.729 | 0.256 | 0.485 | 0.362 | 7 | 0.000 [0/44] | 0 | 0.0 |
| mock-b | test_double | 0.750 [33/44] | 0.977 | 0.760 | 0.138 | 0.302 | 0.058 | 0 | 0.023 [1/44] | 0 | 0.0 |

## Error analysis: 6_full_sentinel

- False approvals: none
- False blocks: none
- All outcome mismatches (expectedâ†’actual): APPROVEâ†’HUMAN_REVIEW Ã—9

## Notes and limitations

- All examples are synthetic; results describe behaviour on this dataset only.
- n=44: confidence intervals are wide; differences of a few examples are not significant.
- mock-a is a SIMULATED provider scripted with seeded noise around the true label; its accuracy and calibration are properties of the script, not of any model.
- mock-b is a SIMULATED provider scripted with seeded noise around the true label; its accuracy and calibration are properties of the script, not of any model.
- Expected outcomes encode the reference policy's intent, written by the dataset author. Agreement with them shows consistency with that policy, not real-world correctness.
