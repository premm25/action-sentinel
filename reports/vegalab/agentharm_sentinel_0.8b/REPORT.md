# AgentHarm through Sentinel (VegaML head, 0.8B)

Dataset `agentharm-sentinel-v1` (sha256 `caf785490d96â€¦`), splits test, 352 examples per configuration.
Policy `gov-1.0.0+core@1.0.0+agent_tasks@1.0.0`. Sentinel 0.1.0, Python 3.14.6.

Values are `rate [k/n]`. Lower is better where marked â†“. 95% Wilson intervals are in the JSON.

## Governance outcomes

| metric | 1_head_only_naive | 2_full_sentinel_head |
|---|---:|---:|
| outcome accuracy | 0.824 [290/352] | 0.270 [95/352] |
| false approval rate â†“ | 0.136 [24/176] | 0.000 [0/176] |
| hard violations blocked | n/a (n=0) | n/a (n=0) |
| policy-violation detection (expected BLOCK) | 0.852 [150/176] | 0.000 [0/176] |
| false block rate â†“ | 0.205 [36/176] | 0.000 [0/176] |
| missed escalation rate â†“ | n/a (n=0) | n/a (n=0) |
| over-escalation of safe cases | 0.000 [0/176] | 0.460 [81/176] |
| auto-approval of safe cases | 0.795 [140/176] | 0.540 [95/176] |
| human escalation rate | 0.000 [0/352] | 0.730 [257/352] |
| abstention rate | 0.000 [0/352] | 0.000 [0/352] |
| error rate | 0.006 [2/352] | 0.000 [0/352] |
| disagreement rate | 0.528 [186/352] | 0.534 [188/352] |
| latency p50 ms (pipeline) | 2642.1 | 2.1 |

Configurations:

- **1_head_only_naive**: providers ['vegaml-head']; validation=False, disagreement=True, uncertainty=True, review=True, mode=naive
- **2_full_sentinel_head**: providers ['vegaml-head']; validation=True, disagreement=True, uncertainty=True, review=True, mode=sentinel

## Provider predictions (model accuracy, independent of governance)

| provider | kind | accuracy | coverage | macro F1 | ECE | Brier | AURC | confâ‰¥0.9 errors | abstain | errors+timeouts | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vegaml-head | statistical | 0.824 [290/352] | 0.994 | 0.826 | 0.036 | 0.231 | 0.063 | 11 | 0.000 [0/352] | 2 | 2640.4 |

## Error analysis: 2_full_sentinel_head

- False approvals: none
- False blocks: none
- All outcome mismatches (expectedâ†’actual): APPROVEâ†’HUMAN_REVIEW Ã—81, BLOCKâ†’HUMAN_REVIEW Ã—176

## Notes and limitations

- Expected outcomes encode the reference policy's intent, written by the dataset author. Agreement with them shows consistency with that policy, not real-world correctness.
