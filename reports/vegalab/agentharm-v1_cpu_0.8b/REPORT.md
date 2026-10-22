# Vega lab: AgentHarm (agent-request safety), 0.8B on CPU

Run `vegalab-agentharm-v1-20261022_123045` · dataset `agentharm-v1` · split `test` · model `0.8b` on `cpu`.

Values are `value [k/n]` where a count applies. All numbers are computed from the saved predictions in this folder.

## Accuracy by task

| task | backbone-optscore | majority | vega-auto | vega-head | vega-off |
|---|---:|---:|---:|---:|---:|
| agentharm | 0.500 [176/352] | 0.500 [176/352] | 0.733 [258/352] | 0.841 [296/352] | 0.736 [259/352] |
| **macro mean** | 0.500 | 0.500 | 0.733 | 0.841 | 0.736 |

## Balanced accuracy

| task | backbone-optscore | majority | vega-auto | vega-head | vega-off |
|---|---:|---:|---:|---:|---:|
| agentharm | 0.500 | 0.500 | 0.733 | 0.841 | 0.736 |

## Expected calibration error (10 bins) ↓

| task | backbone-optscore | majority | vega-auto | vega-head | vega-off |
|---|---:|---:|---:|---:|---:|
| agentharm | 0.133 | 0.000 | 0.117 | 0.063 | 0.068 |

## Brier score ↓

| task | backbone-optscore | majority | vega-auto | vega-head | vega-off |
|---|---:|---:|---:|---:|---:|
| agentharm | 0.515 | 0.500 | 0.388 | 0.234 | 0.353 |

## AURC (risk–coverage) ↓

| task | backbone-optscore | majority | vega-auto | vega-head | vega-off |
|---|---:|---:|---:|---:|---:|
| agentharm | 0.316 | 0.846 | 0.145 | 0.059 | 0.123 |

## VegaML abstention, conformal sets, unbound (native signals)

| system | task | abstain rate | coverage | selective acc. | acc. on abstained | conformal coverage (α=0.1) | mean set size | unbound |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vega-auto | agentharm | 0.301 [106/352] | 0.699 | 0.785 [193/246] | 0.613 [65/106] | 0.875 [308/352] | 1.38 | 0.122 [43/352] |
| vega-head | agentharm | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| vega-off | agentharm | 0.312 [110/352] | 0.688 | 0.810 [196/242] | 0.573 [63/110] | 0.991 [349/352] | 1.77 | 0.946 [333/352] |

## Post-hoc temperature (fitted on the calibration split, applied to test)

| system | task | T | ECE before | ECE after | log loss before | log loss after |
|---|---|---:|---:|---:|---:|---:|
| backbone-optscore | agentharm | 4.26 | 0.133 | 0.032 | 0.708 | 0.690 |
| majority | agentharm | 0.05 | 0.000 | 0.000 | 0.693 | 0.693 |
| vega-auto | agentharm | 1.49 | 0.117 | 0.074 | 0.597 | 0.539 |
| vega-off | agentharm | 0.42 | 0.068 | 0.064 | 0.529 | 0.512 |

## Paired accuracy differences (bootstrap 95% CI, same examples)

| comparison | task | n | difference | 95% CI | only A correct | only B correct |
|---|---|---:|---:|---|---:|---:|
| vega-off vs backbone-optscore | agentharm | 352 | +0.236 | [+0.185, +0.293] | 100 | 17 |
| vega-head vs vega-off | agentharm | 352 | +0.105 | [+0.065, +0.148] | 48 | 11 |
| vega-auto vs vega-off | agentharm | 352 | -0.003 | [-0.057, +0.054] | 44 | 45 |
| vega-head vs backbone-optscore | agentharm | 352 | +0.341 | [+0.284, +0.403] | 138 | 18 |

## Latency per example (wall clock, ms)

| task | backbone-optscore | majority | vega-auto | vega-head | vega-off |
|---|---:|---:|---:|---:|---:|
| agentharm | 4459 | 0 | 3248 | 2477 | 1856 |

## Slices (all test tasks pooled)

- **backbone-optscore** by difficulty: brief 0.500 [88/176]; detailed 0.500 [88/176]
- **backbone-optscore** by length: medium(30-120w) 0.500 [136/272]; short(<30w) 0.500 [40/80]
- **backbone-optscore** by label_space: 2 0.500 [176/352]
- **majority** by difficulty: brief 0.500 [88/176]; detailed 0.500 [88/176]
- **majority** by length: medium(30-120w) 0.500 [136/272]; short(<30w) 0.500 [40/80]
- **majority** by label_space: 2 0.500 [176/352]
- **vega-auto** by difficulty: brief 0.710 [125/176]; detailed 0.756 [133/176]
- **vega-auto** by length: medium(30-120w) 0.743 [202/272]; short(<30w) 0.700 [56/80]
- **vega-auto** by label_space: 2 0.733 [258/352]
- **vega-head** by difficulty: brief 0.835 [147/176]; detailed 0.847 [149/176]
- **vega-head** by length: medium(30-120w) 0.838 [228/272]; short(<30w) 0.850 [68/80]
- **vega-head** by label_space: 2 0.841 [296/352]
- **vega-off** by difficulty: brief 0.761 [134/176]; detailed 0.710 [125/176]
- **vega-off** by length: medium(30-120w) 0.728 [198/272]; short(<30w) 0.762 [61/80]
- **vega-off** by label_space: 2 0.736 [259/352]
