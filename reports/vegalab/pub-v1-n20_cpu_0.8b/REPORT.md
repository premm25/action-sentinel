# Vega lab: local CPU verification run (0.8B, 20 test examples per task)

Run `vegalab-pub-v1-20261018_020353` · dataset `pub-v1` · split `test` · model `0.8b` on `cpu`.

Values are `value [k/n]` where a count applies. All numbers are computed from the saved predictions in this folder.

## Accuracy by task

| task | backbone-optscore | majority | vega-auto | vega-off |
|---|---:|---:|---:|---:|
| anli_r3 | 0.500 [10/20] | 0.300 [6/20] | 0.250 [5/20] | 0.250 [5/20] |
| boolq | 0.650 [13/20] | 0.650 [13/20] | 0.600 [12/20] | 0.600 [12/20] |
| dbpedia14 | 0.300 [6/20] | 0.000 [0/20] | 0.850 [17/20] | 0.850 [17/20] |
| dbpedia14_k2 | 0.850 [17/20] | 0.400 [8/20] | 0.950 [19/20] | 0.950 [19/20] |
| dbpedia14_k4 | 0.650 [13/20] | 0.000 [0/20] | 0.900 [18/20] | 0.900 [18/20] |
| dbpedia14_k8 | 0.650 [13/20] | 0.000 [0/20] | 0.850 [17/20] | 0.850 [17/20] |
| glue_rte | 0.500 [10/20] | 0.500 [10/20] | 0.650 [13/20] | 0.650 [13/20] |
| massive_intent_en | 0.050 [1/20] | 0.000 [0/20] | 0.450 [9/20] | 0.500 [10/20] |
| prompt_injections | 0.550 [11/20] | 0.550 [11/20] | 0.700 [14/20] | 0.700 [14/20] |
| sst2 | 0.500 [10/20] | 0.600 [12/20] | 0.950 [19/20] | 0.950 [19/20] |
| superglue_cb | 0.300 [6/20] | 0.550 [11/20] | 0.550 [11/20] | 0.550 [11/20] |
| yelp_review_stars | 0.200 [4/20] | 0.150 [3/20] | 0.650 [13/20] | 0.650 [13/20] |
| **macro mean** | 0.475 | 0.308 | 0.696 | 0.700 |

## Balanced accuracy

| task | backbone-optscore | majority | vega-auto | vega-off |
|---|---:|---:|---:|---:|
| anli_r3 | 0.333 | 0.333 | 0.389 | 0.389 |
| boolq | 0.632 | 0.500 | 0.626 | 0.626 |
| dbpedia14 | 0.233 | 0.000 | 0.833 | 0.833 |
| dbpedia14_k2 | 0.833 | 0.333 | 0.933 | 0.933 |
| dbpedia14_k4 | 0.567 | 0.000 | 0.867 | 0.867 |
| dbpedia14_k8 | 0.600 | 0.000 | 0.833 | 0.833 |
| glue_rte | 0.500 | 0.500 | 0.650 | 0.650 |
| massive_intent_en | 0.077 | 0.000 | 0.385 | 0.526 |
| prompt_injections | 0.591 | 0.500 | 0.707 | 0.707 |
| sst2 | 0.583 | 0.500 | 0.938 | 0.938 |
| superglue_cb | 0.333 | 0.333 | 0.333 | 0.333 |
| yelp_review_stars | 0.200 | 0.200 | 0.597 | 0.597 |

## Expected calibration error (10 bins) ↓

| task | backbone-optscore | majority | vega-auto | vega-off |
|---|---:|---:|---:|---:|
| anli_r3 | 0.031 | 0.122 | 0.410 | 0.410 |
| boolq | 0.124 | 0.052 | 0.156 | 0.156 |
| dbpedia14 | 0.156 | 0.160 | 0.455 | 0.455 |
| dbpedia14_k2 | 0.127 | 0.222 | 0.073 | 0.073 |
| dbpedia14_k4 | 0.251 | 0.388 | 0.181 | 0.181 |
| dbpedia14_k8 | 0.390 | 0.213 | 0.322 | 0.322 |
| glue_rte | 0.054 | 0.037 | 0.241 | 0.241 |
| massive_intent_en | 0.015 | 0.050 | 0.255 | 0.359 |
| prompt_injections | 0.030 | 0.084 | 0.234 | 0.234 |
| sst2 | 0.051 | 0.076 | 0.083 | 0.083 |
| superglue_cb | 0.167 | 0.008 | 0.317 | 0.317 |
| yelp_review_stars | 0.020 | 0.085 | 0.165 | 0.165 |

## Brier score ↓

| task | backbone-optscore | majority | vega-auto | vega-off |
|---|---:|---:|---:|---:|
| anli_r3 | 0.633 | 0.663 | 0.841 | 0.841 |
| boolq | 0.493 | 0.460 | 0.405 | 0.405 |
| dbpedia14 | 0.867 | 0.980 | 0.482 | 0.483 |
| dbpedia14_k2 | 0.228 | 0.626 | 0.084 | 0.084 |
| dbpedia14_k4 | 0.557 | 0.905 | 0.159 | 0.159 |
| dbpedia14_k8 | 0.734 | 0.946 | 0.280 | 0.280 |
| glue_rte | 0.498 | 0.503 | 0.495 | 0.495 |
| massive_intent_en | 0.983 | 0.992 | 0.772 | 0.824 |
| prompt_injections | 0.495 | 0.509 | 0.339 | 0.339 |
| sst2 | 0.476 | 0.491 | 0.113 | 0.113 |
| superglue_cb | 0.796 | 0.603 | 0.658 | 0.658 |
| yelp_review_stars | 0.793 | 0.778 | 0.557 | 0.557 |

## AURC (risk–coverage) ↓

| task | backbone-optscore | majority | vega-auto | vega-off |
|---|---:|---:|---:|---:|
| anli_r3 | 0.544 | 0.820 | 0.810 | 0.810 |
| boolq | 0.359 | 0.444 | 0.183 | 0.183 |
| dbpedia14 | 0.735 | 1.000 | 0.049 | 0.049 |
| dbpedia14_k2 | 0.040 | 0.780 | 0.005 | 0.005 |
| dbpedia14_k4 | 0.270 | 1.000 | 0.010 | 0.010 |
| dbpedia14_k8 | 0.280 | 1.000 | 0.016 | 0.016 |
| glue_rte | 0.248 | 0.375 | 0.188 | 0.188 |
| massive_intent_en | 0.943 | 1.000 | 0.427 | 0.408 |
| prompt_injections | 0.432 | 0.574 | 0.125 | 0.125 |
| sst2 | 0.307 | 0.231 | 0.011 | 0.011 |
| superglue_cb | 0.687 | 0.429 | 0.330 | 0.330 |
| yelp_review_stars | 0.731 | 0.795 | 0.299 | 0.299 |

## VegaML abstention, conformal sets, unbound (native signals)

| system | task | abstain rate | coverage | selective acc. | acc. on abstained | conformal coverage (α=0.1) | mean set size | unbound |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vega-auto | anli_r3 | 0.550 [11/20] | 0.450 | 0.222 [2/9] | 0.273 [3/11] | 0.500 [10/20] | 1.60 | 0.000 [0/20] |
| vega-auto | boolq | 0.300 [6/20] | 0.700 | 0.643 [9/14] | 0.500 [3/6] | 0.750 [15/20] | 1.20 | 0.000 [0/20] |
| vega-auto | dbpedia14 | 0.900 [18/20] | 0.100 | 1.000 [2/2] | 0.833 [15/18] | 0.850 [17/20] | 1.00 | 0.350 [7/20] |
| vega-auto | dbpedia14_k2 | 0.100 [2/20] | 0.900 | 1.000 [18/18] | 0.500 [1/2] | 1.000 [20/20] | 1.20 | 0.200 [4/20] |
| vega-auto | dbpedia14_k4 | 0.200 [4/20] | 0.800 | 1.000 [16/16] | 0.500 [2/4] | 1.000 [20/20] | 1.40 | 0.450 [9/20] |
| vega-auto | dbpedia14_k8 | 0.450 [9/20] | 0.550 | 1.000 [11/11] | 0.667 [6/9] | 0.850 [17/20] | 1.00 | 0.400 [8/20] |
| vega-auto | glue_rte | 0.150 [3/20] | 0.850 | 0.706 [12/17] | 0.333 [1/3] | 0.750 [15/20] | 1.10 | 0.100 [2/20] |
| vega-auto | massive_intent_en | 0.450 [9/20] | 0.550 | 0.636 [7/11] | 0.222 [2/9] | 0.450 [9/20] | 1.30 | 0.000 [0/20] |
| vega-auto | prompt_injections | 0.100 [2/20] | 0.900 | 0.778 [14/18] | 0.000 [0/2] | 0.750 [15/20] | 1.05 | 0.000 [0/20] |
| vega-auto | sst2 | 0.050 [1/20] | 0.950 | 0.947 [18/19] | 1.000 [1/1] | 1.000 [20/20] | 1.25 | 0.350 [7/20] |
| vega-auto | superglue_cb | 0.100 [2/20] | 0.900 | 0.611 [11/18] | 0.000 [0/2] | 0.550 [11/20] | 1.05 | 0.000 [0/20] |
| vega-auto | yelp_review_stars | 0.750 [15/20] | 0.250 | 0.800 [4/5] | 0.600 [9/15] | 0.950 [19/20] | 2.25 | 0.000 [0/20] |
| vega-off | anli_r3 | 0.550 [11/20] | 0.450 | 0.222 [2/9] | 0.273 [3/11] | 0.500 [10/20] | 1.60 | 0.000 [0/20] |
| vega-off | boolq | 0.300 [6/20] | 0.700 | 0.643 [9/14] | 0.500 [3/6] | 0.750 [15/20] | 1.20 | 0.000 [0/20] |
| vega-off | dbpedia14 | 0.900 [18/20] | 0.100 | 1.000 [2/2] | 0.833 [15/18] | 0.850 [17/20] | 1.00 | 0.350 [7/20] |
| vega-off | dbpedia14_k2 | 0.100 [2/20] | 0.900 | 1.000 [18/18] | 0.500 [1/2] | 1.000 [20/20] | 1.20 | 0.200 [4/20] |
| vega-off | dbpedia14_k4 | 0.200 [4/20] | 0.800 | 1.000 [16/16] | 0.500 [2/4] | 1.000 [20/20] | 1.40 | 0.450 [9/20] |
| vega-off | dbpedia14_k8 | 0.450 [9/20] | 0.550 | 1.000 [11/11] | 0.667 [6/9] | 0.850 [17/20] | 1.00 | 0.400 [8/20] |
| vega-off | glue_rte | 0.150 [3/20] | 0.850 | 0.706 [12/17] | 0.333 [1/3] | 0.750 [15/20] | 1.10 | 0.100 [2/20] |
| vega-off | massive_intent_en | 0.900 [18/20] | 0.100 | 0.500 [1/2] | 0.500 [9/18] | 0.500 [10/20] | 1.00 | 0.500 [10/20] |
| vega-off | prompt_injections | 0.100 [2/20] | 0.900 | 0.778 [14/18] | 0.000 [0/2] | 0.750 [15/20] | 1.05 | 0.000 [0/20] |
| vega-off | sst2 | 0.050 [1/20] | 0.950 | 0.947 [18/19] | 1.000 [1/1] | 1.000 [20/20] | 1.25 | 0.350 [7/20] |
| vega-off | superglue_cb | 0.100 [2/20] | 0.900 | 0.611 [11/18] | 0.000 [0/2] | 0.550 [11/20] | 1.05 | 0.000 [0/20] |
| vega-off | yelp_review_stars | 0.750 [15/20] | 0.250 | 0.800 [4/5] | 0.600 [9/15] | 0.950 [19/20] | 2.25 | 0.000 [0/20] |

## Post-hoc temperature (fitted on the calibration split, applied to test)

| system | task | T | ECE before | ECE after | log loss before | log loss after |
|---|---|---:|---:|---:|---:|---:|
| backbone-optscore | anli_r3 | 1.61 | 0.031 | 0.080 | 1.052 | 1.051 |
| backbone-optscore | boolq | 0.32 | 0.124 | 0.114 | 0.686 | 0.685 |
| backbone-optscore | dbpedia14 | 0.42 | 0.156 | 0.138 | 2.182 | 2.041 |
| backbone-optscore | glue_rte | 0.52 | 0.054 | 0.249 | 0.691 | 0.699 |
| backbone-optscore | massive_intent_en | 0.88 | 0.015 | 0.024 | 3.985 | 4.024 |
| backbone-optscore | prompt_injections | 20.09 | 0.030 | 0.049 | 0.688 | 0.693 |
| backbone-optscore | sst2 | 0.51 | 0.051 | 0.097 | 0.669 | 0.660 |
| backbone-optscore | superglue_cb | 0.84 | 0.167 | 0.191 | 1.339 | 1.401 |
| backbone-optscore | yelp_review_stars | 0.26 | 0.020 | 0.091 | 1.592 | 1.565 |
| majority | anli_r3 | 2.52 | 0.122 | 0.068 | 1.087 | 1.088 |
| majority | boolq | 1.96 | 0.052 | 0.100 | 0.653 | 0.668 |
| majority | dbpedia14 | 20.09 | 0.160 | 0.075 | 2.954 | 2.650 |
| majority | glue_rte | 20.09 | 0.037 | 0.002 | 0.696 | 0.693 |
| majority | massive_intent_en | 1.00 | 0.050 | 0.050 | 4.265 | 4.265 |
| majority | prompt_injections | 0.25 | 0.084 | 0.348 | 0.703 | 1.087 |
| majority | sst2 | 0.48 | 0.076 | 0.050 | 0.685 | 0.678 |
| majority | superglue_cb | 0.90 | 0.008 | 0.006 | 1.035 | 1.054 |
| majority | yelp_review_stars | 0.52 | 0.085 | 0.118 | 1.551 | 1.518 |
| vega-auto | anli_r3 | 5.75 | 0.410 | 0.131 | 1.306 | 1.102 |
| vega-auto | boolq | 1.13 | 0.156 | 0.157 | 0.572 | 0.571 |
| vega-auto | dbpedia14 | 0.28 | 0.455 | 0.126 | 1.107 | 0.591 |
| vega-auto | glue_rte | 16.04 | 0.241 | 0.113 | 0.754 | 0.658 |
| vega-auto | massive_intent_en | 1.38 | 0.255 | 0.223 | 3.038 | 2.659 |
| vega-auto | prompt_injections | 0.70 | 0.234 | 0.204 | 0.526 | 0.632 |
| vega-auto | sst2 | 0.58 | 0.083 | 0.081 | 0.192 | 0.155 |
| vega-auto | superglue_cb | 7.77 | 0.317 | 0.143 | 1.145 | 1.017 |
| vega-auto | yelp_review_stars | 0.56 | 0.165 | 0.256 | 0.959 | 0.984 |
| vega-off | anli_r3 | 5.75 | 0.410 | 0.131 | 1.306 | 1.102 |
| vega-off | boolq | 1.13 | 0.156 | 0.157 | 0.572 | 0.571 |
| vega-off | dbpedia14 | 0.28 | 0.455 | 0.126 | 1.108 | 0.591 |
| vega-off | glue_rte | 16.04 | 0.241 | 0.113 | 0.754 | 0.658 |
| vega-off | massive_intent_en | 0.50 | 0.359 | 0.268 | 2.520 | 2.538 |
| vega-off | prompt_injections | 0.70 | 0.234 | 0.204 | 0.526 | 0.632 |
| vega-off | sst2 | 0.58 | 0.083 | 0.081 | 0.192 | 0.155 |
| vega-off | superglue_cb | 7.77 | 0.317 | 0.143 | 1.145 | 1.017 |
| vega-off | yelp_review_stars | 0.56 | 0.165 | 0.256 | 0.959 | 0.984 |

## Paired accuracy differences (bootstrap 95% CI, same examples)

| comparison | task | n | difference | 95% CI | only A correct | only B correct |
|---|---|---:|---:|---|---:|---:|
| vega-off vs backbone-optscore | anli_r3 | 20 | -0.250 | [-0.600, +0.100] | 5 | 10 |
| vega-off vs backbone-optscore | boolq | 20 | -0.050 | [-0.350, +0.200] | 3 | 4 |
| vega-off vs backbone-optscore | dbpedia14 | 20 | +0.550 | [+0.350, +0.750] | 11 | 0 |
| vega-off vs backbone-optscore | dbpedia14_k2 | 20 | +0.100 | [-0.100, +0.300] | 3 | 1 |
| vega-off vs backbone-optscore | dbpedia14_k4 | 20 | +0.250 | [+0.000, +0.500] | 6 | 1 |
| vega-off vs backbone-optscore | dbpedia14_k8 | 20 | +0.200 | [+0.050, +0.400] | 4 | 0 |
| vega-off vs backbone-optscore | glue_rte | 20 | +0.150 | [-0.250, +0.550] | 10 | 7 |
| vega-off vs backbone-optscore | massive_intent_en | 20 | +0.450 | [+0.250, +0.650] | 9 | 0 |
| vega-off vs backbone-optscore | prompt_injections | 20 | +0.150 | [-0.100, +0.400] | 5 | 2 |
| vega-off vs backbone-optscore | sst2 | 20 | +0.450 | [+0.200, +0.700] | 10 | 1 |
| vega-off vs backbone-optscore | superglue_cb | 20 | +0.250 | [-0.150, +0.600] | 11 | 6 |
| vega-off vs backbone-optscore | yelp_review_stars | 20 | +0.450 | [+0.100, +0.750] | 12 | 3 |
| vega-auto vs vega-off | anli_r3 | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | boolq | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | dbpedia14 | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | dbpedia14_k2 | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | dbpedia14_k4 | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | dbpedia14_k8 | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | glue_rte | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | massive_intent_en | 20 | -0.050 | [-0.250, +0.150] | 2 | 3 |
| vega-auto vs vega-off | prompt_injections | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | sst2 | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | superglue_cb | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-auto vs vega-off | yelp_review_stars | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-off vs majority | anli_r3 | 20 | -0.050 | [-0.350, +0.250] | 4 | 5 |
| vega-off vs majority | boolq | 20 | -0.050 | [-0.400, +0.300] | 5 | 6 |
| vega-off vs majority | dbpedia14 | 20 | +0.850 | [+0.700, +1.000] | 17 | 0 |
| vega-off vs majority | dbpedia14_k2 | 20 | +0.550 | [+0.300, +0.750] | 12 | 1 |
| vega-off vs majority | dbpedia14_k4 | 20 | +0.900 | [+0.750, +1.000] | 18 | 0 |
| vega-off vs majority | dbpedia14_k8 | 20 | +0.850 | [+0.650, +1.000] | 17 | 0 |
| vega-off vs majority | glue_rte | 20 | +0.150 | [+0.000, +0.300] | 3 | 0 |
| vega-off vs majority | massive_intent_en | 20 | +0.500 | [+0.300, +0.700] | 10 | 0 |
| vega-off vs majority | prompt_injections | 20 | +0.150 | [-0.150, +0.450] | 7 | 4 |
| vega-off vs majority | sst2 | 20 | +0.350 | [+0.150, +0.600] | 7 | 0 |
| vega-off vs majority | superglue_cb | 20 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| vega-off vs majority | yelp_review_stars | 20 | +0.500 | [+0.200, +0.750] | 12 | 2 |

## Latency per example (wall clock, ms)

| task | backbone-optscore | majority | vega-auto | vega-off |
|---|---:|---:|---:|---:|
| anli_r3 | 4418 | 0 | 2734 | 6621 |
| boolq | 4018 | 0 | 2980 | 5749 |
| dbpedia14 | 10138 | 0 | 3736 | 3943 |
| dbpedia14_k2 | 3689 | 0 | 2170 | 2180 |
| dbpedia14_k4 | 5568 | 0 | 2405 | 2560 |
| dbpedia14_k8 | 7620 | 0 | 3052 | 3414 |
| glue_rte | 4344 | 0 | 1951 | 2361 |
| massive_intent_en | 37187 | 0 | 7056 | 8358 |
| prompt_injections | 2523 | 0 | 1313 | 1426 |
| sst2 | 4797 | 0 | 1031 | 1315 |
| superglue_cb | 8974 | 0 | 1798 | 2412 |
| yelp_review_stars | 8460 | 0 | 2780 | 3877 |

## Slices (all test tasks pooled)

- **backbone-optscore** by difficulty: adversarial 0.500 [10/20]; ordinal 0.200 [4/20]; standard 0.550 [99/180]; wide_label_space 0.050 [1/20]
- **backbone-optscore** by length: long(>120w) 0.412 [7/17]; medium(30-120w) 0.572 [83/145]; short(<30w) 0.308 [24/78]
- **backbone-optscore** by label_space: 14 0.300 [6/20]; 2 0.610 [61/100]; 3 0.400 [16/40]; 4 0.650 [13/20]; 5 0.200 [4/20]; 60 0.050 [1/20]; 8 0.650 [13/20]
- **majority** by difficulty: adversarial 0.300 [6/20]; ordinal 0.150 [3/20]; standard 0.361 [65/180]; wide_label_space 0.000 [0/20]
- **majority** by length: long(>120w) 0.294 [5/17]; medium(30-120w) 0.276 [40/145]; short(<30w) 0.372 [29/78]
- **majority** by label_space: 14 0.000 [0/20]; 2 0.540 [54/100]; 3 0.425 [17/40]; 4 0.000 [0/20]; 5 0.150 [3/20]; 60 0.000 [0/20]; 8 0.000 [0/20]
- **vega-auto** by difficulty: adversarial 0.250 [5/20]; ordinal 0.650 [13/20]; standard 0.778 [140/180]; wide_label_space 0.450 [9/20]
- **vega-auto** by length: long(>120w) 0.529 [9/17]; medium(30-120w) 0.717 [104/145]; short(<30w) 0.692 [54/78]
- **vega-auto** by label_space: 14 0.850 [17/20]; 2 0.770 [77/100]; 3 0.400 [16/40]; 4 0.900 [18/20]; 5 0.650 [13/20]; 60 0.450 [9/20]; 8 0.850 [17/20]
- **vega-off** by difficulty: adversarial 0.250 [5/20]; ordinal 0.650 [13/20]; standard 0.778 [140/180]; wide_label_space 0.500 [10/20]
- **vega-off** by length: long(>120w) 0.529 [9/17]; medium(30-120w) 0.717 [104/145]; short(<30w) 0.705 [55/78]
- **vega-off** by label_space: 14 0.850 [17/20]; 2 0.770 [77/100]; 3 0.400 [16/40]; 4 0.900 [18/20]; 5 0.650 [13/20]; 60 0.500 [10/20]; 8 0.850 [17/20]
