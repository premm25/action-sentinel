# Confidence Lab results

All numbers below were produced by actual runs on 2026-10-14. Environment: Windows 11, Python 3.14.6,
Sentinel 0.1.0, policy `gov-1.0.0+core@1.0.0`, dataset `sentinel-synthetic-v1` (sha256
`3eafcbd3f6bd…`), **test split, n = 44**. The full generated reports, with per-example records and
95% intervals, are:

- `reports/llm_test/REPORT.md` and `*.json`: live `llama3.1:8b` through Ollama 0.40.3, CPU only
- `reports/offline_test/REPORT.md` and `*.json`: rules plus the two simulated providers

To reproduce them:
`sentinel lab suite --model ollama --second "" --out reports/llm_test` and
`sentinel lab suite --out reports/offline_test`.

Read the [methodology](METHODOLOGY.md) first, especially the threats to validity.

## 1. Live local LLM (llama3.1:8b, Ollama, CPU)

### Model prediction quality (test split, n = 44)

| provider | accuracy | macro F1 | ECE | Brier | log loss | AURC | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| rules (deterministic) | 0.727 [32/44] | 0.706 | n/a (no confidence) | n/a | n/a | n/a | 0.2 ms |
| llama3.1:8b | **0.409 [18/44]** | 0.290 | **0.394** | 0.839 | 1.373 | 0.453 | 12.1 s |

Confusion matrix for llama3.1:8b (rows are expected, columns are predicted, in the order safe,
needs_review, unsafe):

```
safe          0  10   0
needs_review  0  18   2
unsafe        0  14   0
```

**Finding.** With this prompt the model answered `needs_review` on 42 of 44 examples. It never chose
`safe`, and it never chose `unsafe` for any of the 14 actions that should be blocked (destructive
production deletes, SSH open to the internet, forbidden actions). It was also overconfident: mean
confidence 0.80 against accuracy 0.41. The 23 answers with confidence 0.8–0.9 were right 43% of the
time. The p99 latency (213 s) includes the cold model load; p95 is 17.3 s.

### Governance outcome (same examples, same model outputs)

| metric | ollama naive | rules only | validation + ollama, no review | full Sentinel (rules + ollama) |
|---|---:|---:|---:|---:|
| false approvals ↓ | 0/34 | 0/34 | 0/34 | **0/34** |
| hard violations blocked | **0/10** | 10/10 | 10/10 | **10/10** |
| expected-BLOCK cases blocked | 0/14 | 14/14 | 14/14 | 14/14 |
| false blocks ↓ | 2/30 | 0/30 | **30/30** | 0/30 |
| safe cases auto-approved | 0/10 | 10/10 | 0/10 | 0/10 |
| human escalation rate | 0.955 | 0.455 | 0.000 | 0.682 |
| outcome accuracy | 0.409 | 1.000* | 0.318 | 0.773 |

\* circular: the expected outcomes were written against this rule pack. See the methodology.

Ablations of full Sentinel (rules + ollama): removing validation dropped hard-violation blocking
from 10/10 to 0/10 and sent all 44 cases to review. Removing review produced 30/30 false blocks.
Removing disagreement or uncertainty gating changed little, because this model's outputs were
uniformly `needs_review`.

**What this shows.** On this dataset the deterministic layer did all the protective work, and the
LLM added no useful signal. Wrapping the model in Sentinel turned a model that missed every hard
violation into a pipeline that blocked all of them with zero false approvals. The price was that
none of the 10 safe cases was auto-approved, because the model always disagreed with the rules
(`PROVIDER_NOT_PERMISSIVE`). Adding this LLM to rules-only made outcomes *worse* in automation
terms: auto-approval coverage fell from 10/10 to 0/10.

## 2. Offline pipeline check (rules + simulated providers)

`mock-a` and `mock-b` are scripted simulations (see the methodology). These rows test that the
governance machinery behaves as designed under confident errors, disagreement, abstention and
timeouts. They are **not** evidence about any model.

| metric | mock-a naive | rules only | validation + mock-a, no review | rules+mocks, no uncertainty/review | full Sentinel | full minus validation |
|---|---:|---:|---:|---:|---:|---:|
| false approvals ↓ | 5/34 | 0/34 | 1/34 | 0/34 | **0/34** | 0/34 |
| hard violations blocked | 4/10 | 10/10 | 10/10 | 10/10 | **10/10** | 0/10 |
| false blocks ↓ | 3/30 | 0/30 | 21/30 | 24/30 | **0/30** | 0/30 |
| safe cases auto-approved | 8/10 | 10/10 | 8/10 | 6/10 | 1/10 | 1/10 |
| human escalation rate | 0.455 | 0.455 | 0.000 | 0.000 | 0.659 | 0.977 |

The simulated `mock-a` had 7 errors at confidence ≥ 0.9, including rating production database
deletes `safe` at 0.96. In naive mode these became 5 false approvals. Under full Sentinel every one
was blocked by policy (`model_vs_policy` disagreement, `policy_precedence`).

## What these results do not show

- Nothing about real agent traffic: the data is synthetic, and the expected outcomes come from one
  author and one reference policy.
- Nothing about LLMs in general: one 8B model, one prompt, CPU only. Prompt or model changes were
  deliberately not tried against the test split. A dev-split prompt study is future work.
- Nothing about VegaML: it was not installed or run (see PROJECT_STATUS.md).
- No statistical significance claims between configurations that differ by a few examples. With
  n = 44 the 95% Wilson interval for 0/34 false approvals is 0.00–0.10.
- Calibration profiles were not applied in these runs, so every statistical provider ran with
  calibration status `unknown`.
