# Sentinel — Project Status

_Last updated 2026-10-14. This file records what was actually done and verified. Anything not
verified is labelled as such._

## Summary

All 13 phases (0–12) are implemented. The test suite passes (**203 passed**, plus 1 optional live-LLM
test that also passed when run separately), statement coverage is **94%**, the offline demo passes
all of its checks, and a live Confidence Lab run against a local LLM was completed. Sentinel is a
working local prototype, **not production-ready** (see Known limitations).

## Phase 0 — Discovery and baseline

- The working directory `C:\Users\sasi\OneDrive\Vega` was **empty** and not a git repository, so
  this is a new project with no existing contracts or tests to preserve. `git init` was run;
  **nothing has been committed**.
- Baseline tests: none existed.
- Environment: Windows 11 Pro; Python 3.14.6 (venv `.venv/`); pip 26.2.1; Ollama 0.40.3 with
  `llama3.1:8b`, **CPU only**; VegaML not installed.
- The VegaML repository (`NandhaKishorM/vegaml`, commit `7803fcf`, package 0.9.0) was cloned into a
  scratch directory **outside** the project for API inspection only. None of its code was copied.

## Phases 1–12: what was built and how it was verified

| phase | delivered | verified by |
|---|---|---|
| 1 Contracts | `sentinel/contracts/`: strict pydantic models per stage, `DecisionProvider` result contract with explicit unknowns | `tests/unit/test_contracts.py` (17) |
| 2 Validation | `sentinel/validation/`: restricted grammar, versioned YAML packs, static pack validator (duplicates, conflicts, unsafe input), engine (types, forbidden, source/target, required fields, evidence, rules, risk, runtime conflicts, fail-closed) | `tests/unit/test_validation.py` (53) |
| 3 Providers | rule-based, mock (scripted), Ollama (logprob-based probabilities), VegaML adapter, registry, runner (single/sequential/parallel with enforced timeout) | `tests/unit/test_providers.py` (29 + 1 optional live) |
| 4 Uncertainty and disagreement | `uncertainty.py`, `disagreement.py` | `tests/unit/test_uncertainty_disagreement.py` |
| 5 Governance and review | `governance/policy.py` (fixed precedence plus ablation flags), `governance/review.py` + service (state machine, levels, self-review ban, reasons, expiry, cancel, override with full re-validation) | `tests/unit/test_governance.py`, `tests/integration/test_review_lifecycle.py` |
| 6 Authorization and execution | HMAC grants bound to action hash, policy version and expiry; revocation; `ExecutionGateway`; `MockExecutionAdapter` | `tests/integration/test_execution_boundary.py`, `tests/integration/test_critical_paths.py` |
| 7 Audit and storage | `audit.py` (sanitize, JSONL export), `storage/` (`Store` protocol, SQLite with WAL, CAS, atomic idempotency) | `tests/integration/test_audit_storage.py` |
| 8 Confidence Lab | dataset generator, loader, metrics, paired lab runner with caching, suite and ablations, markdown reports, calibration profile | `tests/unit/test_metrics.py`, `tests/integration/test_confidence_lab.py`; live runs below |
| 9 API and CLI | `api/app.py` (FastAPI, key auth, roles, pagination), `cli.py` | `tests/api/test_api.py` (in-process and a **real uvicorn subprocess**), `tests/end_to_end` (CLI subprocesses), in-process CLI test |
| 10 Dashboard | `dashboard/app.py` (6 tabs, real store, empty states, keyed review actions) | `tests/api/test_dashboard.py` (Streamlit AppTest, empty and populated DB, approval through the UI) |
| 11 Adversarial and E2E | adversarial cases throughout, `demo.py` | `tests/end_to_end/test_end_to_end.py`; `sentinel demo` → ALL CHECKS PASSED |
| 12 Documentation | README and the 9 documents listed in it | Manual review against the code |

## Verified results

- **Tests:** unit 140, integration 47, API and dashboard 12, end-to-end 4, for **203 passed**. The
  full suite passed on three consecutive runs after the last fix. The optional `-m ollama` live test
  passed (about 30 s).
- **Coverage:** 94% of statements in `sentinel/` (pytest-cov). Critical modules:
  `governance/policy.py` 100%, `execution/executor.py` 100%, `execution/authorization.py` 98%,
  `governance/review.py` 95%, `service.py` 95%, `uncertainty.py` 100%, `disagreement.py` 100%.
  `cli.py` is 78% because most CLI paths are exercised via subprocesses that coverage does not trace.
- **Installation:** a fresh, non-editable `pip install ".[dev,dashboard]"` into a new venv ran
  `sentinel demo` successfully and passed the unit tests.
- **Lint:** `ruff check` (E, F, W, B) is clean.
- **Confidence Lab:** see `docs/evaluation/RESULTS.md`. Headline on the test split (n = 44,
  synthetic): llama3.1:8b alone was 0.409 accurate with ECE 0.394 and blocked 0/10 hard violations
  in naive mode. Full Sentinel (rules + LLM) blocked 10/10 with 0/34 false approvals, but
  auto-approved 0/10 safe cases.

## Bugs found and fixed during implementation (regression tests added)

1. **Double execution under concurrency (serious).** The SQLite store read cursors outside its lock
   on a shared connection. Another thread's COMMIT could fail and leave a transaction open, which
   then rolled back an idempotency claim, so 2–3 of 8 concurrent requests executed (about 1 run in 8).
   Fix: every statement and fetch is lock-scoped, and rollback happens only if a transaction is open.
   Regression: `test_regression_concurrent_execution_stress` (25 trials × 8 threads) and 120
   instrumented trials, all clean.
2. **All-abstain routed to review instead of ABSTAIN.** `TOO_FEW_PROVIDER_RESULTS` fired with zero
   results. Fixed: it only fires when some providers answered, but fewer than the minimum.
3. **Mock scripts would have leaked into LLM prompts** (scripts were in `action.context`). Moved out
   of the proposal; a test asserts no leakage.
4. **FastAPI dependency aliases broken by `from __future__ import annotations`** (every endpoint
   returned 422). Removed from the API module.
5. **Dashboard cached one service across databases.** The DB path is now part of the cache key.
6. **Non-editable install could not find `config/` and `examples/`.** Paths now resolve from
   `$SENTINEL_CONFIG`, then `./config`, then the source checkout.
7. Earlier: an override event-id formatting bug and a test that expected the wrong error. The test
   was corrected to assert the actual, correct BLOCK; nothing was weakened.

## Optional integrations

| integration | status |
|---|---|
| Ollama (`llama3.1:8b`) | **Verified live**: health check, provider test, full Confidence Lab suite on 44 test examples. Logprobs are supported by Ollama 0.40.3 and used for probabilities |
| VegaML | **Implemented, not run live.** The adapter was written against the inspected v0.9.0 API (`vegaml.load`, `Vega.decide(..., mode="engine")`, answer fields `choice/probs/confidence/prediction_set/abstain/unbound`) and tested against a stub of that API. The package was not installed: it needs torch, transformers and about 1.8 GB of weights. The provider reports `unavailable` and fails safe |

## Known limitations

- SQLite, single node; audit records are not tamper-evident.
- Dev API keys are public; there is no TLS, rate limiting or key rotation; the signing key is
  ephemeral unless `SENTINEL_SIGNING_KEY` is set.
- The dataset is synthetic and small (86 examples; 44 test). Expected outcomes come from the same
  reference policy the rules enforce, so the rules-only score is circular.
- The LLM was measured with one prompt and one model, on CPU (about 12 s per decision). No prompt
  study was done, deliberately, to avoid tuning on the test split.
- Calibration profiles can be measured (`sentinel lab calibrate`) but are not applied
  automatically, and none were applied in the reported runs.
- Hung provider threads are abandoned, not killed. Rule regexes have no timeout.
- The executor is a mock only.

## Next steps (highest value first)

1. **Make the model layer useful.** Run a dev-split prompt and model study (and VegaML once
   installed), then measure calibration on the calibration split and apply profiles, so a
   statistical provider can help auto-approve safe cases instead of only escalating them.
2. **Production hardening.** Persistent signing keys from a KMS, OIDC auth, TLS, a Postgres
   `Store`, hash-chained audit, and rate limits.
3. **Better evaluation data.** A larger, independently labelled dataset (ideally sampled from real
   agent traffic, with multiple annotators) so governance metrics stop being circular and
   intervals tighten.

## How to resume

```bash
.venv/Scripts/activate
pytest                                   # 203 tests
sentinel demo                            # offline end-to-end check
sentinel lab suite --model ollama --second ""   # live LLM evaluation (about 30–40 min on CPU)
```
