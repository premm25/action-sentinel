# Contributing to Sentinel

Thanks for helping. Sentinel guards what AI agents are allowed to do, so correctness of the
governance and execution paths matters more than features.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,dashboard]"
pytest                                # all tests run offline
action-sentinel demo                         # end-to-end check
```

## Before opening a pull request

- `ruff check action_sentinel tests scripts` is clean.
- `pytest` passes. New behaviour comes with tests. Bug fixes come with a regression test that fails
  before the fix.
- If you touch `action_sentinel/governance/`, `action_sentinel/execution/` or `action_sentinel/storage/`, add tests for
  the failure paths, not only the happy path. Those modules are the safety boundary.
- Never weaken or delete a test to make the suite pass. If a test was wrong, explain why in the PR.
- If you change `scripts/build_dataset.py`, regenerate the dataset and bump `VERSION`. Results
  from different dataset versions must not be compared.
- Don't commit secrets, real customer data or production identifiers. Examples and datasets must be
  synthetic and labelled as such.

## Design rules (please keep them)

1. Hard policy violations always win. No provider, vote or reviewer can override a `block`.
2. Model confidence can only make an outcome stricter. It never authorizes anything on its own.
3. Missing data (confidence, calibration, evidence) is explicit `None`/`unknown`, never a default value.
4. Every state change writes an audit event.
5. Evaluation numbers in docs must come from a real, reproducible run.

## Adding things

- **A decision provider:** subclass `action_sentinel.providers.DecisionProvider`, implement `_decide`
  returning a `RawAnswer`, and register a factory with `ProviderRegistry.register_type`. See
  `providers/mock.py` for the smallest example.
- **A rule pack:** YAML in the grammar documented in `action_sentinel/validation/grammar.py`. Validate it
  with `action-sentinel validate-rules path/to/pack.yaml`.

## Reporting security issues

Please don't open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).
