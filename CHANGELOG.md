# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- VegaML-first operation: `config/sentinel-vegaml.yaml` (rules and the VegaML head side by side, warmed up
  at startup), `action-sentinel vega fit` (fit the head on your labelled actions, never the test split, and
  save it), `ttt_head_path` to serve a saved head, `Sentinel.warm_up()` and the `warm_up` setting. A saved
  head is refused for a different question, framing, checkpoint or engine. Guide: `docs/VEGAML.md`.
- `action_sentinel.guard`: a `@guard.tool` decorator for agent tool functions, with shadow mode
  (assess and record, never block) and enforce mode (run only through the execution gateway).
- AgentHarm studies: classifier baselines for the VegaML head, and AgentHarm through full Sentinel
  (`examples/rule_packs/agent_tasks.yaml`).

### Changed
- Project renamed to **Action Sentinel**: distribution `action-sentinel`, import package `action_sentinel`,
  command `action-sentinel` (`agent-sentinel` and the import name `sentinel` are already taken on PyPI).

## [0.1.0] - 2026-10-14

### Added
- Typed contracts for proposals, provider results, validation, uncertainty, disagreement, governance
  decisions, reviews, authorization grants, execution results, audit events and evaluation reports.
- Deterministic validation engine with versioned YAML rule packs, a restricted condition grammar and
  a static rule-pack validator.
- Decision providers: rule-based, scripted mock, Ollama (logprob-based probabilities), VegaML adapter
  (not run live); a provider registry; single, sequential and parallel strategies with enforced timeouts.
- Uncertainty assessment and disagreement detection.
- Governance engine with fixed precedence; human-review state machine with reviewer levels, re-validated
  approvals and overrides.
- HMAC-signed authorization bound to the exact action and policy version; execution gateway with
  atomic idempotency; mock executor.
- SQLite storage, sanitized audit trail and JSONL export.
- Confidence Lab: synthetic dataset (86 examples), metrics, paired runs, ablations, reports.
- FastAPI service, CLI, Streamlit dashboard, offline end-to-end demo.

### Fixed (during initial development)
- Concurrent executions of one approved action could run more than once because of a shared SQLite
  connection read outside its lock.
