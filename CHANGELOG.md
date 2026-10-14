# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

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
