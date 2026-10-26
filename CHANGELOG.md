# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- MCP support through the sluice proxy, shadow mode: `config/sentinel-mcp.yaml` with
  `examples/rule_packs/mcp.yaml` (action type `mcp_tool_call`) and `examples/rule_packs/mcp_classifier.yaml`,
  and a `dev-mcp-key` principal for the proxy. Guide: `docs/MCP.md`.
- MCP enforce mode: resubmitting a held proposal (same event id and action) after a reviewer approved it
  returns that approval's authorization token, so a proxy can run the call when the agent retries.
  `POST /decisions/evaluate` (and `RemoteEvaluation`) now report `review_state`; a pending review past its
  deadline is expired when resubmitted.
- `action-sentinel vega dataset --from calls.jsonl --out ds.jsonl`: labelled MCP calls from
  `sluice sentinel export` as a Confidence Lab dataset (stable hash splits, verdicts kept out), for `vega fit`.

### Changed
- `reviews list` also shows what each held action is (type, target, payload).
- `--db` paths are resolved from the directory you run the command in, not the config file's directory.

## [0.2.1] - 2026-10-26

### Added
- A demo recording in the README (`docs/assets/demo.gif`): a real run of `examples/guard_demo.py`, in which a
  safe call runs, a database delete is blocked and a scale-up waits for a human. `scripts/make_demo_gif.py`
  regenerates it; `action-sentinel init` copies the demo, and a test keeps it working.

## [0.2.0] - 2026-10-25

First release on PyPI.

### Added
- The package ships its settings profiles, rule packs, synthetic dataset and sample actions, so a plain
  `pip install` works without a clone. Relative paths in settings fall back to the shipped files;
  `action-sentinel init` copies them into a project to edit. CI installs the built wheel in a clean
  environment and runs it outside the repository.
- `@guard.tool` supports `async def` tools (`await guard.aresume(...)` for held ones); Sentinel's work runs
  in a worker thread so the event loop isn't blocked.
- Remote mode: `action_sentinel.remote.RemoteSentinel` lets a guard use Sentinel running as a separate
  service. New endpoints: `POST /decisions/assess`, `POST /shadow-records`, `POST /executions/begin` and
  `POST /executions/{id}/finish`; `POST /decisions/evaluate` accepts a `question`.
- The execution gateway has `begin` and `finish` for callers that run the action themselves. Executions
  record `started_by`, and a new status `STARTED` covers a cleared action that hasn't reported back.
- VegaML-first operation: `config/sentinel-vegaml.yaml` (rules and the VegaML head side by side, warmed up
  at startup), `action-sentinel vega fit` (fit the head on your labelled actions, never the test split, and
  save it), `ttt_head_path` to serve a saved head, `Sentinel.warm_up()` and the `warm_up` setting. A saved
  head is refused for a different question, framing, checkpoint or engine. Guide: `docs/VEGAML.md`.
- `action_sentinel.guard`: a `@guard.tool` decorator for agent tool functions, with shadow mode
  (assess and record, never block) and enforce mode (run only through the execution gateway).
- AgentHarm studies: classifier baselines for the VegaML head, and AgentHarm through full Sentinel
  (`examples/rule_packs/agent_tasks.yaml`).

### Changed
- `Guard` no longer needs `Sentinel(adapter=GuardAdapter())`. It runs the tool itself once the gateway's
  `begin` clears it; `GuardAdapter` remains for existing code but refuses direct `Sentinel.execute`.
- `ExecutionGateway.execute` now records the execution as `STARTED` before the adapter runs.
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
