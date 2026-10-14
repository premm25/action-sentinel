# Action Sentinel

**A safety checkpoint between AI agents and the real world.** Before an agent restarts a server,
refunds a payment or deletes a database, Sentinel decides whether it may, and the action can only
run with an authorization token issued for that exact action.

```
agent proposes action ──► Sentinel ──► APPROVE  → signed token → executor runs it (exactly once)
                                   ├─► HUMAN_REVIEW → a person approves, rejects or edits it
                                   ├─► BLOCK   (policy forbids it, whatever any model says)
                                   ├─► ABSTAIN (not enough confidence to recommend anything)
                                   └─► ERROR   (evaluation failed; never treated as approval)
```

## Why

Agents built on LLMs are increasingly given real tools. The usual safeguard is a model judging
whether an action is safe, but models are confidently wrong, and their confidence is rarely
calibrated. Sentinel is built on a few non-negotiable rules:

- **Deterministic policy wins.** A rule that blocks an action can't be outvoted, out-scored or
  overridden by a model.
- **Models can only escalate.** High confidence never authorizes anything on its own.
- **Uncertainty is explicit.** Missing confidence, unknown calibration, abstention and provider
  failure are first-class states, not defaults.
- **Humans are in the loop, but checked too.** Reviewer authority levels, no self-review, and every
  approval or edit is re-validated against the current policy.
- **Authorization is bound to the exact payload.** Change one field and the token is useless.
  Replays execute once.
- **Everything is auditable**, and the pipeline is measurable with a built-in evaluation lab.

## 30-second example

```bash
pip install -e ".[dev,dashboard]"   # from a clone of this repository
action-sentinel demo                       # offline end-to-end walkthrough, no model or API key needed
```

```python
from action_sentinel.config import load_settings
from action_sentinel.contracts import ActionProposal
from action_sentinel.service import Sentinel

sentinel = Sentinel(load_settings())          # config/sentinel.yaml: rule pack + rule-based provider

action = ActionProposal.model_validate({
    "event_id": "evt-42", "task_id": "t-1", "agent_id": "ops-agent", "agent_version": "1.0",
    "source_system": "infra-agent", "action_type": "delete_database",
    "target_resource": "prod/orders", "action_payload": {"database": "orders"},
    "evidence_refs": [], "proposed_at": "2026-10-14T10:00:00+00:00", "idempotency_key": "evt-42",
})
result = sentinel.evaluate(action)
print(result.decision.status.value, result.decision.reason_codes)
# BLOCK ['HARD_POLICY_VIOLATION', 'EVIDENCE_INSUFFICIENT', 'RULE_PROD_DATABASE_DELETE']
```

An approved action returns `result.authorization.token`. The executor accepts only that token
together with the identical action: `sentinel.execute(token, action)`.

## How it differs from related tools

| | Sentinel | LLM output guardrails | general policy engines |
|---|---|---|---|
| Guards | **actions** (tool calls) before execution | model text / structured output | any request, given policy input |
| Model-based judgement | optional, pluggable, can only escalate | often central | none |
| Uncertainty, abstention, disagreement | explicit outcomes | varies | n/a |
| Human review workflow with re-validation | built in | generally not | not built in |
| Execution token bound to the exact payload, idempotent | built in | no | no |
| Evaluation of the governance pipeline | built in (Confidence Lab) | varies | no |

Sentinel complements these tools rather than replacing them: a policy engine can feed it rules,
and an output guardrail can still check what the agent says.

> **Status:** working prototype, 0.1.0. 203 automated tests, 94% coverage, offline demo. It is
> **not production-ready** yet; see [Known limitations](#known-limitations) and the
> [roadmap](#roadmap). Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Architecture

```
ActionProposal ─► ValidationEngine (rule packs) ──────────────┐  hard violation ⇒ BLOCK, always
               └► ProviderRunner ─► rules | ollama | vegaml | mock
                         │ (single / sequential / parallel, enforced timeout)
                         ▼
          UncertaintyEvaluator + DisagreementDetector
                         ▼
               GovernanceEngine (fixed precedence) ─► GovernanceDecision
                    │ APPROVE                    │ HUMAN_REVIEW
                    ▼                            ▼
              AuthorizationGrant ◄── ReviewManager (approve / reject / override → re-validate)
                    ▼
        ExecutionGateway (re-checks everything) ─► MockExecutionAdapter
   every step ─► AuditLogger ─► SQLiteStore ─► sanitized JSONL export
```

Details: [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md).

## Prerequisites

- Python 3.10 or newer. Developed and tested on 3.14.6 on Windows 11.
- Optional: [Ollama](https://ollama.com) with a pulled model (default `llama3.1:8b`) for the LLM provider.
- Optional: `pip install vegaml` (pulls torch and transformers, about 1.8 GB of weights on first use) for the VegaML provider.

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,dashboard]"
```

## Configuration

`config/sentinel.yaml` holds the rule packs, providers, active providers, evaluation strategy,
governance thresholds, reviewers and API principals. Environment overrides are listed in
[`.env.example`](.env.example): `SENTINEL_CONFIG`, `SENTINEL_DB_PATH`, `SENTINEL_ENV` and
`SENTINEL_SIGNING_KEY`.

By default only the deterministic `rules` provider is active, so evaluation works without any model.
To also consult the local LLM, set `active_providers: [rules, ollama]`, or pass
`--providers rules,ollama` on the CLI.

**Development API keys.** These are public, intended for local use only, and stored in the config as SHA-256 hashes.

| key | principal | roles |
|---|---|---|
| `dev-agent-key` | ops-agent | agent |
| `dev-operator-key` | olivia | reviewer (operator), auditor |
| `dev-senior-key` | sam | reviewer (senior), auditor, evaluator |
| `dev-admin-key` | ada | admin |
| `dev-auditor-key` | auditor | auditor |

## Quick start

```bash
action-sentinel validate-rules                                 # check the rule packs
action-sentinel providers                                      # provider health
action-sentinel evaluate examples/sample_actions/actions.jsonl -v
action-sentinel reviews list
action-sentinel reviews approve <review_id> --reviewer olivia --reason "verified"
action-sentinel demo                                           # full offline end-to-end demonstration
```

`python -m action_sentinel ...` works the same as `action-sentinel ...`.

## API

```bash
action-sentinel serve --port 8000          # interactive docs at http://127.0.0.1:8000/docs
curl -s -X POST http://127.0.0.1:8000/decisions/evaluate -H "X-API-Key: dev-agent-key" \
     -H "Content-Type: application/json" -d @examples/sample_actions/one.json
```

Endpoints:
- `GET /health`, `GET /providers`
- `POST /decisions/evaluate`, `GET /decisions`, `GET /decisions/{id}`, `GET /decisions/{id}/events`
- `GET /reviews/pending`, `GET /reviews/{id}`, `POST /reviews/{id}/approve|reject|override`
- `POST /executions`, `POST /authorizations/{id}/revoke`
- `POST /evaluations/run`, `GET /evaluations`, `GET /evaluations/{run_id}`, `GET /evaluations/{run_id}/report`

## Dashboard

```bash
action-sentinel dashboard                  # http://127.0.0.1:8501
action-sentinel demo --db var/sentinel.db  # optional: populate the default database first
```

The dashboard has six tabs: Overview, Decision Explorer, Confidence Lab, Human Review, Audit
Explorer and Configuration. It reads the same SQLite database as the API and CLI. Review actions
require the reviewer's API key.

## Testing

```bash
pytest                                  # 203 tests, offline (optional live tests are deselected)
pytest --cov=action_sentinel                   # with coverage
pytest -m ollama                        # optional: live test against a running Ollama
```

## Evaluation (Confidence Lab)

```bash
python scripts/build_dataset.py         # regenerate the synthetic dataset (deterministic)
action-sentinel lab suite                      # offline: rules + simulated providers, test split
action-sentinel lab suite --model ollama --second ""     # live local LLM (slow on CPU)
action-sentinel lab run --providers rules,ollama --splits test
action-sentinel lab calibrate --provider ollama          # ECE on the calibration split
action-sentinel lab show [run_id]   /   action-sentinel lab export <run_id...> --out report.md
```

Measured results and their limits are in [`docs/evaluation/RESULTS.md`](docs/evaluation/RESULTS.md).
The method is in [`docs/evaluation/METHODOLOGY.md`](docs/evaluation/METHODOLOGY.md).

## Demo

```bash
action-sentinel demo
```

The demo runs 12 steps offline and checks the expected result of each one. It exits non-zero if
any check fails. See [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md).

## Known limitations

- Single-process SQLite storage, with no high availability, replication or retention management.
- The authentication in the dev configuration uses publicly known keys. There is no TLS, OAuth,
  rate limiting or key rotation.
- If `SENTINEL_SIGNING_KEY` is unset, each process generates its own key, so a token issued by the
  API can't be executed through the CLI.
- The Confidence Lab dataset is synthetic (86 examples, 44 in the test split). Its expected outcomes
  were written against the same reference policy that the rules enforce, so the rules-only score
  is circular.
- The local LLM (`llama3.1:8b`, CPU) was measured at 0.41 accuracy on the test split, and it never
  chose `unsafe`. On its own it is not a useful classifier for this task as prompted.
- The VegaML adapter is implemented against the inspected v0.9.0 API and tested with a stub. It
  was **not run live** because the package isn't installed in this environment.
- The mock executor changes nothing real. A real adapter would also need compensation, dry-run and
  scoping controls.

## Roadmap

1. Python SDK (`@guard` decorator) and an MCP proxy, so existing agents can be protected without rewiring.
2. PyPI package and Docker image.
3. A better model layer: a dev-split prompt and model study, automatic calibration profiles, live VegaML results.
4. A public, independently labelled benchmark of risky agent actions, plus a reusable policy library.
5. Production hardening: Postgres storage, managed signing keys, OIDC, tamper-evident audit, review notifications.

## Documents

- [Architecture](docs/architecture/ARCHITECTURE.md)
- [Threat model](docs/architecture/THREAT_MODEL.md)
- [Architecture decisions](docs/decisions/ARCHITECTURE_DECISIONS.md)
- [Evaluation methodology](docs/evaluation/METHODOLOGY.md)
- [Evaluation results](docs/evaluation/RESULTS.md)
- [Security](docs/security/SECURITY.md)
- [Runbook](docs/operations/RUNBOOK.md)
- [Demo guide](docs/DEMO_GUIDE.md)
- [Contributing](CONTRIBUTING.md) · [Security policy](SECURITY.md) · [Changelog](CHANGELOG.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
