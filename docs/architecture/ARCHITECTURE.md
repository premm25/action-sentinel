# Sentinel architecture

This document describes the code as implemented in `action_sentinel/` (version 0.1.0).

## Components and responsibilities

| component | module | responsibility | may it authorize execution? |
|---|---|---|---|
| Contracts | `action_sentinel/contracts/` | Typed, strict (`extra="forbid"`) pydantic models, one per stage: proposal, request, provider result, finding, uncertainty, disagreement, decision, review, grant, execution result, audit event, dataset/report | n/a |
| Validation engine | `action_sentinel/validation/` | Deterministic checks against YAML rule packs: action type, forbidden actions, source and target restrictions, required fields, evidence, rule conditions, risk level, policy conflicts. Fails closed on internal error | no; it can only block or require review |
| Rule grammar | `validation/grammar.py` | Restricted condition language (`all`/`any`/`not`/field-op-value, 16 operators). Plain data, never executed | n/a |
| Providers | `action_sentinel/providers/` | `DecisionProvider` interface: `decide(DecisionRequest) -> ProviderResult`, which never raises. Implementations: `rule_based`, `mock`, `ollama`, `vegaml` | no |
| Provider runner | `providers/runner.py` | `single` / `sequential` / `parallel` strategies, with a hard per-provider timeout enforced by Sentinel | no |
| Registry | `providers/registry.py` | Builds providers from config, validates their config, reports health, accepts new provider types | no |
| Uncertainty | `action_sentinel/uncertainty.py` | Per-provider confidence, calibration status, ambiguity and health, plus an overall status (`confident`/`uncertain`/`unknown`/`abstained`/`failed`) | no |
| Disagreement | `action_sentinel/disagreement.py` | Records category, confidence, model-vs-policy, evidence-vs-model, missing-result and contradictory-finding disagreements. Records them only and never resolves them by voting | no |
| Governance | `governance/policy.py` | A fixed precedence that turns all of the above into one `Outcome` with reason codes | it decides `APPROVE`, but only governance and review issue grants |
| Human review | `governance/review.py` + `service.py` | State machine (PENDING to APPROVED/REJECTED/OVERRIDDEN/EXPIRED/CANCELLED), reviewer levels, self-review ban, required reasons, re-validation | yes, via a grant, only after re-validation |
| Authorization | `execution/authorization.py` | Issues HMAC-signed grants bound to the action hash, idempotency key, policy version and expiry. Handles revocation | issues grants |
| Execution gateway | `execution/executor.py` | The only path to an adapter. Re-checks token, hash, decision/review state, policy version, current hard rules and idempotency | enforces |
| Audit | `action_sentinel/audit.py` | Sanitized, ordered events (`seq`) with correlation id, component version, policy version and reason codes. JSONL export | n/a |
| Storage | `action_sentinel/storage/` | `Store` protocol and SQLite implementation (WAL, compare-and-set on reviews, atomic idempotency claim) | n/a |
| Service | `action_sentinel/service.py` | `Sentinel` wires the components together. Used by the API, CLI, dashboard, demo and lab | n/a |
| Confidence Lab | `action_sentinel/evaluation/` | Datasets, metrics, paired runs over the real pipeline, ablations, markdown reports | n/a |
| Interfaces | `api/app.py`, `cli.py`, `dashboard/app.py` | Thin layers over `Sentinel` | n/a |

## Data flow for one proposal

1. **Intake** (`Sentinel.evaluate`). The `event_id` is looked up first. The same event with the same
   action hash returns the stored decision (`proposal.duplicate`). The same event with a different
   action is rejected (`proposal.conflict`, HTTP 409). Otherwise a correlation id is created and
   `proposal.received` is logged.
2. **Validation**. The rule packs produce findings (`block` / `require_review` / `flag`), a risk
   assessment (base risk from the action type, raised only by `set_risk` rules; conflicting levels
   add a `POLICY_CONFLICT` review finding) and an evidence assessment. Evidence counts as
   supporting only if its stance is `supports`. Unknown stance is never counted as support.
3. **Providers**. The runner calls the configured providers. Each result is normalized: a label
   outside the categories, malformed probabilities, an exception or a timeout all become an
   `error`/`timeout`/`unavailable` result with no label.
4. **Uncertainty and disagreement** are computed and logged.
5. **Governance** applies the precedence below.
6. **Side effects**. `HUMAN_REVIEW` creates a review record. `APPROVE` issues a grant whose token
   is returned once to the caller. `BLOCK`, `ABSTAIN` and `ERROR` issue nothing.
7. **Execution** (`ExecutionGateway.execute(token, action)`). Checks run in order, and the first
   failure rejects: signature and grant exist, not revoked, not expired; action hash matches; the
   decision is not BLOCK/ERROR and either (a) the grant is the decision's own auto-approval grant or
   (b) the grant's review is APPROVED/OVERRIDDEN and issued exactly this grant; the policy version
   is still accepted; current hard rules still pass; then the idempotency key is claimed atomically.
   Only then is the adapter called.

## Governance precedence

```
1. any validation finding with effect=block          -> BLOCK        (nothing overrides it)
2. no provider answered (all error/timeout/unavail.) -> ERROR
3. collect review reasons:
     review findings, risk > max_auto_approve_risk (default: low), evidence not sufficient,
     disagreement with handling=human_review and severity >= medium,
     any provider label not in permissive_labels ("safe"),
     uncertainty status uncertain/unknown
4. review reasons present                            -> HUMAN_REVIEW (+ PROVIDERS_ABSTAINED if they did)
5. providers abstained and no review reasons         -> ABSTAIN
6. otherwise                                         -> APPROVE (ALL_CONDITIONS_MET)
```

A model can only make an outcome *more* restrictive. A high-confidence `safe` label cannot remove a
review reason, and a unanimous vote cannot remove a `block` finding. Abstention and errors never
lead to approval.

## Trust boundaries

1. **Agent to Sentinel** (API). Proposals are untrusted input. They are strictly validated (size,
   depth, identifier patterns, timezone-aware timestamps, unknown fields rejected). An agent key
   can only submit actions whose `agent_id` equals its principal id.
2. **Sentinel to providers**. Provider output is untrusted. It is normalized and checked against
   the question's categories, and it can never grant anything.
3. **Rule packs**. These are reviewed data, loaded with `yaml.safe_load` and parsed into a closed
   grammar. Regular expressions are length-limited and inputs are truncated.
4. **Reviewer to Sentinel**. Reviewer identity comes from the authenticated principal, never from
   the request body. Every approval and override is re-validated against the current policy.
5. **Sentinel to executor**. The executor trusts only a token whose grant still passes every check
   at execution time.

## Failure modes

| failure | behaviour |
|---|---|
| A provider raises, returns garbage or hangs | Normalized to `error` / `timeout` (enforced by the runner); other providers continue |
| All providers fail | `ERROR`, no grant |
| Validation raises internally | Treated as a critical `block` finding (`VALIDATION_INTERNAL_ERROR`) |
| Ollama down or model missing | Health shows unavailable; the provider returns `unavailable`; the lab refuses to run it unless asked |
| VegaML not installed | Health shows unavailable; results are `unavailable`; nothing imports torch |
| Two reviewers act at once | Compare-and-set on review state: one wins, the other gets 409, and any grant issued in the race is revoked |
| Duplicate execution (including concurrent) | Atomic idempotency claim: exactly one `EXECUTED`, the rest `DUPLICATE` |
| Adapter fails mid-execution | `FAILED` is recorded; the key stays claimed so a partial action is not blindly retried |
| Policy changes after approval | The grant is rejected as `POLICY_VERSION_STALE` unless that version is in `accepted_policy_versions`; even then, current hard rules are re-checked |

## Key design decisions

See [`../decisions/ARCHITECTURE_DECISIONS.md`](../decisions/ARCHITECTURE_DECISIONS.md).
