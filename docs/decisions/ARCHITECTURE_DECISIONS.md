# Architecture decisions

Short records of each significant decision: context, decision and consequence.

### ADR-1: Separate contracts per stage
A single overloaded "decision" object would blur proposal, validation, provider output, governance
outcome, authorization and execution. Each stage therefore has its own strict pydantic model, and an
`ExecutionResult` can only be produced by the gateway. The cost is more types; the benefit is that
"is this authorized?" can't be answered by reading the wrong field.

### ADR-2: Deterministic precedence, not scoring or voting
Governance is a fixed ordered list of checks (see ARCHITECTURE.md), not a weighted score. Models can
only add review reasons. This makes outcomes explainable from reason codes and makes "confidence
cannot authorize" structurally true rather than a matter of tuning.

### ADR-3: Restricted rule grammar instead of an expression language
Rules are YAML data in a closed grammar (`all`/`any`/`not`/field-op-value). This rules out code
injection and makes static checks possible: duplicate ids, conflicting `set_risk` levels on
identical conditions, sources permitting forbidden actions, and cross-pack conflicts. Rules that
need computed values get them from the small `derived` context instead.

### ADR-4: Risk can only be raised by rules
`set_risk` rules act as floors and the highest wins. Two matching rules with different levels add a
`POLICY_CONFLICT` review finding at runtime. A rule can't lower the action type's base risk, so a
mistaken rule can't silently relax policy.

### ADR-5: Evidence stance defaults to unknown
Evidence counts toward sufficiency only if it is marked `supports`. Refs of unknown stance produce
`unverified`, which triggers review. This implements "missing evidence is not positive evidence".

### ADR-6: Confidence values are never compared across providers
The uncertainty evaluator checks each provider against the thresholds separately and records the
calibration status per provider (`measured` / `unknown` / `not_applicable`). The disagreement
detector reports a confidence gap only as informational. Deterministic providers report no
confidence at all, rather than an invented 1.0.

### ADR-7: Ollama confidence from token logprobs
The LLM answers with one option letter. The answer distribution is read from that token's
`top_logprobs` and renormalized over the option letters; the uncovered mass is recorded as
`option_mass`. This is a model probability, unlike asking the model to state a number. When
logprobs are unavailable the provider returns the label with **no** confidence. This was verified
against Ollama 0.40.3 during development.

### ADR-8: VegaML is an optional, lazily imported adapter
It was implemented only against the inspected public API of vegaml 0.9.0 (`load`, `decide(...,
mode="engine")` and its answer fields). `mode="engine"` is requested so VegaML's own abstain flag and
conformal set are kept; `unbound` and `abstain` both map to an abstained result. VegaML's
calibration was fitted on its authors' data, so it is reported as `unknown` for Sentinel. Nothing
imports torch unless the provider is used.

### ADR-9: Authorization bound to the exact action and policy version
A grant stores the SHA-256 of the canonical action (type, payload, target, idempotency key, agent,
source), and the token is an HMAC over that and the expiry. Overrides therefore always produce a
new grant for the new payload. Policy changes invalidate grants unless the operator lists the
previous version in `accepted_policy_versions`, and hard rules are re-checked at execution time
regardless.

### ADR-10: SQLite with atomic primitives
SQLite in WAL mode is enough for a local prototype and lets the API, CLI and dashboard share one
file. The correctness-critical operations are single SQL statements: idempotency claim (`INSERT`
with a primary key), review compare-and-set (`UPDATE ... WHERE state=?`) and event append (one
transaction). The `Store` protocol keeps the backend replaceable.

### ADR-11: Simple concurrency
Providers run in a thread pool only in `parallel` mode. The first implementation is deterministic
and synchronous everywhere else. Timeouts are enforced by Sentinel, not by trusting providers.

### ADR-12: Confidence Lab runs the production code path, paired
The lab calls `Sentinel.assess` with an isolated in-memory store and caches provider results per
example, so every configuration in a suite sees identical provider outputs and the LLM is called
once per example. Ablations are flags on the same governance engine, not separate
implementations. The naive baseline maps the first provider's label straight to an outcome.

### ADR-13: Mock scripts live outside the proposal
Scripted mock outputs are passed to the mock provider by event id, never inside the proposal, so a
real provider evaluated alongside the mock can't see the "answers". A test asserts this for the
dataset.

### ADR-14: Abstention is a terminal outcome unless review is otherwise required
If providers abstain and no other review reason exists, the result is `ABSTAIN`; the caller may
resubmit with more evidence. If review is required anyway, the case goes to `HUMAN_REVIEW` with
`PROVIDERS_ABSTAINED` added. This keeps the five outcomes distinct, as specified.
