# Threat model

Scope: the Sentinel prototype as implemented, deployed locally. The assets are the integrity of the
execution boundary (only authorized actions run), the integrity of the audit trail, and the
confidentiality of payload data in logs.

## Actors

| actor | capabilities assumed |
|---|---|
| Honest agent | Submits proposals with its own API key |
| Compromised or misaligned agent | Submits arbitrary, adversarial or replayed proposals and tokens; may try to act as another agent |
| Malicious or mistaken reviewer | Holds a reviewer key with some approval level |
| Faulty or manipulated model | Returns wrong, overconfident, malformed or slow answers, and may be prompt-injected through proposal text |
| Local operator / admin | Controls config, rule packs and the signing key (trusted) |

## Threats and mitigations

| # | threat | mitigation in code | test |
|---|---|---|---|
| T1 | A model persuaded (e.g. by prompt injection in the payload) to rate a destructive action safe | Hard rules take precedence; a model can only escalate | `test_hard_violation_blocks_despite_confident_unanimous_models`, demo step 3 |
| T2 | Majority of providers out-voting policy | No voting; `block` findings are terminal | `test_majority_cannot_outvote_policy` |
| T3 | Provider failure or abstention treated as approval | `ERROR`/`ABSTAIN` are never executable | `test_non_approved_decisions_cannot_execute` |
| T4 | Malformed provider output | Normalization turns it into an error result | `test_mock_behaviours`, `test_malformed_provider_output_cannot_approve` |
| T5 | Executing a different payload than the one approved | Grant bound to SHA-256 of the canonical action | `test_changed_payload_invalidates_authorization` |
| T6 | Token forgery or tampering | HMAC-SHA256 over grant id, hash, policy version, expiry and key; grant row looked up | `test_expired_revoked_forged_and_unknown_tokens` |
| T7 | Replay / double execution | Atomic idempotency claim in SQLite | `test_concurrent_duplicate_execution_runs_once` |
| T8 | Using a stale approval after a policy change | Policy version check plus re-validation of hard rules at execution time | `test_policy_change_invalidates_stale_authorization`, `test_safety_constraints_rechecked_at_execution` |
| T9 | Reviewer approving beyond their authority, or approving their own action | Level check, self-review ban, identity from the principal | `test_unauthorized_reviewers_rejected_and_recorded`, `test_self_review_and_inactive_reviewer_rejected` |
| T10 | A reviewer override sneaking in a dangerous change | The override runs the full pipeline; BLOCK/ERROR rejects it; a higher risk level needs a higher reviewer level; identity fields can't be changed | `test_override_failing_revalidation_is_rejected`, `test_override_raising_risk_needs_higher_authority`, `test_override_field_restrictions` |
| T11 | A grant minted for an unfinished or rejected review | The gateway requires review state APPROVED/OVERRIDDEN *and* `review.authorization_id == grant_id` | `test_grant_bound_to_unfinished_review_cannot_execute` |
| T12 | Code execution through rule packs | `yaml.safe_load`, closed grammar, no eval, dunder paths rejected, regex length limit | `test_grammar_rejects_unsafe_or_invalid`, `test_pack_rejects_python_yaml_tags` |
| T13 | Secrets or PII leaking into audit logs | Key-based redaction, truncation, email redaction on export; `SECRET_IN_PAYLOAD` rule blocks such actions | `test_sensitive_values_redacted_in_audit_and_export` |
| T14 | Agent impersonation over the API | Agents may only submit and execute actions with `agent_id == principal_id` | `test_role_checks` |
| T15 | Resource exhaustion through huge proposals | 64 KB payload/context limit, nesting depth 10, at most 100 evidence refs, rule pack at most 512 KB | `test_payload_size_and_depth_bounded` |

## Residual risks (not mitigated in this prototype)

- **Holder of the signing key**. Anyone with `SENTINEL_SIGNING_KEY` and database write access can
  mint grants. T11 limits the damage for review-backed grants, but an auto-approval grant
  requires only that the decision is APPROVE and points at it. Key management (HSM/KMS, rotation)
  is out of scope.
- **Development keys are public.** The dev config must never be deployed.
- **No transport security.** The API binds to 127.0.0.1 over plain HTTP.
- **Audit-log integrity.** Events are append-only by convention, not cryptographically chained. A
  database writer can alter history.
- **ReDoS.** Python's `re` has no timeout. Patterns are length-limited and inputs truncated to
  10,000 characters, but a pathological pattern written by a rule author could still be slow.
- **Thread-based timeouts.** A provider that hangs keeps its worker thread. The result is discarded,
  but the thread isn't killed.
- **Rule correctness.** Sentinel enforces the rules it is given. Gaps in the rule pack are gaps in
  protection.
