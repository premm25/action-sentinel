# Security

This prototype is meant for local development and evaluation. It is not hardened for production.
Threats and mitigations are mapped in [`../architecture/THREAT_MODEL.md`](../architecture/THREAT_MODEL.md).

## Controls that are implemented and tested

- **Fail-closed decisions.** Provider errors, timeouts, malformed output, abstention and internal
  validation errors can never produce `APPROVE` or an authorization.
- **Policy precedence.** Hard rule violations produce `BLOCK` regardless of model confidence or how
  many providers agree.
- **Execution boundary.** Tokens are HMAC-SHA256 signed and bound to the action hash, idempotency
  key, policy version and expiry (default 15 minutes). Every check is repeated at execution time,
  including current hard rules. Idempotency is claimed atomically. Grants are revocable.
- **Human review.** Reviewer levels (operator < senior < admin) are derived from risk. Self-review
  is forbidden. Inactive reviewers are refused. Rejections and overrides require a reason.
  Approvals and overrides are re-validated against the current policy, and overrides can't change
  identity fields (agent, source, action type, idempotency key).
- **API authentication and authorization.** Keys are compared by SHA-256 hash in constant time.
  Endpoints check roles, reviewer identity comes from the key, agents are limited to their own
  `agent_id`, and only `/health` is public.
- **Input bounds.** Proposals are strict (unknown fields rejected), payload and context are limited
  to 64 KB and depth 10, and timestamps must be timezone-aware. Rule packs are loaded with
  `yaml.safe_load` into a closed grammar with no code execution and are capped at 512 KB.
- **Sensitive data.** Audit data redacts values under keys matching
  password/secret/token/api_key/authorization/credential/private_key/cookie and truncates long
  values. JSONL export also replaces email addresses. The `SECRET_IN_PAYLOAD` rule blocks actions
  that carry credentials.
- **Secrets.** The source tree holds none. The signing key comes only from `SENTINEL_SIGNING_KEY`
  (required when `environment: prod`). The dev API keys are documented as public and stored only
  as hashes.
- **No production side effects.** The only execution adapter is `MockExecutionAdapter`, which
  records calls and changes nothing.

## Known limitations

- Public development keys are in `config/sentinel.yaml`. Replace the `principals` before any shared
  deployment.
- There is no TLS, rate limiting, key rotation, account lockout or key expiry.
- Without `SENTINEL_SIGNING_KEY`, each process uses an ephemeral random key, so tokens can't be
  verified across processes or restarts.
- Audit records are not tamper-evident. Hash chaining or an external append-only sink would be needed.
- A local SQLite file can be read by anyone with filesystem access, and it holds sanitized but
  otherwise unencrypted proposal data.
- Python `re` patterns in rule packs have no execution timeout (they are length-limited only).
- Prompt-injection text in proposals reaches LLM providers. This is mitigated only by policy
  precedence: the model can't approve on its own.

## What production would require

A secrets manager or HSM for the signing key, with rotation. SSO/OIDC with short-lived credentials.
TLS. A transactional database with backups and audit retention. Tamper-evident audit logs.
Rate limits and quotas. Signed and reviewed rule-pack releases. Real execution adapters with least
privilege, dry-run and rollback. Monitoring and alerting on BLOCK/ERROR spikes.

## Reporting

This is a local project with no disclosure process. Record issues in `PROJECT_STATUS.md`.
