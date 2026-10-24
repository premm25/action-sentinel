# Guarding an agent's tools (`@guard.tool`)

`action_sentinel.guard` puts Sentinel in front of the Python functions an agent calls as tools, without
the agent having to build `ActionProposal`s by hand.

```python
from action_sentinel.config import load_settings
from action_sentinel.guard import ActionBlocked, ActionHeld, Guard
from action_sentinel.service import Sentinel

sentinel = Sentinel(load_settings("config/sentinel.yaml"))
guard = Guard(sentinel, agent_id="ops-agent", agent_version="1.4.0", source_system="infra-agent",
              mode="shadow", shadow_log="var/shadow.jsonl")

@guard.tool("scale_deployment", target=lambda a: "prod/%s" % a["deployment"],
            evidence=lambda a: [{"ref_id": a["alert_id"], "kind": "alert", "stance": "supports"}],
            payload=lambda a: {"deployment": a["deployment"], "replicas": a["replicas"]})
def scale(deployment: str, replicas: int, alert_id: str) -> str:
    ...
```

- Each call becomes one proposal. `action_type` defaults to the function name, and `payload` defaults
  to the call's bound arguments, with defaults applied and anything that isn't JSON turned into a string.
- `target`, `payload`, `evidence`, `context` and `idempotency_key` can be values or callables. A callable
  receives those arguments as a dict.
- `async def` tools work the same way: decorate them and `await` them. Sentinel's work runs in a worker
  thread, so the event loop keeps running while it decides.
- By default every call gets a fresh idempotency key. Pass `idempotency_key=` when retries of the same
  logical operation must execute only once.

## Shadow mode: measure before trusting

In shadow mode the tool **always runs**, exactly as it would without Sentinel. Before it runs, Sentinel
assesses the call (validation, providers, uncertainty, governance) and records what it would have done:

- an audit event `shadow.decision` (component `guard`), with the outcome, reason codes, provider labels
  and the action hash
- `guard.records`, an in-memory list of `ShadowRecord`
- one JSON line per call in `shadow_log`, if one is given

Shadow mode creates no decision, review queue item or authorization grant. If Sentinel fails, the
failure is recorded in `guard_error` and the tool still runs. If the tool raises, its error is recorded
in `tool_error` and re-raised unchanged.

This is the way to collect real agent traffic and Sentinel's verdicts on it before turning blocking on.
Pair it with human labels to get evaluation data that isn't synthetic.

## Enforce mode

In enforce mode the function runs **only after Sentinel's execution gateway clears it**:

| outcome | what happens |
|---|---|
| APPROVE | a grant is issued. The gateway verifies it (signature, expiry, action hash, idempotency, current policy) and claims the idempotency key (`begin`, status `STARTED`). Then the function runs, and its outcome is recorded (`finish`: `EXECUTED` or `FAILED`). The return value comes back to the caller. |
| BLOCK | `ActionBlocked` is raised and the function never runs |
| HUMAN_REVIEW | `ActionHeld(review_id)` is raised and the call is held. After a reviewer approves, `guard.resume(review_id, token)` (or `await guard.aresume(...)` in async code) runs it through the gateway, which re-checks everything. A grant issued for a different call is refused (`ACTION_MISMATCH`). |
| ABSTAIN, ERROR, or Sentinel failing | `ActionDenied` is raised (fail closed) |

If the gateway refuses a grant, the caller gets `ActionBlocked` with the gateway's reason codes. If the
tool itself raises, that exception reaches the caller and the execution is recorded as `FAILED`.

If Sentinel can't be reached *after* the tool ran, the tool's result is still returned (the action
happened), a warning is issued and the execution id is added to `guard.unfinished`. In Sentinel that
execution stays `STARTED` with its idempotency key claimed, so it is never re-run blindly.

## Sentinel as a separate service (remote mode)

The guard can use a Sentinel running elsewhere (`action-sentinel serve`) instead of one in the agent's
process. Only the client changes:

```python
from action_sentinel.remote import RemoteSentinel

sentinel = RemoteSentinel("http://sentinel.internal:8000", api_key=os.environ["SENTINEL_API_KEY"])
guard = Guard(sentinel, agent_id="ops-agent", agent_version="1.4.0", source_system="infra-agent",
              mode="enforce")
```

- The API key's principal needs the `agent` role and must have the same id as `agent_id`.
- Shadow mode calls `POST /decisions/assess`, which stores nothing but audit events, and then
  `POST /shadow-records`.
- Enforce mode calls `POST /decisions/evaluate`, then `POST /executions/begin` and
  `POST /executions/{id}/finish`. Only the agent that started an execution may finish it, and only once.
- Reviewers approve over the API as usual (`POST /reviews/{id}/approve`) and hand the token to
  `guard.resume`.
- The client uses only the standard library. Its default timeout is 300 s, because a cold VegaML provider
  can take minutes on CPU; start the server with `warm_up: true`.

What remote mode does and doesn't guarantee: Sentinel decides, holds, audits and refuses to clear an
action twice, but the tool runs in the agent's process, so it relies on the agent going through the
guard. If an agent must be *unable* to act without approval, keep the credentials for the real system
on the Sentinel side and run the action there, with an execution adapter behind `POST /executions`.

## Limits

- Held calls live in the `Guard` object's memory. If the process restarts they are lost, but the review
  stays in Sentinel's store.
- The tests are in `tests/integration/test_guard.py` and `tests/api/test_remote_guard.py`.
