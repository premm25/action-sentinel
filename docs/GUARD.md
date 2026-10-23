# Guarding an agent's tools (`@guard.tool`)

`action_sentinel.guard` puts Sentinel in front of the Python functions an agent calls as tools, without
the agent having to build `ActionProposal`s by hand.

```python
from action_sentinel.config import load_settings
from action_sentinel.guard import ActionBlocked, ActionHeld, Guard, GuardAdapter
from action_sentinel.service import Sentinel

sentinel = Sentinel(load_settings("config/sentinel.yaml"), adapter=GuardAdapter())
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

In enforce mode the function runs **only through Sentinel's execution gateway**:

| outcome | what happens |
|---|---|
| APPROVE | a grant is issued. The gateway verifies it (signature, expiry, action hash, idempotency, current policy) and then calls the function via `GuardAdapter`, which runs only the call it was told to expect. The return value comes back to the caller. |
| BLOCK | `ActionBlocked` is raised and the function never runs |
| HUMAN_REVIEW | `ActionHeld(review_id)` is raised and the call is held. After a reviewer approves, `guard.resume(review_id, token)` runs it through the gateway, which re-checks everything. A grant issued for a different call is refused (`ACTION_MISMATCH`). |
| ABSTAIN, ERROR, or Sentinel failing | `ActionDenied` is raised (fail closed) |

If the gateway refuses a grant, the caller gets `ActionBlocked` with the gateway's reason codes. If the
tool itself raises, that exception reaches the caller and the execution is recorded as `FAILED`.

Enforce mode needs a Sentinel built with `adapter=GuardAdapter()`; otherwise `Guard` refuses to switch.

## Limits

- Synchronous functions only; async tools are rejected when you decorate them.
- In-process Sentinel only. A remote Sentinel (over the HTTP API) runs its own executor, so it can't call
  a function in the agent's process.
- Held calls live in the `Guard` object's memory. If the process restarts they are lost, but the review
  stays in Sentinel's store.
- The tests are in `tests/integration/test_guard.py`.
