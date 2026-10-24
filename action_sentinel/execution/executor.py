"""The execution boundary. The only path to an adapter runs through ExecutionGateway.execute.

`execute` runs the configured adapter. `begin` and `finish` split that in two for a caller that runs the
action itself (the guard, in the agent's process, locally or over the API): `begin` runs every check below
and claims the idempotency key, then the caller runs the action and reports the outcome with `finish`.

Checks, in order (the first failure rejects; nothing is executed):
 1. token signature, grant existence, revocation, expiry           (Authorizer.verify)
 2. the action presented matches the authorized action hash        ACTION_MISMATCH
 3. the decision is eligible (APPROVE, or a completed review)      DECISION_NOT_EXECUTABLE
 4. the grant's policy version is still accepted                   POLICY_VERSION_STALE
 5. current hard policy constraints still pass                     SAFETY_CONSTRAINT_FAILED
 6. the idempotency key has not been executed (atomic claim)       DUPLICATE
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from action_sentinel.contracts import (
    ActionProposal,
    ExecutionResult,
    Outcome,
    ReviewState,
    ValidationReport,
    new_id,
    utcnow,
)

from .authorization import AuthorizationError, Authorizer


class ExecutionStateError(Exception):
    """`finish` was called for an execution that does not exist or is not STARTED."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ExecutionAdapter(Protocol):
    name: str

    def execute(self, action: ActionProposal) -> dict[str, Any]: ...


class MockExecutionAdapter:
    """Records what it was asked to do. Touches no real system."""

    name = "mock-executor"

    def __init__(self, fail_actions: set[str] | None = None):
        self.executed: list[ActionProposal] = []
        self.fail_actions = fail_actions or set()

    def execute(self, action: ActionProposal) -> dict[str, Any]:
        if action.action_type in self.fail_actions:
            raise RuntimeError("simulated adapter failure for %s" % action.action_type)
        self.executed.append(action)
        return {"simulated": True, "action_type": action.action_type, "target_resource": action.target_resource,
                "message": "mock execution only; no real system was changed"}


class ExecutionGateway:
    component = "execution"

    def __init__(self, store, authorizer: Authorizer, adapter: ExecutionAdapter,
                 revalidate: Callable[[ActionProposal], ValidationReport],
                 accepted_policy_versions: Callable[[], set[str]], audit):
        self.store = store
        self.authorizer = authorizer
        self.adapter = adapter
        self.revalidate = revalidate
        self.accepted_policy_versions = accepted_policy_versions
        self.audit = audit

    def execute(self, token: str, action: ActionProposal, actor: str = "unknown") -> ExecutionResult:
        started = self.begin(token, action, actor, self.adapter.name)
        if started.status != "STARTED":
            return started
        try:
            output = self.adapter.execute(action)
        except Exception as exc:
            # The key stays claimed: a partially applied action must not be retried blindly.
            return self.finish(started.execution_id, False, {"error": "%s: %s" % (type(exc).__name__, exc)}, actor)
        return self.finish(started.execution_id, True, output, actor)

    def begin(self, token: str, action: ActionProposal, actor: str = "unknown",
              adapter: str | None = None) -> ExecutionResult:
        """Run every check and claim the idempotency key. STARTED means the caller may now run the action."""
        corr = None
        try:
            grant = self.authorizer.verify(token)
            corr = grant.correlation_id
            self._check(grant, action)
        except AuthorizationError as exc:
            return self._reject(exc.code, str(exc), action, actor, corr, grant_id=_grant_id(token))

        execution_id = new_id("exe")
        holder = self.store.claim_idempotency_key(action.idempotency_key, grant.grant_id, execution_id)
        if holder is not None:
            result = ExecutionResult(
                grant_id=grant.grant_id, decision_id=grant.decision_id, correlation_id=corr,
                idempotency_key=action.idempotency_key, status="DUPLICATE", reason_codes=["DUPLICATE_EXECUTION"],
                finished_at=utcnow(), original_execution_id=holder)
            self.store.save_execution(result)
            self.audit.emit("execution.duplicate", corr, self.component, decision_id=grant.decision_id, actor=actor,
                            reason_codes=result.reason_codes, data={"original_execution_id": holder})
            return result

        result = ExecutionResult(execution_id=execution_id, grant_id=grant.grant_id, decision_id=grant.decision_id,
                                 correlation_id=corr, idempotency_key=action.idempotency_key, status="STARTED",
                                 adapter=adapter, started_by=actor)
        self.store.save_execution(result)
        self.audit.emit("execution.started", corr, self.component, decision_id=grant.decision_id, actor=actor,
                        data={"execution_id": execution_id, "adapter": adapter})
        return result

    def finish(self, execution_id: str, ok: bool, output: dict[str, Any] | None = None,
               actor: str = "unknown") -> ExecutionResult:
        """Record how a STARTED execution ended. Each execution finishes once; the key stays claimed either way."""
        started = self.store.get_execution(execution_id)
        if started is None:
            raise ExecutionStateError("NOT_FOUND", "no execution %s" % execution_id)
        if started.status != "STARTED":
            raise ExecutionStateError("NOT_STARTED", "execution %s is %s, not STARTED" % (execution_id, started.status))
        result = started.model_copy(update={"status": "EXECUTED" if ok else "FAILED", "finished_at": utcnow(),
                                            "reason_codes": ["EXECUTED" if ok else "ADAPTER_FAILURE"],
                                            "output": output or {}})
        result = ExecutionResult.model_validate(result.model_dump())  # validates the caller's output
        self.store.save_execution(result)
        self.audit.emit("execution.%s" % result.status.lower(), result.correlation_id or "uncorrelated",
                        self.component, decision_id=result.decision_id, actor=actor, reason_codes=result.reason_codes,
                        data={"execution_id": result.execution_id, "output": result.output})
        return result

    def _check(self, grant, action: ActionProposal) -> None:
        if action.action_hash() != grant.action_hash:
            raise AuthorizationError("ACTION_MISMATCH", "the action differs from the one that was authorized")
        decision = self.store.get_decision(grant.decision_id)
        if decision is None:
            raise AuthorizationError("DECISION_NOT_EXECUTABLE", "authorizing decision not found")
        if decision.status in (Outcome.BLOCK, Outcome.ERROR):
            raise AuthorizationError("DECISION_NOT_EXECUTABLE", "decision status %s can never execute"
                                     % decision.status.value)
        if grant.review_id:
            # Human path: a completed review must have issued exactly this grant.
            review = self.store.get_review(grant.review_id)
            if review is None or review.state not in (ReviewState.APPROVED, ReviewState.OVERRIDDEN) \
                    or review.authorization_id != grant.grant_id:
                raise AuthorizationError("DECISION_NOT_EXECUTABLE", "review has not approved this authorization")
        elif decision.status != Outcome.APPROVE or decision.authorization_id != grant.grant_id:
            raise AuthorizationError("DECISION_NOT_EXECUTABLE", "decision status %s was not auto-approved with "
                                     "this grant" % decision.status.value)
        if grant.policy_version not in self.accepted_policy_versions():
            raise AuthorizationError("POLICY_VERSION_STALE", "authorized under policy %s, which is no longer "
                                     "accepted" % grant.policy_version)
        report = self.revalidate(action)
        if report.hard_violations:
            raise AuthorizationError("SAFETY_CONSTRAINT_FAILED", "current policy blocks the action: %s"
                                     % ", ".join(f.code for f in report.hard_violations))

    def _reject(self, code: str, message: str, action: ActionProposal, actor: str, corr: str | None,
                grant_id: str | None) -> ExecutionResult:
        result = ExecutionResult(grant_id=grant_id, decision_id=None, correlation_id=corr,
                                 idempotency_key=action.idempotency_key, status="REJECTED", reason_codes=[code],
                                 finished_at=utcnow(), output={"message": message})
        self.store.save_execution(result)
        self.audit.emit("execution.rejected", corr or "uncorrelated", self.component, actor=actor,
                        reason_codes=[code], data={"message": message, "event_id": action.event_id})
        return result


def _grant_id(token: str) -> str | None:
    return token.split(".")[0] if isinstance(token, str) and "." in token else None
