"""Put an agent's tool functions behind Sentinel with a decorator.

    from action_sentinel.guard import Guard
    sentinel = Sentinel(settings)                       # in-process, or RemoteSentinel(url, api_key)
    guard = Guard(sentinel, agent_id="ops-agent", agent_version="1.4.0", source_system="monitoring-agent",
                  mode="shadow")

    @guard.tool("restart_service", target=lambda a: "staging/%s" % a["service"])
    def restart_service(service: str, reason: str) -> str:      # `async def` tools work the same way
        ...

Each call becomes an `ActionProposal` (payload = the call's arguments unless `payload=` says otherwise).

shadow   Sentinel assesses the call and records what it *would* have done (audit event `shadow.decision`,
         `Guard.records`, optional JSONL log), then the tool runs regardless. Nothing is stored as a decision,
         no review is opened and no grant is issued. A Sentinel failure is recorded and never breaks the tool.
         This is for measuring Sentinel on real traffic before it is trusted to block anything.
enforce  The tool runs only after Sentinel's execution gateway clears it: APPROVE issues a grant, the gateway
         checks it (action hash, idempotency, current policy) and claims the idempotency key, the tool runs,
         and its outcome is recorded with the execution. BLOCK raises `ActionBlocked`; HUMAN_REVIEW raises
         `ActionHeld` with the review id, and the held call runs later with `Guard.resume(review_id, token)`
         once a reviewer approves; ABSTAIN, ERROR, or Sentinel itself failing raise `ActionDenied`. Enforce
         mode fails closed.

Sentinel can be in-process (`Sentinel`) or a separate service (`action_sentinel.remote.RemoteSentinel`).
For an async tool, Sentinel's work runs in a worker thread so the event loop is not blocked.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from action_sentinel.contracts import (
    ActionProposal,
    DecisionQuestion,
    ExecutionResult,
    GovernanceDecision,
    Outcome,
    new_id,
    utcnow,
)

Mode = Literal["shadow", "enforce"]
Args = dict[str, Any]


class ActionDenied(Exception):
    """The guarded call did not run. `decision` is None when Sentinel itself failed."""

    def __init__(self, message: str, decision: GovernanceDecision | None = None,
                 reason_codes: list[str] | None = None):
        super().__init__(message)
        self.decision = decision
        self.reason_codes = reason_codes if reason_codes is not None else (decision.reason_codes if decision else [])


class ActionBlocked(ActionDenied):
    """Sentinel blocked the call, or the execution gateway rejected its grant."""


class ActionHeld(ActionDenied):
    """The call waits for a human. Run it with `Guard.resume(review_id, token)` after approval."""

    def __init__(self, message: str, decision: GovernanceDecision, review_id: str):
        super().__init__(message, decision)
        self.review_id = review_id


@dataclass
class _Call:
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict


@dataclass
class _Held:
    call: _Call
    action: ActionProposal
    tool: str


@dataclass
class ShadowRecord:
    tool: str
    action_type: str
    event_id: str
    correlation_id: str
    outcome: str | None             # what Sentinel decided; None if Sentinel failed
    would_run: bool | None          # True only for APPROVE
    reason_codes: list[str] = field(default_factory=list)
    provider_labels: dict[str, str | None] = field(default_factory=dict)
    assess_ms: float | None = None
    guard_error: str | None = None  # Sentinel failed; the tool still ran
    tool_error: str | None = None   # the tool itself raised (re-raised to the caller)
    at: str = field(default_factory=lambda: utcnow().isoformat())

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


class GuardAdapter:
    """Kept so existing `Sentinel(settings, adapter=GuardAdapter())` code keeps working; `Guard` no longer
    needs it. Guarded tools run in the guard, after the gateway's `begin` clears them, so this adapter
    refuses every direct `Sentinel.execute`."""

    name = "guard"

    def execute(self, action: ActionProposal) -> dict[str, Any]:
        raise RuntimeError("guarded tools run through Guard, not Sentinel.execute")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class _LocalBackend:
    """An in-process Sentinel."""

    def __init__(self, sentinel):
        self.sentinel = sentinel

    def assess(self, action, corr, question, providers) -> GovernanceDecision:
        return self.sentinel.assess(action, correlation_id=corr, question=question, providers=providers)

    def evaluate(self, action, question, providers, actor) -> tuple[GovernanceDecision, str | None, str | None]:
        res = self.sentinel.evaluate(action, question=question, providers=providers, actor=actor)
        return (res.decision, res.review.review_id if res.review else None,
                res.authorization.token if res.authorization else None)

    def begin(self, token, action, actor) -> ExecutionResult:
        return self.sentinel.begin_execution(token, action, actor=actor, adapter="guard")

    def finish(self, execution_id, ok, output, actor) -> ExecutionResult:
        return self.sentinel.finish_execution(execution_id, ok, output, actor=actor)

    def record(self, rec: ShadowRecord, action: ActionProposal, actor: str) -> None:
        self.sentinel.audit.emit("shadow.decision", rec.correlation_id, "guard", actor=actor,
                                 reason_codes=rec.reason_codes,
                                 data={**rec.to_json(), "action_hash": action.action_hash()})


class _RemoteBackend:
    """A Sentinel service over HTTP (`RemoteSentinel`). The server takes the actor from the API key."""

    def __init__(self, remote):
        self.remote = remote

    def assess(self, action, corr, question, providers) -> GovernanceDecision:
        return self.remote.assess(action, question=question, providers=providers)

    def evaluate(self, action, question, providers, actor) -> tuple[GovernanceDecision, str | None, str | None]:
        res = self.remote.evaluate(action, question=question, providers=providers)
        return res.decision, res.review_id, res.authorization_token

    def begin(self, token, action, actor) -> ExecutionResult:
        return self.remote.begin_execution(token, action, adapter="guard")

    def finish(self, execution_id, ok, output, actor) -> ExecutionResult:
        return self.remote.finish_execution(execution_id, ok, output)

    def record(self, rec: ShadowRecord, action: ActionProposal, actor: str) -> None:
        self.remote.record_shadow(rec.to_json(), action.action_hash())


class Guard:
    def __init__(self, sentinel, *, agent_id: str, agent_version: str, source_system: str,
                 mode: Mode = "shadow", task_id: str | Callable[[], str] | None = None,
                 question: DecisionQuestion | None = None, providers: list[str] | None = None,
                 shadow_log: str | Path | None = None, keep_records: int = 10_000):
        from action_sentinel.remote import RemoteSentinel
        self.sentinel = sentinel
        self._backend = _RemoteBackend(sentinel) if isinstance(sentinel, RemoteSentinel) else _LocalBackend(sentinel)
        self.agent_id, self.agent_version, self.source_system = agent_id, agent_version, source_system
        self.task_id = task_id
        self.question, self.providers = question, providers
        self.shadow_log = Path(shadow_log) if shadow_log else None
        self.records: deque[ShadowRecord] = deque(maxlen=keep_records)
        # Executions whose tool ran but whose outcome could not be reported (Sentinel unreachable afterwards).
        # They stay STARTED in Sentinel with the idempotency key claimed, so they are never re-run blindly.
        self.unfinished: list[str] = []
        self._held: dict[str, _Held] = {}
        self._lock = threading.Lock()
        self.mode = mode  # validated by the property setter

    @property
    def mode(self) -> Mode:
        return self._mode

    @mode.setter
    def mode(self, value: Mode) -> None:
        if value not in ("shadow", "enforce"):
            raise ValueError("mode must be 'shadow' or 'enforce'")
        self._mode = value

    # ---------------------------------------------------------------- decorator
    def tool(self, action_type: str | None = None, *, target: str | Callable[[Args], str],
             payload: Callable[[Args], dict] | None = None, evidence: Callable[[Args], list] | None = None,
             context: Callable[[Args], dict] | None = None,
             idempotency_key: Callable[[Args], str] | None = None):
        """Decorate a tool, sync or async. Callables receive the call's bound arguments (defaults applied)."""
        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            sig = inspect.signature(fn)
            at = action_type or fn.__name__

            def prepare(args: tuple, kwargs: dict) -> tuple[ActionProposal, _Call]:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                action = self._proposal(at, dict(bound.arguments), target, payload, evidence, context,
                                        idempotency_key)
                return action, _Call(fn, args, kwargs)

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def aguarded(*args: Any, **kwargs: Any) -> Any:
                    action, call = prepare(args, kwargs)
                    if self.mode == "shadow":
                        return await self._ashadow(fn.__qualname__, action, call)
                    return await self._aenforce(fn.__qualname__, action, call)
                aguarded.guard = self  # type: ignore[attr-defined]
                return aguarded

            @functools.wraps(fn)
            def guarded(*args: Any, **kwargs: Any) -> Any:
                action, call = prepare(args, kwargs)
                if self.mode == "shadow":
                    return self._shadow(fn.__qualname__, action, call)
                return self._enforce(fn.__qualname__, action, call)
            guarded.guard = self  # type: ignore[attr-defined]
            return guarded
        return wrap

    def _proposal(self, action_type, a, target, payload, evidence, context, idempotency_key) -> ActionProposal:
        event_id = new_id("evt")
        task = self.task_id() if callable(self.task_id) else (self.task_id or new_id("task"))
        return ActionProposal.model_validate({
            "event_id": event_id, "task_id": task, "agent_id": self.agent_id, "agent_version": self.agent_version,
            "action_type": action_type, "action_payload": _jsonable(payload(a) if payload else a),
            "target_resource": target(a) if callable(target) else target, "source_system": self.source_system,
            "evidence_refs": _jsonable(evidence(a)) if evidence else [], "proposed_at": utcnow(),
            "context": _jsonable(context(a)) if context else {},
            "idempotency_key": idempotency_key(a) if idempotency_key else "idem-%s" % event_id})

    # ---------------------------------------------------------------- shadow
    def _assess(self, tool: str, action: ActionProposal) -> ShadowRecord:
        rec = ShadowRecord(tool=tool, action_type=action.action_type, event_id=action.event_id,
                           correlation_id=new_id("shadow"), outcome=None, would_run=None)
        t0 = time.perf_counter()
        try:
            d = self._backend.assess(action, rec.correlation_id, self.question, self.providers)
            rec.correlation_id = d.correlation_id
            rec.outcome, rec.would_run, rec.reason_codes = d.status.value, d.status == Outcome.APPROVE, d.reason_codes
            rec.provider_labels = {r.provider_name: r.label for r in d.provider_results}
        except Exception as exc:  # shadow mode must never break the tool
            rec.guard_error = "%s: %s" % (type(exc).__name__, str(exc)[:500])
        rec.assess_ms = round(1000 * (time.perf_counter() - t0), 2)
        return rec

    def _shadow(self, tool: str, action: ActionProposal, call: _Call) -> Any:
        rec = self._assess(tool, action)
        try:
            return call.fn(*call.args, **call.kwargs)
        except BaseException as exc:
            rec.tool_error = "%s: %s" % (type(exc).__name__, str(exc)[:500])
            raise
        finally:
            self._record(rec, action)

    async def _ashadow(self, tool: str, action: ActionProposal, call: _Call) -> Any:
        rec = await asyncio.to_thread(self._assess, tool, action)
        try:
            return await call.fn(*call.args, **call.kwargs)
        except BaseException as exc:
            rec.tool_error = "%s: %s" % (type(exc).__name__, str(exc)[:500])
            raise
        finally:
            await asyncio.to_thread(self._record, rec, action)

    def _record(self, rec: ShadowRecord, action: ActionProposal) -> None:
        self.records.append(rec)
        try:
            self._backend.record(rec, action, self.agent_id)
        except Exception:  # recording is best effort in shadow mode; the in-memory record is kept
            pass
        try:
            if self.shadow_log:
                with self._lock, self.shadow_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec.to_json()) + "\n")
        except Exception:
            pass

    # ---------------------------------------------------------------- enforce
    def _decide(self, tool: str, action: ActionProposal, call: _Call) -> str:
        """Ask Sentinel. Returns the grant token of an approved call; otherwise holds the call or raises."""
        try:
            d, review_id, token = self._backend.evaluate(action, self.question, self.providers, self.agent_id)
        except Exception as exc:
            raise ActionDenied("Sentinel could not evaluate %s: %s" % (tool, exc),
                               reason_codes=["SENTINEL_FAILED"]) from exc
        if d.status == Outcome.APPROVE and token is not None:
            return token
        if d.status == Outcome.HUMAN_REVIEW and review_id is not None:
            with self._lock:
                self._held[review_id] = _Held(call, action, tool)
            raise ActionHeld("%s is waiting for human review %s" % (tool, review_id), d, review_id)
        if d.status == Outcome.BLOCK:
            raise ActionBlocked("Sentinel blocked %s: %s" % (tool, ", ".join(d.reason_codes)), d)
        raise ActionDenied("Sentinel did not approve %s (%s)" % (tool, d.status.value), d)

    def _begin(self, token: str, action: ActionProposal, tool: str) -> str:
        """The gateway checks the grant and claims the idempotency key. Returns the execution id."""
        try:
            r = self._backend.begin(token, action, self.agent_id)
        except Exception as exc:
            raise ActionDenied("Sentinel could not start %s: %s" % (tool, exc),
                               reason_codes=["SENTINEL_FAILED"]) from exc
        if r.status != "STARTED":
            raise ActionBlocked("the execution gateway refused %s: %s" % (tool, ", ".join(r.reason_codes)),
                                reason_codes=r.reason_codes)
        return r.execution_id

    def _finish(self, execution_id: str, call: _Call, error: BaseException | None, result: Any = None) -> None:
        if error is None:
            ok, output = True, {"tool": _name(call.fn), "returned": type(result).__name__}
        else:
            ok, output = False, {"error": "%s: %s" % (type(error).__name__, str(error)[:500])}
        try:
            self._backend.finish(execution_id, ok, output, self.agent_id)
        except Exception as exc:  # the tool already ran: never turn its outcome into a different one now
            with self._lock:
                self.unfinished.append(execution_id)
            warnings.warn("could not record the outcome of execution %s: %s" % (execution_id, exc), stacklevel=3)

    def _run(self, token: str, action: ActionProposal, call: _Call, tool: str) -> Any:
        execution_id = self._begin(token, action, tool)
        try:
            result = call.fn(*call.args, **call.kwargs)
        except BaseException as exc:
            self._finish(execution_id, call, exc)
            raise
        self._finish(execution_id, call, None, result)
        return result

    async def _arun(self, token: str, action: ActionProposal, call: _Call, tool: str) -> Any:
        execution_id = await asyncio.to_thread(self._begin, token, action, tool)
        try:
            result = await call.fn(*call.args, **call.kwargs)
        except BaseException as exc:
            await asyncio.to_thread(self._finish, execution_id, call, exc)
            raise
        await asyncio.to_thread(self._finish, execution_id, call, None, result)
        return result

    def _enforce(self, tool: str, action: ActionProposal, call: _Call) -> Any:
        return self._run(self._decide(tool, action, call), action, call, tool)

    async def _aenforce(self, tool: str, action: ActionProposal, call: _Call) -> Any:
        token = await asyncio.to_thread(self._decide, tool, action, call)
        return await self._arun(token, action, call, tool)

    # ---------------------------------------------------------------- held calls
    def _take_held(self, review_id: str, *, sync: bool) -> _Held:
        with self._lock:
            held = self._held.get(review_id)
            if held is None:
                raise KeyError("no held call for review %s" % review_id)
            if sync and inspect.iscoroutinefunction(held.call.fn):
                raise TypeError("%s is async: use `await guard.aresume(review_id, token)`" % held.tool)
            return self._held.pop(review_id)

    def resume(self, review_id: str, token: str) -> Any:
        """Run a held call with the grant a reviewer's approval issued. The gateway re-checks everything."""
        held = self._take_held(review_id, sync=True)
        return self._run(token, held.action, held.call, held.tool)

    async def aresume(self, review_id: str, token: str) -> Any:
        """`resume` for code inside an event loop; works for sync and async tools."""
        held = self._take_held(review_id, sync=False)
        if inspect.iscoroutinefunction(held.call.fn):
            return await self._arun(token, held.action, held.call, held.tool)
        return await asyncio.to_thread(self._run, token, held.action, held.call, held.tool)

    def held(self) -> list[str]:
        with self._lock:
            return list(self._held)


def _name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__qualname__", repr(fn))
