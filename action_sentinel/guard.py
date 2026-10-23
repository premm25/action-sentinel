"""Put an agent's tool functions behind Sentinel with a decorator.

    from action_sentinel.guard import Guard, GuardAdapter
    sentinel = Sentinel(settings, adapter=GuardAdapter())
    guard = Guard(sentinel, agent_id="ops-agent", agent_version="1.4.0", source_system="monitoring-agent",
                  mode="shadow")

    @guard.tool("restart_service", target=lambda a: "staging/%s" % a["service"])
    def restart_service(service: str, reason: str) -> str:
        ...

Each call becomes an `ActionProposal` (payload = the call's arguments unless `payload=` says otherwise).

shadow   Sentinel assesses the call and records what it *would* have done (audit event `shadow.decision`,
         `Guard.records`, optional JSONL log), then the tool runs regardless. Nothing is stored as a decision,
         no review is opened and no grant is issued. A Sentinel failure is recorded and never breaks the tool.
         This is for measuring Sentinel on real traffic before it is trusted to block anything.
enforce  The call runs only through Sentinel's execution gateway: APPROVE issues a grant, the gateway checks
         it (action hash, idempotency, current policy) and only then calls the tool, via `GuardAdapter`.
         BLOCK raises `ActionBlocked`; HUMAN_REVIEW raises `ActionHeld` with the review id, and the held call
         runs later with `Guard.resume(review_id, token)` once a reviewer approves; ABSTAIN, ERROR, or
         Sentinel itself failing raise `ActionDenied`. Enforce mode fails closed.

Only synchronous functions are supported, and the Sentinel instance must be in-process.
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from action_sentinel.contracts import ActionProposal, DecisionQuestion, GovernanceDecision, Outcome, new_id, utcnow

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
    result: Any = None
    error: BaseException | None = None
    ran: bool = False


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
    """Execution adapter that runs the guarded call the gateway has just authorized, and nothing else."""

    name = "guard"

    def __init__(self) -> None:
        self._calls: dict[str, _Call] = {}
        self._lock = threading.Lock()

    def _expect(self, key: str, call: _Call) -> None:
        with self._lock:
            if key in self._calls:
                raise RuntimeError("a guarded call with idempotency key %s is already in flight" % key)
            self._calls[key] = call

    def _release(self, key: str) -> None:
        with self._lock:
            self._calls.pop(key, None)

    def execute(self, action: ActionProposal) -> dict[str, Any]:
        with self._lock:
            call = self._calls.get(action.idempotency_key)
        if call is None:
            raise RuntimeError("no guarded call is waiting for idempotency key %s" % action.idempotency_key)
        call.ran = True
        try:
            call.result = call.fn(*call.args, **call.kwargs)
        except BaseException as exc:
            call.error = exc
            raise
        return {"tool": getattr(call.fn, "__qualname__", repr(call.fn)), "returned": type(call.result).__name__}


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class Guard:
    def __init__(self, sentinel, *, agent_id: str, agent_version: str, source_system: str,
                 mode: Mode = "shadow", task_id: str | Callable[[], str] | None = None,
                 question: DecisionQuestion | None = None, providers: list[str] | None = None,
                 shadow_log: str | Path | None = None, keep_records: int = 10_000):
        self.sentinel = sentinel
        self.agent_id, self.agent_version, self.source_system = agent_id, agent_version, source_system
        self.task_id = task_id
        self.question, self.providers = question, providers
        self.shadow_log = Path(shadow_log) if shadow_log else None
        self.records: deque[ShadowRecord] = deque(maxlen=keep_records)
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
        if value == "enforce" and not isinstance(self.sentinel.adapter, GuardAdapter):
            raise ValueError("enforce mode needs a Sentinel built with adapter=GuardAdapter(), so the tool only "
                             "runs through the execution gateway")
        self._mode = value

    # ---------------------------------------------------------------- decorator
    def tool(self, action_type: str | None = None, *, target: str | Callable[[Args], str],
             payload: Callable[[Args], dict] | None = None, evidence: Callable[[Args], list] | None = None,
             context: Callable[[Args], dict] | None = None,
             idempotency_key: Callable[[Args], str] | None = None):
        """Decorate a tool. Callables receive the call's bound arguments (defaults applied) as a dict."""
        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(fn):
                raise TypeError("Guard supports synchronous tools only (%s is async)" % fn.__qualname__)
            sig = inspect.signature(fn)
            at = action_type or fn.__name__

            @functools.wraps(fn)
            def guarded(*args: Any, **kwargs: Any) -> Any:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                a = dict(bound.arguments)
                action = self._proposal(at, a, target, payload, evidence, context, idempotency_key)
                call = _Call(fn, args, kwargs)
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
    def _shadow(self, tool: str, action: ActionProposal, call: _Call) -> Any:
        corr = new_id("shadow")
        rec = ShadowRecord(tool=tool, action_type=action.action_type, event_id=action.event_id, correlation_id=corr,
                           outcome=None, would_run=None)
        t0 = time.perf_counter()
        try:
            d = self.sentinel.assess(action, correlation_id=corr, question=self.question, providers=self.providers)
            rec.outcome, rec.would_run, rec.reason_codes = d.status.value, d.status == Outcome.APPROVE, d.reason_codes
            rec.provider_labels = {r.provider_name: r.label for r in d.provider_results}
        except Exception as exc:  # shadow mode must never break the tool
            rec.guard_error = "%s: %s" % (type(exc).__name__, str(exc)[:500])
        rec.assess_ms = round(1000 * (time.perf_counter() - t0), 2)
        try:
            return call.fn(*call.args, **call.kwargs)
        except BaseException as exc:
            rec.tool_error = "%s: %s" % (type(exc).__name__, str(exc)[:500])
            raise
        finally:
            self._record(rec, action)

    def _record(self, rec: ShadowRecord, action: ActionProposal) -> None:
        self.records.append(rec)
        try:
            self.sentinel.audit.emit("shadow.decision", rec.correlation_id, "guard", actor=self.agent_id,
                                     reason_codes=rec.reason_codes,
                                     data={**rec.to_json(), "action_hash": action.action_hash()})
            if self.shadow_log:
                with self._lock, self.shadow_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec.to_json()) + "\n")
        except Exception:  # recording is best effort in shadow mode; the in-memory record is kept
            pass

    # ---------------------------------------------------------------- enforce
    def _enforce(self, tool: str, action: ActionProposal, call: _Call) -> Any:
        try:
            res = self.sentinel.evaluate(action, question=self.question, providers=self.providers,
                                         actor=self.agent_id)
        except Exception as exc:
            raise ActionDenied("Sentinel could not evaluate %s: %s" % (tool, exc),
                               reason_codes=["SENTINEL_FAILED"]) from exc
        d = res.decision
        if d.status == Outcome.APPROVE and res.authorization is not None:
            return self._execute(res.authorization.token, action, call, tool)
        if d.status == Outcome.HUMAN_REVIEW and res.review is not None:
            with self._lock:
                self._held[res.review.review_id] = _Held(call, action, tool)
            raise ActionHeld("%s is waiting for human review %s" % (tool, res.review.review_id), d,
                             res.review.review_id)
        if d.status == Outcome.BLOCK:
            raise ActionBlocked("Sentinel blocked %s: %s" % (tool, ", ".join(d.reason_codes)), d)
        raise ActionDenied("Sentinel did not approve %s (%s)" % (tool, d.status.value), d)

    def resume(self, review_id: str, token: str) -> Any:
        """Run a held call with the grant a reviewer's approval issued. The gateway re-checks everything."""
        with self._lock:
            held = self._held.pop(review_id, None)
        if held is None:
            raise KeyError("no held call for review %s" % review_id)
        return self._execute(token, held.action, held.call, held.tool)

    def held(self) -> list[str]:
        with self._lock:
            return list(self._held)

    def _execute(self, token: str, action: ActionProposal, call: _Call, tool: str) -> Any:
        adapter: GuardAdapter = self.sentinel.adapter
        adapter._expect(action.idempotency_key, call)
        try:
            r = self.sentinel.execute(token, action, actor=self.agent_id)
        finally:
            adapter._release(action.idempotency_key)
        if r.status == "EXECUTED":
            return call.result
        if r.status == "FAILED" and call.error is not None:
            raise call.error
        raise ActionBlocked("the execution gateway refused %s: %s" % (tool, ", ".join(r.reason_codes)),
                            reason_codes=r.reason_codes)
