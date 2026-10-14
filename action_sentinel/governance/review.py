"""Human review state machine and reviewer authorization.

The rules for state changes and permissions live here. Revalidation and authorization are injected
by the service, so this module never decides on its own that an action may run.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from action_sentinel.contracts import (
    REVIEW_TRANSITIONS,
    ActionProposal,
    ApprovalLevel,
    GovernanceDecision,
    ReviewHistoryEntry,
    Reviewer,
    ReviewRecord,
    ReviewState,
    utcnow,
)

MIN_REASON_LEN = 5


class ReviewError(Exception):
    http_status = 400

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ReviewNotFound(ReviewError):
    http_status = 404


class InvalidTransition(ReviewError):
    http_status = 409


class ReviewerNotAuthorized(ReviewError):
    http_status = 403


class RevalidationFailed(ReviewError):
    http_status = 422


def transition(record: ReviewRecord, to: ReviewState, actor: str, action: str, reason: str | None = None,
               detail: dict | None = None, now: datetime | None = None) -> ReviewRecord:
    if to not in REVIEW_TRANSITIONS[record.state]:
        raise InvalidTransition("INVALID_TRANSITION", "cannot move review from %s to %s"
                                % (record.state.value, to.value))
    entry = ReviewHistoryEntry(at=now or utcnow(), actor=actor, action=action, from_state=record.state, to_state=to,
                               reason=reason, detail=detail or {})
    return record.model_copy(update={"state": to, "history": [*record.history, entry]})


def note(record: ReviewRecord, actor: str, action: str, reason: str | None = None, detail: dict | None = None,
         now: datetime | None = None) -> ReviewRecord:
    """Record an attempt that did not change state (denied, failed revalidation)."""
    entry = ReviewHistoryEntry(at=now or utcnow(), actor=actor, action=action, from_state=record.state,
                               to_state=record.state, reason=reason, detail=detail or {})
    return record.model_copy(update={"history": [*record.history, entry]})


def new_review(decision: GovernanceDecision, ttl_seconds: int, now: datetime | None = None) -> ReviewRecord:
    now = now or utcnow()
    r = ReviewRecord(decision_id=decision.decision_id, correlation_id=decision.correlation_id,
                     required_level=max(decision.required_approval_level, ApprovalLevel.OPERATOR,
                                        key=lambda lv: lv.rank),
                     reason_codes=decision.reason_codes, created_at=now,
                     expires_at=now + timedelta(seconds=ttl_seconds), original_action=decision.action)
    return r.model_copy(update={"history": [ReviewHistoryEntry(
        at=now, actor="sentinel", action="created", from_state=ReviewState.PENDING, to_state=ReviewState.PENDING,
        detail={"reason_codes": decision.reason_codes})]})


def check_reviewer(record: ReviewRecord, reviewer: Reviewer | None, required: ApprovalLevel | None = None) -> Reviewer:
    if reviewer is None:
        raise ReviewerNotAuthorized("UNKNOWN_REVIEWER", "reviewer is not registered")
    if not reviewer.active:
        raise ReviewerNotAuthorized("REVIEWER_INACTIVE", "reviewer %s is inactive" % reviewer.reviewer_id)
    need = required or record.required_level
    if reviewer.level.rank < need.rank:
        raise ReviewerNotAuthorized("INSUFFICIENT_APPROVAL_LEVEL", "review requires %s, reviewer %s is %s"
                                    % (need.value, reviewer.reviewer_id, reviewer.level.value))
    if reviewer.reviewer_id == record.original_action.agent_id:
        raise ReviewerNotAuthorized("SELF_REVIEW", "the proposing agent cannot review its own action")
    return reviewer


def require_reason(reason: str | None) -> str:
    if reason is None or len(reason.strip()) < MIN_REASON_LEN:
        raise ReviewError("REASON_REQUIRED", "a reason of at least %d characters is required" % MIN_REASON_LEN)
    return reason.strip()


def ensure_open(record: ReviewRecord, now: datetime, expire: Callable[[ReviewRecord], ReviewRecord]) -> None:
    """Raise if the review is terminal; expire it (via callback) if its deadline has passed."""
    if record.state != ReviewState.PENDING:
        raise InvalidTransition("REVIEW_CLOSED", "review is already %s" % record.state.value)
    if now >= record.expires_at:
        expire(record)
        raise InvalidTransition("REVIEW_EXPIRED", "review expired at %s" % record.expires_at.isoformat())


OVERRIDABLE_FIELDS = frozenset({"action_payload", "target_resource", "context"})


def apply_override(original: ActionProposal, changes: dict, n: int) -> ActionProposal:
    """Build the overridden proposal. Identity fields cannot be changed by a reviewer."""
    bad = set(changes) - OVERRIDABLE_FIELDS
    if bad:
        raise ReviewError("FIELD_NOT_OVERRIDABLE", "fields %s cannot be overridden (allowed: %s)"
                          % (sorted(bad), sorted(OVERRIDABLE_FIELDS)))
    if not changes:
        raise ReviewError("EMPTY_OVERRIDE", "an override must change something; use approve instead")
    data = original.model_dump(mode="json")
    data.update(changes)
    data["event_id"] = "%s.override%d" % (original.event_id[:110], n)
    try:
        return ActionProposal.model_validate(data)
    except ValueError as exc:
        raise ReviewError("INVALID_OVERRIDE", "overridden action is invalid: %s" % exc) from exc
