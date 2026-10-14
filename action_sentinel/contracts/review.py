"""Human review records and their lifecycle."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, JsonValue

from .actions import ActionProposal
from .common import ApprovalLevel, StrictModel, new_id, utcnow


class ReviewState(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    OVERRIDDEN = "OVERRIDDEN"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


TERMINAL_REVIEW_STATES = frozenset({
    ReviewState.APPROVED, ReviewState.REJECTED, ReviewState.OVERRIDDEN, ReviewState.EXPIRED, ReviewState.CANCELLED,
})

# The only legal transitions. Anything else is rejected.
REVIEW_TRANSITIONS: dict[ReviewState, frozenset[ReviewState]] = {
    ReviewState.PENDING: frozenset({
        ReviewState.APPROVED, ReviewState.REJECTED, ReviewState.OVERRIDDEN, ReviewState.EXPIRED, ReviewState.CANCELLED,
    }),
    **{s: frozenset() for s in TERMINAL_REVIEW_STATES},
}


class Reviewer(StrictModel):
    reviewer_id: str
    display_name: str = ""
    level: ApprovalLevel
    active: bool = True


class ReviewHistoryEntry(StrictModel):
    at: datetime = Field(default_factory=utcnow)
    actor: str
    action: Literal[
        "created", "approve", "reject", "override", "override_rejected", "approve_rejected",
        "expire", "cancel", "denied",
    ]
    from_state: ReviewState
    to_state: ReviewState
    reason: str | None = None
    detail: dict[str, JsonValue] = Field(default_factory=dict)


class ReviewRecord(StrictModel):
    review_id: str = Field(default_factory=lambda: new_id("rev"))
    decision_id: str
    correlation_id: str
    state: ReviewState = ReviewState.PENDING
    required_level: ApprovalLevel
    reason_codes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    original_action: ActionProposal
    final_action: ActionProposal | None = None
    reviewer_id: str | None = None
    reviewer_reason: str | None = None
    decided_at: datetime | None = None
    final_disposition: Literal["approved", "rejected", "overridden", "expired", "cancelled"] | None = None
    override_decision_id: str | None = None
    authorization_id: str | None = None
    history: list[ReviewHistoryEntry] = Field(default_factory=list)
