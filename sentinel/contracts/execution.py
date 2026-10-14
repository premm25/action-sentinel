"""Execution authorization grants and execution results."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue

from .common import StrictModel, new_id, utcnow


class AuthorizationGrant(StrictModel):
    """Permission to execute one exact action (bound by hash) under one policy version, until it expires."""

    grant_id: str = Field(default_factory=lambda: new_id("auth"))
    decision_id: str
    review_id: str | None = None
    correlation_id: str
    action_hash: str
    idempotency_key: str
    policy_version: str
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    issued_by: str  # "governance:auto" or "reviewer:<id>"
    approvals: list[str] = Field(default_factory=list)
    revoked: bool = False
    revoked_at: datetime | None = None
    revoked_reason: str | None = None


class IssuedAuthorization(StrictModel):
    grant: AuthorizationGrant
    token: str


class ExecutionResult(StrictModel):
    execution_id: str = Field(default_factory=lambda: new_id("exe"))
    grant_id: str | None
    decision_id: str | None
    correlation_id: str | None
    idempotency_key: str | None
    status: Literal["EXECUTED", "REJECTED", "DUPLICATE", "FAILED"]
    reason_codes: list[str] = Field(default_factory=list)
    adapter: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    output: dict[str, JsonValue] = Field(default_factory=dict)
    original_execution_id: str | None = None  # set on DUPLICATE
