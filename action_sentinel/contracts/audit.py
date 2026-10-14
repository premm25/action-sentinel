"""Structured audit events."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, JsonValue

from .common import StrictModel, new_id, utcnow


class AuditEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    seq: int | None = None  # assigned by the store, strictly increasing
    event_type: str
    correlation_id: str
    decision_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    component: str
    component_version: str
    policy_version: str | None = None
    actor: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    data: dict[str, JsonValue] = Field(default_factory=dict)
