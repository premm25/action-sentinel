"""The proposed agent action and the evidence attached to it."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator

from .common import StrictModel, canonical_json, sha256_hex

MAX_PAYLOAD_BYTES = 64 * 1024
MAX_NESTING_DEPTH = 10

_ID = r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$"


def _depth(value: Any, level: int = 0) -> int:
    if isinstance(value, dict):
        return max([level + 1] + [_depth(v, level + 1) for v in value.values()])
    if isinstance(value, list):
        return max([level + 1] + [_depth(v, level + 1) for v in value])
    return level


class EvidenceRef(StrictModel):
    """A pointer to supporting material. `stance` defaults to unknown: evidence is never assumed to support."""

    ref_id: str = Field(pattern=_ID)
    kind: str = Field(min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=2000)
    uri: str | None = Field(default=None, max_length=2000)
    stance: Literal["supports", "contradicts", "neutral", "unknown"] = "unknown"


class ActionProposal(StrictModel):
    """An action an agent wants to take. A proposal is a request, never an authorization."""

    event_id: str = Field(pattern=_ID)
    task_id: str = Field(pattern=_ID)
    agent_id: str = Field(pattern=_ID)
    agent_version: str = Field(min_length=1, max_length=64)
    action_type: str = Field(pattern=r"^[a-z][a-z0-9_.]{0,63}$")
    action_payload: dict[str, JsonValue] = Field(default_factory=dict)
    target_resource: str = Field(min_length=1, max_length=512)
    source_system: str = Field(pattern=_ID)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=100)
    proposed_at: datetime
    context: dict[str, JsonValue] = Field(default_factory=dict)
    idempotency_key: str = Field(pattern=_ID)

    @field_validator("proposed_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("proposed_at must be timezone-aware")
        return v

    @field_validator("action_payload", "context")
    @classmethod
    def _bounded(cls, v: dict) -> dict:
        if len(json.dumps(v, default=str).encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ValueError("must serialize to at most %d bytes" % MAX_PAYLOAD_BYTES)
        if _depth(v) > MAX_NESTING_DEPTH:
            raise ValueError("nesting deeper than %d levels" % MAX_NESTING_DEPTH)
        return v

    @field_validator("evidence_refs")
    @classmethod
    def _unique_refs(cls, v: list[EvidenceRef]) -> list[EvidenceRef]:
        ids = [e.ref_id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ref_id values must be unique")
        return v

    def action_hash(self) -> str:
        """Hash of the fields that define *what* will be executed. Authorization is bound to it."""
        return sha256_hex(canonical_json({
            "action_type": self.action_type,
            "action_payload": self.action_payload,
            "target_resource": self.target_resource,
            "idempotency_key": self.idempotency_key,
            "agent_id": self.agent_id,
            "source_system": self.source_system,
        }))
