"""Shared base types for every Sentinel contract."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for all contracts: unknown fields are rejected, values are validated on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_ORDER[self]


_SEVERITY_ORDER = {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3, Severity.CRITICAL: 4}


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _RISK_ORDER[self]

    @classmethod
    def max(cls, *levels: "RiskLevel") -> "RiskLevel":
        return max(levels, key=lambda lv: lv.rank)


_RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}


class ApprovalLevel(str, Enum):
    """Reviewer authority. A reviewer may act on a review whose required level is <= their own."""

    NONE = "none"
    OPERATOR = "operator"
    SENIOR = "senior"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return _APPROVAL_ORDER[self]


_APPROVAL_ORDER = {ApprovalLevel.NONE: 0, ApprovalLevel.OPERATOR: 1, ApprovalLevel.SENIOR: 2, ApprovalLevel.ADMIN: 3}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid.uuid4().hex)


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for hashing: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
