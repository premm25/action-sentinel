"""Structured audit logging with sanitization, and sanitized JSONL export."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from action_sentinel import __version__
from action_sentinel.contracts import AuditEvent

SENSITIVE_KEY = re.compile(r"(pass(word)?|secret|token|api[_-]?key|authorization|credential|private[_-]?key|cookie)",
                           re.IGNORECASE)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
MAX_STRING = 2000
MAX_ITEMS = 200
REDACTED = "[REDACTED]"


def sanitize(value: Any, depth: int = 0, redact_emails: bool = False) -> Any:
    """Redact values under sensitive keys, truncate long strings and large collections."""
    if depth > 12:
        return "[TRUNCATED: depth]"
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= MAX_ITEMS:
                out["__truncated__"] = "%d more keys" % (len(value) - MAX_ITEMS)
                break
            out[str(k)] = REDACTED if SENSITIVE_KEY.search(str(k)) else sanitize(v, depth + 1, redact_emails)
        return out
    if isinstance(value, (list, tuple)):
        items = [sanitize(v, depth + 1, redact_emails) for v in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append("[TRUNCATED: %d more items]" % (len(value) - MAX_ITEMS))
        return items
    if isinstance(value, str):
        s = EMAIL.sub("[EMAIL]", value) if redact_emails else value
        return s if len(s) <= MAX_STRING else s[:MAX_STRING] + "...[TRUNCATED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize(str(value), depth + 1, redact_emails)


class AuditLogger:
    def __init__(self, store, policy_version: str | None = None):
        self.store = store
        self.policy_version = policy_version

    def emit(self, event_type: str, correlation_id: str, component: str, *, decision_id: str | None = None,
             actor: str | None = None, reason_codes: Iterable[str] = (), data: dict[str, Any] | None = None,
             policy_version: str | None = None) -> AuditEvent:
        event = AuditEvent(
            event_type=event_type, correlation_id=correlation_id, decision_id=decision_id, component=component,
            component_version=__version__, policy_version=policy_version or self.policy_version, actor=actor,
            reason_codes=list(reason_codes), data=sanitize(data or {}))
        return self.store.append_event(event)


def export_jsonl(events: Iterable[AuditEvent], path: str | Path, redact_emails: bool = True) -> int:
    """Write events as JSONL with a second sanitization pass (emails redacted by default)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for e in events:
            row = sanitize(json.loads(e.model_dump_json()), redact_emails=redact_emails)
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n
