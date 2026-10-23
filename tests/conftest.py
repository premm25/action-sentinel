from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_sentinel.contracts import ActionProposal  # noqa: E402

CORE_PACK = ROOT / "examples" / "rule_packs" / "core.yaml"
CLASSIFIER = ROOT / "examples" / "rule_packs" / "classifier.yaml"
DATASET = ROOT / "examples" / "evaluation_datasets" / "sentinel_synthetic_v1.jsonl"


def action_dict(**overrides) -> dict:
    base = {
        "event_id": "evt-1",
        "task_id": "task-1",
        "agent_id": "ops-agent",
        "agent_version": "1.2.0",
        "action_type": "restart_service",
        "action_payload": {"service": "checkout-api", "reason": "health check failing"},
        "target_resource": "staging/checkout-api",
        "source_system": "monitoring-agent",
        "evidence_refs": [{"ref_id": "alert-1", "kind": "alert", "stance": "supports",
                           "description": "health check failing for 5 minutes"}],
        "proposed_at": "2026-10-14T10:00:00+00:00",
        "context": {},
        "idempotency_key": "idem-1",
    }
    base.update(overrides)
    return base


def make_action(**overrides) -> ActionProposal:
    return ActionProposal.model_validate(action_dict(**overrides))


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)


class Clock:
    """A controllable clock for expiry tests."""

    def __init__(self, start: datetime | None = None):
        self.t = start or datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        from datetime import timedelta
        self.t = self.t + timedelta(seconds=seconds)


def make_sentinel(scripts: dict | None = None, active: list[str] | None = None, clock=None, adapter=None,
                  **settings_kw):
    """An in-memory Sentinel with the rule-based provider and two scripted mocks (mock-a, mock-b)."""
    from action_sentinel.config import load_settings
    from action_sentinel.providers import MockDecisionProvider, ProviderRegistry, RuleBasedDecisionProvider
    from action_sentinel.service import Sentinel
    from action_sentinel.storage import SQLiteStore

    settings = load_settings(ROOT / "config" / "sentinel.yaml", db_path=":memory:", environment="test",
                             active_providers=active if active is not None else ["rules"], **settings_kw)
    reg = ProviderRegistry()
    reg.register(RuleBasedDecisionProvider.from_file("rules", CLASSIFIER))
    scripts = scripts or {}
    reg.register(MockDecisionProvider("mock-a", script=scripts.get("mock-a", {}),
                                      default={"label": "safe", "confidence": 0.95}))
    reg.register(MockDecisionProvider("mock-b", script=scripts.get("mock-b", {}),
                                      default={"label": "safe", "confidence": 0.9}))
    return Sentinel(settings, store=SQLiteStore(":memory:"), registry=reg, clock=clock or Clock(),
                    key=b"test-signing-key-0123456789abcdef", adapter=adapter)


@pytest.fixture
def clock():
    return Clock()
