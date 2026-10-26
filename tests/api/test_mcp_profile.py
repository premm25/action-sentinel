"""config/sentinel-mcp.yaml: MCP tool calls from a proxy (sluice), as they arrive over the API."""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

from action_sentinel.api import create_app
from action_sentinel.config import load_settings
from action_sentinel.service import Sentinel
from action_sentinel.storage import SQLiteStore
from tests.conftest import ROOT

PROXY = {"X-API-Key": "dev-mcp-key"}


@pytest.fixture
def client():
    settings = load_settings(ROOT / "config" / "sentinel-mcp.yaml", db_path=":memory:", environment="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Sentinel(settings, store=SQLiteStore(":memory:"), key=b"test-signing-key-0123456789abcdef")
    with TestClient(create_app(settings, service=s)) as c:
        yield c


def call(server: str, tool: str, arguments: dict, n: int = 1) -> dict:
    """The proposal sluice sends for one tools/call."""
    return {"action": {
        "event_id": "sluice-%d" % n, "task_id": "session-1", "agent_id": "mcp-proxy", "agent_version": "sluice-0.1",
        "action_type": "mcp_tool_call", "action_payload": {"server": server, "tool": tool, "arguments": arguments},
        "target_resource": "mcp/%s/%s" % (server, tool), "source_system": "mcp-proxy", "evidence_refs": [],
        "proposed_at": "2026-10-26T12:00:00+00:00", "context": {"proxy": "sluice"},
        "idempotency_key": "sluice-%d" % n}}


@pytest.mark.parametrize("server,tool,arguments,outcome,code", [
    ("notes", "list_notes", {}, "APPROVE", None),
    ("github", "search_issues", {"q": "bug"}, "APPROVE", None),
    ("files", "delete_file", {"path": "/tmp/x"}, "HUMAN_REVIEW", None),
    ("slack", "post_message", {"channel": "#ops", "text": "hi"}, "HUMAN_REVIEW", None),
    ("weird", "frobnicate", {}, "HUMAN_REVIEW", None),                       # unknown tools need a person
    ("db", "run_query", {"sql": "DROP TABLE users;"}, "BLOCK", "RULE_MCP_DESTRUCTIVE_SQL"),
    ("shell", "exec", {"command": "rm -rf /var/data"}, "BLOCK", "RULE_MCP_RECURSIVE_DELETE"),
    ("http", "fetch", {"url": "https://x", "api_key": "sk-123"}, "BLOCK", "RULE_MCP_SECRET_IN_ARGUMENTS"),
])
def test_mcp_calls_get_the_expected_outcome(client, server, tool, arguments, outcome, code):
    r = client.post("/decisions/assess", json=call(server, tool, arguments), headers=PROXY)
    assert r.status_code == 200, r.text
    d = r.json()["decision"]
    assert d["status"] == outcome, d["reason_codes"]
    if code:
        assert code in d["reason_codes"]


def test_the_proxy_principal_may_only_submit_as_itself(client):
    body = call("notes", "list_notes", {})
    body["action"]["agent_id"] = "someone-else"
    assert client.post("/decisions/assess", json=body, headers=PROXY).status_code == 403
