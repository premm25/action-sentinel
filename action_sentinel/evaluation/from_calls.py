"""Labelled MCP tool calls (from `sluice sentinel export`) as a Confidence Lab dataset, to fit VegaML on.

Each exported row is one call a proxy sent to Sentinel: server, tool, arguments (secrets redacted) and the
verdicts. A person fills in ``label`` (safe, needs_review or unsafe); unlabelled rows are skipped. The
verdicts are kept out of the example's situation, so a provider never sees what Sluice or Sentinel decided.

Splits are assigned from a hash of the event id (60% calibration, 20% dev, 20% test), so re-running on a
grown file keeps every earlier call in the split it had. Fit on calibration, measure on test.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from action_sentinel.contracts import ActionProposal, DatasetExample
from action_sentinel.contracts.decisions import DISPOSITION_QUESTION

OUTCOME = {"safe": "APPROVE", "needs_review": "HUMAN_REVIEW", "unsafe": "BLOCK"}
_BAD = re.compile(r"[^A-Za-z0-9._:\-]")


def _ident(value: str) -> str:
    clean = _BAD.sub("-", value)[:120] or "x"
    return clean if clean[0].isalnum() else "x" + clean[:119]


def split_for(event_id: str) -> str:
    bucket = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket < 2 else "dev" if bucket < 4 else "calibration"


def example_from_call(row: dict[str, Any], version: str) -> DatasetExample:
    label = row["label"]
    if label not in OUTCOME:
        raise ValueError("label %r must be one of %s" % (label, ", ".join(OUTCOME)))
    event = _ident(str(row["event_id"]))
    server, tool = str(row["server"]), str(row["tool"])
    situation = ActionProposal(
        event_id=event, task_id=_ident(str(row.get("session") or "session")), agent_id="mcp-proxy",
        agent_version="sluice", action_type="mcp_tool_call",
        action_payload={"server": server, "tool": tool, "arguments": row.get("arguments") or {}},
        target_resource=("mcp/%s/%s" % (server, tool))[:512], source_system="mcp-proxy", evidence_refs=[],
        proposed_at=datetime.fromtimestamp(float(row.get("ts") or 0), UTC), context={}, idempotency_key=event)
    return DatasetExample(
        example_id=event, dataset_version=version, synthetic=False, split=split_for(event), situation=situation,
        question=DISPOSITION_QUESTION, expected_label=label, expected_outcome=OUTCOME[label],
        tags=["mcp", "server:%s" % server], notes=row.get("notes"))


def dataset_from_calls(rows: list[dict[str, Any]], version: str) -> tuple[list[DatasetExample], int]:
    """The labelled rows as examples, and how many rows were skipped for having no label yet."""
    out, skipped = [], 0
    for row in rows:
        if not row.get("label"):
            skipped += 1
            continue
        out.append(example_from_call(row, version))
    return out, skipped
