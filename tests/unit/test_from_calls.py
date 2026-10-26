"""Labelled MCP calls (sluice sentinel export) -> Confidence Lab dataset."""

from __future__ import annotations

import json

import pytest

from action_sentinel.cli import main
from action_sentinel.contracts import DatasetExample
from action_sentinel.evaluation.from_calls import dataset_from_calls, split_for


def row(n: int, tool: str, label: str | None, **extra) -> dict:
    return {"event_id": "sluice-%032x" % n, "ts": 1793000000.0 + n, "session": "client-1 (pid 42)",
            "server": "files", "tool": tool, "arguments": {"path": "/tmp/%d" % n}, "mode": "shadow",
            "sluice": "allow", "sluice_codes": [], "sentinel": "APPROVE", "sentinel_codes": [],
            "review_id": None, "review_state": None, "ran": True, "label": label, **extra}


def test_labelled_rows_become_examples_without_the_verdicts():
    examples, skipped = dataset_from_calls([row(1, "read_file", "safe"), row(2, "delete_file", None),
                                            row(3, "delete_file", "needs_review")], "mcp-test")
    assert skipped == 1 and [e.expected_outcome.value for e in examples] == ["APPROVE", "HUMAN_REVIEW"]
    e = examples[0]
    assert e.situation.action_type == "mcp_tool_call" and e.situation.target_resource == "mcp/files/read_file"
    assert e.situation.action_payload == {"server": "files", "tool": "read_file", "arguments": {"path": "/tmp/1"}}
    assert e.situation.context == {} and "APPROVE" not in e.situation.model_dump_json()  # no answer leakage
    assert e.situation.task_id == "client-1--pid-42-" and not e.synthetic


def test_splits_are_stable_and_bad_labels_are_refused():
    ids = ["sluice-%032x" % n for n in range(400)]
    splits = [split_for(i) for i in ids]
    assert splits == [split_for(i) for i in ids]
    assert 0.1 < splits.count("test") / 400 < 0.3 and 0.5 < splits.count("calibration") / 400 < 0.7
    with pytest.raises(ValueError, match="label"):
        dataset_from_calls([row(1, "x", "dangerous")], "v")


def test_cli_writes_a_dataset_vega_fit_can_read(tmp_path, capsys):
    src, out = tmp_path / "calls.jsonl", tmp_path / "ds.jsonl"
    src.write_text("".join(json.dumps(row(n, "t%d" % n, "unsafe" if n % 2 else "safe")) + "\n" for n in range(10)),
                   encoding="utf-8")
    assert main(["vega", "dataset", "--from", str(src), "--out", str(out)]) == 0
    rows = [DatasetExample.model_validate_json(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10 and {r.dataset_version for r in rows} == {"mcp-calls-v1"}
    assert "wrote 10 examples" in capsys.readouterr().out
