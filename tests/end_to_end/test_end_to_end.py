"""Complete workflows, run the way a user runs them (demo, real CLI subprocesses)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from action_sentinel.demo import run_demo
from tests.conftest import ROOT, action_dict


def cli(*args, db, check=True):
    env = {**os.environ, "SENTINEL_ENV": "dev", "PYTHONWARNINGS": "ignore"}
    p = subprocess.run([sys.executable, "-m", "action_sentinel", "--db", str(db), *args], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    if check:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def test_demo_runs_offline_and_all_checks_pass(tmp_path):
    db = tmp_path / "demo.db"
    assert run_demo(db_path=str(db), quiet=True) == 0
    assert db.exists()  # records kept for the dashboard when a path is given


def test_cli_decision_to_execution_workflow(tmp_path):
    db = tmp_path / "cli.db"
    actions = tmp_path / "actions.jsonl"
    rows = [action_dict(event_id="c1", idempotency_key="k1"),
            action_dict(event_id="c2", idempotency_key="k2", target_resource="prod/checkout-api"),
            action_dict(event_id="c3", idempotency_key="k3", action_type="disable_audit_logging"),
            {"event_id": "bad"}]
    actions.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = cli("evaluate", str(actions), "-v", db=db, check=False)
    assert out.returncode == 2  # the malformed row is rejected, the rest are evaluated
    assert "c1" in out.stdout and "APPROVE" in out.stdout
    assert "HUMAN_REVIEW" in out.stdout and "BLOCK" in out.stdout and "REJECTED (schema)" in out.stdout

    listing = cli("reviews", "list", db=db).stdout
    review_id = listing.split()[0]
    denied = cli("reviews", "reject", review_id, "--reviewer", "olivia", "--reason", "", db=db, check=False)
    assert denied.returncode == 1 and "REASON_REQUIRED" in denied.stderr
    approved = cli("reviews", "approve", review_id, "--reviewer", "olivia", "--reason", "ok", db=db)
    assert "authorization token" in approved.stdout

    export = tmp_path / "audit.jsonl"
    cli("audit-export", "--out", str(export), db=db)
    types = [json.loads(line)["event_type"] for line in export.read_text().splitlines()]
    assert "review.approved" in types and types.count("governance.decision") == 3


def test_cli_rule_validation_and_providers(tmp_path):
    db = tmp_path / "x.db"
    assert "ok (core@1.0.0" in cli("validate-rules", db=db).stdout
    bad = tmp_path / "bad.yaml"
    bad.write_text("pack_id: bad\nversion: 1.0.0\nprovenance: t\nrules:\n  - id: X1\n    description: d\n"
                   "    severity: high\n    effect: block\n    when: {field: a, op: exec, value: 1}\n"
                   "    version: '1'\n    provenance: t\n")
    out = cli("validate-rules", str(bad), db=db, check=False)
    assert out.returncode == 1 and "INVALID_CONDITION" in out.stdout
    prov = cli("providers", db=db).stdout
    assert "rules" in prov and "vegaml" in prov


def test_cli_lab_suite_show_export(tmp_path):
    db = tmp_path / "lab.db"
    out_dir = tmp_path / "rep"
    cli("lab", "suite", "--limit", "12", "--out", str(out_dir), db=db)
    report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "6_full_sentinel" in report and "SIMULATED" in report
    runs = cli("lab", "show", db=db).stdout.split()
    run_id = runs[0]
    shown = json.loads(cli("lab", "show", run_id, db=db).stdout)
    assert shown["run_id"] == run_id and shown["dataset_version"] == "sentinel-synthetic-v1"
    md = tmp_path / "export.md"
    cli("lab", "export", run_id, "--out", str(md), db=db)
    assert "Governance outcomes" in md.read_text(encoding="utf-8")
    cal = cli("lab", "calibrate", "--provider", "mock-a", db=db).stdout
    assert json.loads(cal)["mock-a"]["dataset_version"] == "sentinel-synthetic-v1"
    assert cli("lab", "calibrate", "--provider", "rules", db=db, check=False).returncode == 1
