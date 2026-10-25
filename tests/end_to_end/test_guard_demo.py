"""examples/guard_demo.py (the README recording) keeps showing all three outcomes."""

from __future__ import annotations

import runpy

from tests.conftest import ROOT


def test_guard_demo_shows_approve_block_and_review(capsys):
    runpy.run_path(str(ROOT / "examples" / "guard_demo.py"), run_name="guard_demo")["main"](0.0)
    out = capsys.readouterr().out
    for expected in ("APPROVE   ran: 50 log lines", "BLOCK     never ran: HARD_POLICY_VIOLATION",
                     "REVIEW    held for a human", "RESUMED   ran: web scaled to 8 replicas"):
        assert expected in out
