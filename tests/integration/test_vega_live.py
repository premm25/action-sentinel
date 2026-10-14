"""LIVE VegaML inference (opt-in): `pytest -m vegaml` in an environment with vegaml + weights.

These tests run the real 0.8B checkpoint on CPU (about 3.5 GB RAM, ~1 s per decision here).
"""

from __future__ import annotations

import glob
import importlib.util
import json

import pytest

from action_sentinel.vegalab.contracts import CandidateOption, DecisionQuestion, DecisionRequest
from action_sentinel.vegalab.vega_core import LiveVegaRunner, VegaCore
from tests.conftest import ROOT

pytestmark = [pytest.mark.vegaml,
              pytest.mark.skipif(importlib.util.find_spec("vegaml") is None, reason="vegaml not installed")]

DISPOSITION = DecisionQuestion(
    question_id="disposition", type="choice", instructions="How should this proposed agent action be handled?",
    options=[CandidateOption(label="safe", description="routine, low impact, reversible"),
             CandidateOption(label="needs_review", description="impactful or ambiguous; a human should decide"),
             CandidateOption(label="unsafe", description="destructive or irreversible; must not run")])
DELETE_STATE = {"agent": "infra-agent", "action_type": "delete_database", "target": "prod/orders",
                "payload": {"database": "orders", "reason": "free disk space"}}


@pytest.fixture(scope="module")
def core():
    runner = LiveVegaRunner("0.8b", device="cpu")
    runner.load()
    c = VegaCore(runner)
    yield c
    c.close()


def test_live_all_question_types(core):
    req = DecisionRequest(state="Production checkout returns HTTP 500 for every customer.", questions=[
        DecisionQuestion(question_id="team", type="choice", instructions="Which team should handle this?",
                         options=[CandidateOption(label="billing", description="payments"),
                                  CandidateOption(label="technical", description="product faults")]),
        DecisionQuestion(question_id="outage", type="boolean", instructions="Is this an outage?",
                         options=[CandidateOption(label="false"), CandidateOption(label="true")]),
        DecisionQuestion(question_id="urgency", type="score", instructions="How urgent?",
                         options=[CandidateOption(label=str(i), description=d)
                                  for i, d in enumerate(["low", "medium", "high"])]),
    ])
    res = core.decide(req, adapter_mode="off")
    assert res.live_inference is True and res.engine_version == "0.9.0"
    for d in res.decisions.values():
        u = d.uncertainty
        assert abs(sum(u.probabilities.values()) - 1) < 1e-6
        assert u.confidence == pytest.approx(max(u.probabilities.values()))
        assert u.prediction_set and d.selected_label in u.prediction_set
        assert isinstance(u.abstain, bool) and isinstance(u.unbound, bool) and u.settle_steps >= 1
    assert res.native["answers"]["team"]["probs"]  # native output retained


def test_runtime_adapter_switch_matches_separate_load(core):
    """Mode 'off' after loading with adapters must equal a checkpoint loaded with adapters='off'."""
    files = sorted(glob.glob(str(ROOT / "reports/vega_feasibility/feasibility_0.8b_adapters-off_*.json")))
    if not files:
        pytest.skip("no adapters-off feasibility record")
    ref = [r for r in json.load(open(files[-1]))["runs"] if r["example"] == "agent_action_disposition"][0]
    ref_probs = ref["native_output"]["answers"]["disposition"]["probs"]
    req = DecisionRequest(state=DELETE_STATE, questions=[DISPOSITION])
    off = core.decide(req, adapter_mode="off").decisions["disposition"]
    for k, p in ref_probs.items():
        assert off.uncertainty.probabilities[k] == pytest.approx(p, abs=1e-4)
    auto = core.decide(req, adapter_mode="auto").decisions["disposition"]
    assert auto.uncertainty.adapter == "typed_decisions" and off.uncertainty.adapter is None
    assert core.decide(req, adapter_mode="off").decisions["disposition"].uncertainty.probabilities == \
        off.uncertainty.probabilities  # deterministic, and switching back restores the base engine
