from __future__ import annotations

import json

import pytest

from sentinel.config import load_settings
from sentinel.contracts import EvaluationRunConfig
from sentinel.evaluation import (
    ConfidenceLab,
    DatasetError,
    LabError,
    calibration_profile,
    load_dataset,
    render_comparison,
    suite,
)
from tests.conftest import DATASET, ROOT

DS = "examples/evaluation_datasets/sentinel_synthetic_v1.jsonl"


@pytest.fixture(scope="module")
def lab():
    return ConfidenceLab(load_settings(ROOT / "config" / "sentinel.yaml", db_path=":memory:", environment="test"))


def test_dataset_is_versioned_synthetic_and_split():
    examples, digest = load_dataset(DATASET)
    assert len(digest) == 64 and all(e.synthetic for e in examples)
    assert {e.dataset_version for e in examples} == {"sentinel-synthetic-v1"}
    splits = {e.split for e in examples}
    assert splits == {"dev", "calibration", "test"}
    tags = {t for e in examples for t in e.tags}
    assert {"clear", "ambiguous", "unsupported", "contradictory_evidence", "missing_fields", "hard_violation",
            "provider_timeout", "high_impact", "human_review"} <= tags
    # mock scripts never leak into the proposal that real providers see
    assert all("mock_script" not in json.dumps(e.situation.model_dump(mode="json")) for e in examples)


def test_dataset_validation_errors(tmp_path):
    good = DATASET.read_text().splitlines()[0]
    p = tmp_path / "d.jsonl"
    p.write_text(good + "\n" + good + "\n")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(p)
    p.write_text('{"example_id": "x"}\n')
    with pytest.raises(DatasetError):
        load_dataset(p)
    other = json.loads(good)
    other["example_id"], other["dataset_version"] = "y", "v2"
    p.write_text(good + "\n" + json.dumps(other) + "\n")
    with pytest.raises(DatasetError, match="mixes versions"):
        load_dataset(p)
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "missing.jsonl")


def test_lab_run_full_sentinel(lab):
    r = lab.run(EvaluationRunConfig(name="full", dataset_path=DS, providers=["rules", "mock-a", "mock-b"]))
    assert r.n_examples == len(r.records) and all(rec.split == "test" for rec in r.records)
    g = r.governance_metrics
    assert g["hard_violation_blocked"]["value"] == 1.0  # policy precedence holds on every hard violation
    assert g["false_approval_rate"]["value"] == 0.0
    assert sum(g["outcome_counts"].values()) == r.n_examples
    assert r.provider_metrics["rules"]["calibration"] is None  # deterministic: no confidence, no ECE
    assert r.provider_metrics["mock-a"]["calibration"] is not None
    assert any("SIMULATED" in n for n in r.notes)


def test_lab_is_deterministic_and_paired(lab):
    cfg = EvaluationRunConfig(name="a", dataset_path=DS, providers=["rules", "mock-a", "mock-b"])
    r1, r2 = lab.run(cfg), lab.run(cfg)
    assert [x.outcome for x in r1.records] == [x.outcome for x in r2.records]
    assert r1.governance_metrics["outcome_counts"] == r2.governance_metrics["outcome_counts"]


def test_naive_baseline_misses_violations_policy_catches(lab):
    naive = lab.run(EvaluationRunConfig(name="n", dataset_path=DS, providers=["mock-a"], governance_mode="naive",
                                        use_validation=False))
    full = lab.run(EvaluationRunConfig(name="f", dataset_path=DS, providers=["rules", "mock-a", "mock-b"]))
    assert naive.governance_metrics["hard_violation_blocked"]["value"] < 1.0
    assert full.governance_metrics["false_approval_rate"]["k"] <= naive.governance_metrics["false_approval_rate"]["k"]


def test_suite_report_and_calibration_profile(lab):
    configs = suite(DS, "mock-a", "mock-b", ["test"], limit=10)
    reports = [lab.run(c) for c in configs]
    md = render_comparison(reports)
    assert "6_full_sentinel" in md and "ablation_no_validation" in md and "n/a" in md
    prof = calibration_profile(reports[-1], "mock-a")
    assert prof["dataset_version"] == "sentinel-synthetic-v1" and 0 <= prof["ece"] <= 1
    assert calibration_profile(reports[0], "rules") is None


def test_lab_errors(lab):
    with pytest.raises(LabError, match="unknown providers"):
        lab.run(EvaluationRunConfig(name="x", dataset_path=DS, providers=["nope"]))
    with pytest.raises(LabError, match="unavailable"):
        lab.run(EvaluationRunConfig(name="x", dataset_path=DS, providers=["vegaml"]))
    with pytest.raises(LabError, match="no examples"):
        lab.run(EvaluationRunConfig(name="x", dataset_path=DS, providers=["rules"], splits=[]))
