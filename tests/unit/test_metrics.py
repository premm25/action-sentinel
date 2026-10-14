from __future__ import annotations

import math

import pytest

from action_sentinel.evaluation import metrics as M


def test_classification_hand_computed():
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "b", "b", None, "a"]
    m = M.classification(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"]["value"] == pytest.approx(2 / 5)
    assert m["selective_accuracy"]["value"] == pytest.approx(2 / 4)
    assert m["coverage"] == pytest.approx(4 / 5)
    pa = m["per_class"]["a"]
    assert pa["precision"] == pytest.approx(1 / 2) and pa["recall"] == pytest.approx(1 / 2)
    pb = m["per_class"]["b"]
    assert pb["precision"] == pytest.approx(1 / 2) and pb["recall"] == pytest.approx(1 / 2)
    pc = m["per_class"]["c"]
    assert pc["precision"] is None and pc["recall"] == 0.0 and pc["f1"] is None  # undefined, not zero
    assert m["macro_f1"] == pytest.approx(0.5) and m["macro_f1_classes"] == 2
    assert m["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_empty_inputs_are_undefined():
    m = M.classification([], [], ["a", "b"])
    assert m["accuracy"]["value"] is None and m["coverage"] is None and m["macro_f1"] is None
    assert M.calibration([], []) is None and M.risk_coverage([], []) is None
    assert M.wilson(0, 0) is None
    assert M.percentiles([]) == {"p50": None, "p95": None, "p99": None}


def test_ece_hand_computed():
    cal = M.calibration([0.9, 0.9, 0.6, 0.6], [True, False, True, True], bins=10)
    # bin 0.9: conf .9, acc .5 -> |.4|*.5 ; bin 0.6: conf .6, acc 1 -> |.4|*.5
    assert cal["ece"] == pytest.approx(0.4)
    assert sum(b["count"] for b in cal["reliability"]) == 4
    perfect = M.calibration([1.0, 1.0], [True, True])
    assert perfect["ece"] == pytest.approx(0.0) and perfect["reliability"][-1]["count"] == 2


def test_risk_coverage():
    rc = M.risk_coverage([0.9, 0.8, 0.7], [True, False, True])
    assert [p["risk"] for p in rc["curve"]] == pytest.approx([0.0, 0.5, 1 / 3])
    assert rc["aurc"] == pytest.approx((0 + 0.5 + 1 / 3) / 3)


def test_brier_and_log_loss():
    out = M.brier_and_log_loss([{"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}], ["a", "b"], ["a", "b"])
    assert out["brier"] == pytest.approx((0 + 0.5) / 2)
    assert out["log_loss"] == pytest.approx(-(math.log(1.0) + math.log(0.5)) / 2)
    zero = M.brier_and_log_loss([{"a": 0.0, "b": 1.0}], ["a"], ["a", "b"])
    assert zero["log_loss_clipped_examples"] == 1 and math.isfinite(zero["log_loss"])
    assert M.brier_and_log_loss([{"a": 1.0}], ["a"], ["a", "b"]) is None  # partial distribution
    assert M.brier_and_log_loss([None], ["a"], ["a", "b"]) is None


def test_wilson_and_percentiles():
    lo, hi = M.wilson(5, 10)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert M.wilson(0, 10)[0] == 0.0
    assert M.percentiles([1, 2, 3, 4])["p50"] == pytest.approx(2.5)
