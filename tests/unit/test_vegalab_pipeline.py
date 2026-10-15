"""Confidence Lab v2 pipeline: datasets, leakage checks, runner, baselines, metrics. Offline (test double only)."""

from __future__ import annotations

import json
import math
import shutil

import pytest

from action_sentinel.vegalab import analysis as A
from action_sentinel.vegalab.datasets import _example, check_dataset, load_examples
from action_sentinel.vegalab.runner import run_system
from action_sentinel.vegalab.systems import MajorityBaseline, VegaSystem
from action_sentinel.vegalab.tasks import EXCLUDED, TASKS
from action_sentinel.vegalab.testing import FakeVegaRunner
from action_sentinel.vegalab.vega_core import VegaCore
from tests.conftest import ROOT

PUB = ROOT / "data" / "vegalab" / "pub-v1"
needs_data = pytest.mark.skipif(not (PUB / "manifest.json").exists(), reason="pub-v1 dataset not built")


# ------------------------------------------------------------------ tasks and datasets

def test_excluded_training_families_are_not_evaluated():
    names = " ".join(TASKS).lower()
    for fam in ("banking77", "clinc150", "apibank", "appliances", "esci", "when2call", "emotion", "ag_news"):
        assert fam not in names and fam in EXCLUDED


def test_task_questions_are_fixed_and_valid():
    for spec in TASKS.values():
        if spec.dynamic_labels:
            continue
        q = spec.question()
        assert q.type == spec.qtype and len(q.options) >= 2


def test_example_builder_rejects_invalid_gold():
    spec = TASKS["superglue_cb"]
    q = spec.question()
    labels = q.labels
    assert _example(spec, "test", "validation", 0, {"premise": "p", "hypothesis": "h", "label": 1}, q, labels)
    assert _example(spec, "test", "test", 0, {"premise": "p", "hypothesis": "h", "label": -1}, q, labels) is None
    assert _example(spec, "test", "test", 0, {"premise": "", "hypothesis": "", "label": 0}, q, labels) is not None


@needs_data
def test_built_dataset_passes_checks_and_records_provenance():
    report = check_dataset(PUB)
    assert report and all(v["passed"] for v in report.values())
    m = json.loads((PUB / "manifest.json").read_text(encoding="utf-8"))
    assert m["synthetic"] is False and m["caveats"]
    for name, t in m["tasks"].items():
        assert t["counts"]["test"] > 0 and t["sha256"]
        rows = load_examples(PUB / t["file"])
        assert all({"dataset", "config", "split", "row"} <= set(r["source"]) for r in rows)
        if not name.startswith("dbpedia14_k"):
            assert report[name]["distinct_questions"] == 1  # one fixed question: labels cannot leak per example


@needs_data
def test_leakage_check_detects_test_text_in_calibration(tmp_path):
    shutil.copy(PUB / "sst2.jsonl", tmp_path / "sst2.jsonl")
    rows = load_examples(tmp_path / "sst2.jsonl")
    test_row = next(r for r in rows if r["split"] == "test")
    leaked = dict(test_row, example_id="sst2:calibration:999999", split="calibration")
    with open(tmp_path / "sst2.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(leaked) + "\n")
    rep = check_dataset(tmp_path)["sst2"]
    assert rep["passed"] is False and rep["test_overlap_with_other_splits"]["calibration"] == 1


# ------------------------------------------------------------------ runner and systems

def _examples(task="sst2", n=4):
    spec = TASKS[task]
    q = spec.question()
    out = []
    for i in range(n):
        out.append({"example_id": "%s:test:%d" % (task, i), "task": task, "split": "test", "state": "text %d" % i,
                    "question": q.model_dump(mode="json"), "gold_label": q.labels[i % 2],
                    "slices": {"difficulty": "standard", "length": "short(<30w)", "label_space": "2"}})
    return out


def test_runner_records_resume_and_errors(tmp_path):
    exs = _examples()
    sys_ = VegaSystem(VegaCore(FakeVegaRunner()), "off")
    path = run_system(sys_, exs[:2], tmp_path, "run1", progress_every=0)
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["status"] for r in recs] == ["ok", "ok"] and recs[0]["native"]["probs"]
    run_system(sys_, exs, tmp_path, "run1", progress_every=0)          # resume: only the 2 new ones
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(recs) == 4 and len({r["example_id"] for r in recs}) == 4
    meta = json.loads((tmp_path / "vega-off.run.json").read_text())
    assert [s["skipped_already_done"] for s in meta["sessions"]] == [0, 2]
    assert meta["system_info"]["live"] is False  # the test double is labelled as such in run metadata

    broken = VegaSystem(VegaCore(FakeVegaRunner(raise_on_decide=RuntimeError("boom"))), "auto")
    p2 = run_system(broken, exs[:1], tmp_path, "run1", progress_every=0)
    r = json.loads(p2.read_text().splitlines()[0])
    assert r["status"] == "error" and r["error"]["code"] == "INFERENCE_FAILED" and r["predicted_label"] is None


def test_majority_uses_fit_data_only():
    fit = [dict(e, split="calibration") for e in _examples(n=5)]  # golds alternate: negative x3, positive x2
    m = MajorityBaseline(fit)
    test = _examples(n=1)[0]
    p = m.predict(test)
    assert p["predicted_label"] == "negative" and abs(sum(p["probabilities"].values()) - 1) < 1e-9
    assert p["probabilities"]["negative"] == pytest.approx(4 / 7)  # Laplace: (3+1)/(5+2)


# ------------------------------------------------------------------ analysis

def _rec(i, gold, pred, probs, **kw):
    return {"example_id": "t:test:%d" % i, "task": "t", "split": kw.pop("split", "test"), "system": kw.pop("system", "s"),
            "status": "ok", "gold_label": gold, "predicted_label": pred, "labels": ["a", "b"], "probabilities": probs,
            "confidence": max(probs.values()), "wall_ms": 10.0, "slices": {}, **kw}


def test_balanced_accuracy_and_group_metrics():
    assert A.balanced_accuracy(["a", "a", "a", "b"], ["a", "a", "a", "a"]) == pytest.approx(0.5)
    recs = [_rec(0, "a", "a", {"a": 0.9, "b": 0.1}, abstain=False, prediction_set=["a"], unbound=False),
            _rec(1, "b", "a", {"a": 0.6, "b": 0.4}, abstain=True, prediction_set=["a", "b"], unbound=False),
            {**_rec(2, "b", None, {"a": 0.5, "b": 0.5}), "status": "error", "probabilities": None}]
    m = A.group_metrics(recs)
    assert m["n"] == 3 and m["errors"] == 1 and m["accuracy"]["value"] == pytest.approx(1 / 3)
    assert m["coverage"] == pytest.approx(0.5) and m["selective_accuracy"]["value"] == 1.0
    assert m["conformal_coverage"]["value"] == 1.0 and m["mean_set_size"] == 1.5
    assert m["brier"] is not None and m["ece"] is not None


def test_missing_probabilities_are_not_scored():
    recs = [{**_rec(0, "a", "a", {"a": 1.0, "b": 0.0}), "probabilities": None}]
    m = A.group_metrics(recs)
    assert m["ece"] is None and m["brier"] is None and "probabilistic_note" in m
    assert m["abstain_rate"] is None  # no native flag -> not reported, not assumed False


def test_temperature_fit_reduces_overconfidence():
    cal = []
    for i in range(40):  # 60% accurate but always 0.99 confident
        gold = "a" if i % 5 < 3 else "b"
        cal.append(_rec(i, gold, "a", {"a": 0.99, "b": 0.01}, split="calibration"))
    T = A.fit_temperature(cal)
    assert T > 1.5
    before = A.group_metrics(cal)
    after = A.group_metrics(cal, T)
    assert after["log_loss"] < before["log_loss"] and after["ece"] < before["ece"]
    assert A.fit_temperature(cal[:5]) is None  # too few examples to fit


def test_paired_difference():
    a = [_rec(i, "a", "a", {"a": 0.9, "b": 0.1}) for i in range(10)]
    b = [_rec(i, "a", "a" if i < 5 else "b", {"a": 0.9, "b": 0.1}) for i in range(10)]
    d = A.paired_difference(a, b, resamples=500)
    assert d["diff"] == pytest.approx(0.5) and d["a_only_correct"] == 5 and d["ci95"][0] > 0


def test_ordinal_metrics():
    recs = [{**_rec(i, str(g), str(p), {"0": 0.2, "1": 0.2, "2": 0.6}), "labels": ["0", "1", "2"]}
            for i, (g, p) in enumerate([(0, 0), (0, 2), (2, 1)])]
    m = A.group_metrics(recs)
    assert m["ordinal_mae"] == pytest.approx(1.0) and m["within_1"] == pytest.approx(2 / 3)
    assert not math.isnan(m["brier"])


def test_regression_majority_uses_parent_prior_for_option_variants():
    fit = [dict(e, task="dbpedia14", split="calibration") for e in _examples("dbpedia14", n=4)]
    m = MajorityBaseline(fit)
    q = TASKS["dbpedia14"].question(TASKS["dbpedia14"].options[:2]).model_dump(mode="json")
    ex = {"example_id": "dbpedia14_k2:test:0", "task": "dbpedia14_k2", "split": "test", "state": "x",
          "question": q, "gold_label": "Company", "slices": {"variant_of": "dbpedia14"}}
    p = m.predict(ex)
    assert set(p["probabilities"]) == {"Company", "EducationalInstitution"}
    assert abs(sum(p["probabilities"].values()) - 1) < 1e-9


def test_regression_network_errors_trigger_offline_fallback():
    from action_sentinel.vegalab.vega_core import _network_error

    class ConnectTimeout(Exception):
        pass
    assert _network_error(ConnectTimeout("[WinError 10060] A connection attempt failed"))
    assert not _network_error(ValueError("backbone depth differs from training"))
