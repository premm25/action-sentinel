"""Run the real Streamlit script headlessly (streamlit.testing) against empty and populated databases."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import ROOT, make_action  # noqa: E402

APP = str(ROOT / "action_sentinel" / "dashboard" / "app.py")


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "dash.db"
    monkeypatch.setenv("SENTINEL_DB_PATH", str(path))
    monkeypatch.setenv("SENTINEL_ENV", "test")
    return path


def _run():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_empty_state(db):
    at = _run()
    infos = " ".join(i.value for i in at.info)
    assert "No decisions recorded yet" in infos and "No pending reviews" in infos
    assert "No evaluation runs stored" in infos
    assert [m.value for m in at.metric][0] == "0"


def test_dashboard_populated(db):
    import warnings

    from action_sentinel.config import load_settings
    from action_sentinel.service import Sentinel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Sentinel(load_settings())
    s.evaluate(make_action())
    s.evaluate(make_action(event_id="e2", idempotency_key="k2", target_resource="prod/checkout-api"))
    s.evaluate(make_action(event_id="e3", idempotency_key="k3", action_type="disable_audit_logging"))
    from action_sentinel.contracts import EvaluationRunConfig
    from action_sentinel.evaluation import ConfidenceLab
    s.store.save_evaluation(ConfidenceLab(s.settings).run(EvaluationRunConfig(
        name="dash", dataset_path="examples/evaluation_datasets/sentinel_synthetic_v1.jsonl",
        providers=["rules", "mock-a"], limit=8)))
    s.store.close()

    at = _run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Evaluated decisions"] == "3"
    assert metrics["APPROVE"] == "1" and metrics["HUMAN_REVIEW"] == "1" and metrics["BLOCK"] == "1"
    assert not any("No pending reviews" in i.value for i in at.info)
    assert any("Enter the selected reviewer" in w.value for w in at.warning)  # no key -> no actions

    # authenticate as the operator and approve the pending review through the dashboard
    at.text_input(key="hr_key").set_value("dev-operator-key").run()
    rid_button = [b for b in at.button if b.key and b.key.startswith("a_")][0]
    rid_button.click().run()
    assert not at.exception
    assert any("Approved" in s.value for s in at.success)



def test_vega_lab_tab_reads_saved_runs_only(db, tmp_path, monkeypatch):
    """Empty state without runs; renders an analysed run from saved files; never loads a model."""
    from action_sentinel.vegalab import vega_core
    from action_sentinel.vegalab.analysis import analyze
    from action_sentinel.vegalab.report import write_reports
    from action_sentinel.vegalab.runner import run_system
    from action_sentinel.vegalab.systems import VegaSystem
    from action_sentinel.vegalab.tasks import TASKS
    from action_sentinel.vegalab.testing import FakeVegaRunner

    def no_load(self):
        raise AssertionError("the dashboard must not load a model")
    monkeypatch.setattr(vega_core.LiveVegaRunner, "load", no_load)
    monkeypatch.setenv("SENTINEL_VEGALAB_DIR", str(tmp_path / "lab"))
    at = _run()
    assert any("No analysed Vega lab runs" in i.value for i in at.info)

    q = TASKS["sst2"].question().model_dump(mode="json")
    exs = [{"example_id": "sst2:test:%d" % i, "task": "sst2", "split": "test", "state": "s%d" % i, "question": q,
            "gold_label": ["negative", "positive"][i % 2], "slices": {}} for i in range(6)]
    run_dir = tmp_path / "lab" / "run1"
    run_system(VegaSystem(vega_core.VegaCore(FakeVegaRunner()), "off"), exs, run_dir, "run1", progress_every=0)
    write_reports(run_dir, analyze(run_dir), "test")
    at = _run()
    assert not any("No analysed Vega lab runs" in i.value for i in at.info)
    assert any("Accuracy and calibration by task" in h.value for h in at.subheader)
