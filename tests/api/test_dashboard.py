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
