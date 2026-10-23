"""VegaML-first operation: saved heads, warm-up, question checks and `vega fit` (FakeVegaRunner, no model)."""

from __future__ import annotations

import pytest

from action_sentinel.contracts import DecisionQuestion, DecisionRequest
from action_sentinel.providers import ProviderRegistry, RuleBasedDecisionProvider
from action_sentinel.providers.vegaml_provider import VegaMLDecisionProvider
from tests.conftest import DATASET, ROOT, make_action

np = pytest.importorskip("numpy")


def req(question: DecisionQuestion | None = None, **kw) -> DecisionRequest:
    return DecisionRequest(correlation_id="c1", action=make_action(**kw), **({"question": question} if question else {}))


def head_provider(runner=None, **kw):
    from action_sentinel.vegalab.testing import FakeVegaRunner
    from action_sentinel.vegalab.vega_core import VegaCore
    runner = runner or FakeVegaRunner()
    cfg = dict(adapter_mode="off", framing="compact", readout="ttt", ttt_fit_dataset=str(DATASET))
    cfg.update(kw)
    return VegaMLDecisionProvider("vegaml-head", core=VegaCore(runner), **cfg), runner


# ------------------------------------------------------------------ head files

def test_head_file_round_trip_and_checks(tmp_path):
    from action_sentinel.vegalab.heads import HeadFileError, check_head, load_head, save_head
    arrays = {k: np.arange(4, dtype=np.float32) + i for i, k in enumerate(("mu", "sd", "c", "P", "s", "W", "b"))}
    meta = {"question_id": "disposition", "question_sha256": "abc", "labels": ["safe", "unsafe"],
            "framing": "compact", "checkpoint": "0.8b", "engine_sha256": "e" * 64}
    path = save_head(tmp_path / "h.npz", arrays, meta)
    got, m = load_head(path)
    assert all(np.array_equal(got[k], arrays[k]) for k in arrays)
    assert m["labels"] == ["safe", "unsafe"] and m["format"].startswith("action-sentinel-vegaml-head")
    check_head(m, framing="compact", checkpoint="0.8b", engine_sha256="e" * 64)
    for kw, needle in [(dict(framing="rich"), "framing"), (dict(checkpoint="4b"), "checkpoint"),
                       (dict(engine_sha256="f" * 64), "engine")]:
        args = dict(framing="compact", checkpoint="0.8b", engine_sha256="e" * 64)
        args.update(kw)
        with pytest.raises(HeadFileError, match=needle):
            check_head(m, **args)
    with pytest.raises(HeadFileError, match="vega fit"):
        load_head(tmp_path / "missing.npz")
    (tmp_path / "junk.npz").write_bytes(b"not a head")
    with pytest.raises(HeadFileError):
        load_head(tmp_path / "junk.npz")


def test_head_file_needs_its_metadata(tmp_path):
    from action_sentinel.vegalab.heads import HeadFileError, save_head
    arrays = {k: np.zeros(2) for k in ("mu", "sd", "c", "P", "s", "W", "b")}
    with pytest.raises(HeadFileError, match="question_sha256"):
        save_head(tmp_path / "h.npz", arrays, {"question_id": "q", "labels": [], "framing": "compact",
                                               "checkpoint": "0.8b"})


# ------------------------------------------------------------------ provider

def test_warm_up_fits_once_before_any_decision():
    p, runner = head_provider()
    detail = p.warm_up()
    assert len(runner.fit_calls) == 1 and "readout=ttt" in detail
    assert p.decide(req()).status == "ok"
    assert len(runner.fit_calls) == 1


def test_head_refuses_a_question_it_was_not_fitted_for():
    p, _ = head_provider()
    other = DecisionQuestion(question_id="disposition", instructions="Is this action reversible?",
                             categories={"safe": "yes", "needs_review": "maybe", "unsafe": "no"})
    r = p.decide(req(question=other))
    assert r.status == "error" and r.error.code == "TTT_HEAD_QUESTION_MISMATCH" and r.label is None


def test_saved_head_is_served_without_refitting(tmp_path):
    from action_sentinel.vegalab.testing import FakeVegaRunner
    p, _ = head_provider()
    path = p.save_head(str(tmp_path / "head.npz"))
    q, runner = head_provider(FakeVegaRunner(), ttt_fit_dataset=None, ttt_head_path=path)
    r = q.decide(req())
    assert r.status == "ok" and runner.fit_calls == [] and runner.installed == ["disposition"]
    assert q.fit_report["source"] == "file" and q.fit_report["examples"] == 28


def test_saved_head_for_another_framing_is_refused(tmp_path):
    from action_sentinel.vegalab.testing import FakeVegaRunner
    p, _ = head_provider()
    path = p.save_head(str(tmp_path / "head.npz"))
    q, runner = head_provider(FakeVegaRunner(), framing="rich", ttt_fit_dataset=None, ttt_head_path=path)
    r = q.decide(req())
    assert r.status == "unavailable" and r.error.code == "TTT_HEAD_INVALID" and runner.installed == []


def test_head_needs_a_dataset_or_a_saved_file():
    with pytest.raises(ValueError, match="ttt_head_path"):
        VegaMLDecisionProvider("v", readout="ttt")


# ------------------------------------------------------------------ Sentinel.warm_up

def test_sentinel_warm_up_reports_each_provider_and_audits_it():
    from action_sentinel.vegalab.testing import FakeVegaRunner
    from tests.conftest import make_sentinel
    sentinel = make_sentinel()
    good, _ = head_provider()
    bad, _ = head_provider(FakeVegaRunner(), ttt_fit_dataset=None, ttt_head_path="nowhere.npz")
    bad.name = "vegaml-broken"
    sentinel.registry.register(good)
    sentinel.registry.register(bad)
    out = sentinel.warm_up(["rules", "vegaml-head", "vegaml-broken"])
    assert set(out) == {"vegaml-head", "vegaml-broken"}  # rules has nothing to load
    assert out["vegaml-head"].startswith("ready") and out["vegaml-broken"].startswith("failed: TTT_HEAD_INVALID")
    events = [e for e in sentinel.store.list_events(limit=100) if e.event_type == "provider.warm_up"]
    assert sorted(e.reason_codes[0] for e in events) == ["FAILED", "OK"]


def test_vegaml_profile_runs_the_head_beside_the_rules():
    from action_sentinel.config import load_settings
    s = load_settings(ROOT / "config" / "sentinel-vegaml.yaml")
    assert s.active_providers == ["rules", "vegaml-head"] and s.warm_up is True
    reg = ProviderRegistry()
    reg.build(s.providers, s.path("."))
    head = reg.get("vegaml-head")
    assert head.readout == "ttt" and head.ttt_fit_split == "calibration"
    assert isinstance(reg.get("rules"), RuleBasedDecisionProvider)


# ------------------------------------------------------------------ action-sentinel vega fit

def test_vega_fit_refuses_the_test_split(tmp_path, capsys):
    from action_sentinel.cli import main
    rc = main(["vega", "fit", "--dataset", str(DATASET), "--split", "test", "--out", str(tmp_path / "h.npz")])
    assert rc == 2 and "test split" in capsys.readouterr().err
    assert not (tmp_path / "h.npz").exists()


def test_vega_fit_saves_a_loadable_head(tmp_path, monkeypatch, capsys):
    from action_sentinel.cli import main
    from action_sentinel.vegalab.heads import load_head
    from action_sentinel.vegalab.testing import FakeVegaRunner
    from action_sentinel.vegalab.vega_core import VegaCore
    core = VegaCore(FakeVegaRunner())
    monkeypatch.setattr(VegaMLDecisionProvider, "_load_core", lambda self: core)
    out = tmp_path / "heads" / "actions.npz"
    rc = main(["vega", "fit", "--dataset", str(DATASET), "--out", str(out), "--evaluate", "dev"])
    text = capsys.readouterr().out
    assert rc == 0 and "fitted on 28 calibration examples" in text and "dev split:" in text
    _, meta = load_head(out)
    assert meta["split"] == "calibration" and meta["framing"] == "compact" and meta["question_id"] == "disposition"
