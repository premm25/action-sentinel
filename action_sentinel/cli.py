"""Command-line interface. Run `action-sentinel --help` (or `python -m action_sentinel --help`)."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from action_sentinel.config import Settings, load_settings
from action_sentinel.contracts import ActionProposal, EvaluationRunConfig


def _settings(args) -> Settings:
    over: dict[str, Any] = {}
    if getattr(args, "db", None):
        over["db_path"] = args.db
    return load_settings(args.config, **over)


def _service(args):
    from action_sentinel.service import Sentinel
    warnings.simplefilter("ignore", UserWarning)  # ephemeral-key warning is expected for local CLI use
    return Sentinel(_settings(args))


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ commands
def cmd_validate_rules(args) -> int:
    from action_sentinel.validation import check_pack_file, check_pack_set
    settings = _settings(args)
    paths = args.paths or [str(settings.path(p)) for p in settings.rule_packs]
    packs, errors = [], 0
    for p in paths:
        pack, issues = check_pack_file(p)
        status = "INVALID" if pack is None or any(i.level == "error" for i in issues) else "ok"
        print("%s: %s%s" % (p, status, (" (%s, %d rules)" % (pack.ref, len(pack.rules))) if pack else ""))
        for i in issues:
            print("  " + str(i))
        errors += sum(1 for i in issues if i.level == "error")
        if pack is not None:
            packs.append(pack)
    for i in check_pack_set(packs):
        print("  " + str(i))
        errors += i.level == "error"
    return 1 if errors else 0


def cmd_providers(args) -> int:
    s = _service(args)
    active = set(s.settings.active_providers)
    for h in s.provider_health():
        print("%-10s %-11s %-13s %-9s %s%s" % (h.name, h.provider_type, h.kind, "available" if h.available
                                               else "UNAVAILABLE", h.detail, "  [active]" if h.name in active else ""))
    return 0


def _read_actions(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    if text.startswith("{") and "\n{" not in text:
        return [json.loads(text)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def cmd_evaluate(args) -> int:
    from action_sentinel.service import DuplicateEventConflict
    s = _service(args)
    providers = args.providers.split(",") if args.providers else None
    rc = 0
    for raw in _read_actions(args.file):
        try:
            action = ActionProposal.model_validate(raw)
        except ValidationError as exc:
            print("REJECTED (schema): %s" % "; ".join("%s: %s" % (".".join(map(str, e["loc"])), e["msg"])
                                                      for e in exc.errors()))
            rc = 2
            continue
        try:
            out = s.evaluate(action, providers=providers, strategy=args.strategy, actor="cli")
        except DuplicateEventConflict as exc:
            print("CONFLICT: %s" % exc)
            rc = 2
            continue
        d = out.decision
        print("%-28s %-13s risk=%-8s %s%s" % (d.event_id, d.status.value, d.risk_level.value if d.risk_level else "-",
                                              ", ".join(d.reason_codes), " (duplicate)" if out.duplicate else ""))
        if args.verbose:
            for r in d.provider_results:
                print("    provider %-8s %-10s label=%s conf=%s" % (r.provider_name, r.status, r.label, r.confidence))
            print("    decision_id=%s review_id=%s" % (d.decision_id, d.review_id))
            if out.authorization:
                print("    authorization token: %s" % out.authorization.token)
    return rc


def _lab(args):
    from action_sentinel.evaluation import ConfidenceLab
    warnings.simplefilter("ignore", UserWarning)
    return ConfidenceLab(_settings(args))


def _save_reports(args, reports, title: str) -> Path:
    from action_sentinel.evaluation import render_comparison
    from action_sentinel.storage import SQLiteStore
    settings = _settings(args)
    out = Path(args.out) if args.out else settings.path(settings.report_dir) / reports[0].run_id
    out.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(settings.path(settings.db_path))
    for r in reports:
        store.save_evaluation(r)
        (out / ("%s.json" % r.config.name)).write_text(r.model_dump_json(indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(render_comparison(reports, title), encoding="utf-8")
    return out


def cmd_lab_suite(args) -> int:
    from action_sentinel.evaluation import suite
    lab = _lab(args)
    splits = args.splits.split(",")
    extra = args.extra_naive.split(",") if args.extra_naive else ["rules"]
    configs = suite(args.dataset, args.model, args.second or None, splits, args.limit, extra)
    reports = []
    for c in configs:
        print("running %s ..." % c.name, file=sys.stderr, flush=True)
        reports.append(lab.run(c))
    out = _save_reports(args, reports, args.title or "Confidence Lab: %s" % args.model)
    print("wrote %s" % (out / "REPORT.md"))
    return 0


def cmd_lab_run(args) -> int:
    lab = _lab(args)
    cfg = EvaluationRunConfig(name=args.name, dataset_path=args.dataset, splits=args.splits.split(","),
                              providers=args.providers.split(","), governance_mode=args.mode,
                              use_validation=not args.no_validation, use_disagreement=not args.no_disagreement,
                              use_uncertainty=not args.no_uncertainty, use_review=not args.no_review,
                              strategy=args.strategy, limit=args.limit)
    r = lab.run(cfg)
    out = _save_reports(args, [r], "Confidence Lab: %s" % args.name)
    print("run_id=%s wrote %s" % (r.run_id, out / "REPORT.md"))
    return 0


def cmd_lab_show(args) -> int:
    from action_sentinel.storage import SQLiteStore
    settings = _settings(args)
    store = SQLiteStore(settings.path(settings.db_path))
    if not args.run_id:
        for row in store.list_evaluations():
            print("%(run_id)s  %(name)-32s n=%(n_examples)-4s %(created_at)s" % row)
        return 0
    r = store.get_evaluation(args.run_id)
    if r is None:
        print("no such run", file=sys.stderr)
        return 1
    _print({"run_id": r.run_id, "config": r.config.model_dump(), "dataset_version": r.dataset_version,
            "governance": {k: v for k, v in r.governance_metrics.items()},
            "providers": {n: {"accuracy": m["classification"]["accuracy"], "ece": (m.get("calibration") or {}).get("ece")}
                          for n, m in r.provider_metrics.items()}, "notes": r.notes})
    return 0


def cmd_lab_export(args) -> int:
    from action_sentinel.evaluation import render_comparison
    from action_sentinel.storage import SQLiteStore
    settings = _settings(args)
    store = SQLiteStore(settings.path(settings.db_path))
    reports = [store.get_evaluation(rid) for rid in args.run_ids]
    if any(r is None for r in reports):
        print("unknown run id", file=sys.stderr)
        return 1
    Path(args.out).write_text(render_comparison(reports), encoding="utf-8")  # type: ignore[arg-type]
    print("wrote %s" % args.out)
    return 0


def cmd_lab_calibrate(args) -> int:
    from action_sentinel.evaluation import calibration_profile
    lab = _lab(args)
    r = lab.run(EvaluationRunConfig(name="calibration_%s" % args.provider, dataset_path=args.dataset,
                                    splits=args.splits.split(","), providers=[args.provider], governance_mode="naive",
                                    use_validation=False, limit=args.limit))
    prof = calibration_profile(r, args.provider)
    if prof is None:
        print("provider %s reports no confidence; calibration cannot be measured" % args.provider)
        return 1
    _print({args.provider: prof})
    print("# add under calibration_profiles: in the config to use it", file=sys.stderr)
    return 0


def cmd_reviews(args) -> int:
    s = _service(args)
    if args.action == "list":
        for r in s.store.list_reviews(state=args.state):
            print("%s %-10s level=%-8s decision=%s %s" % (r.review_id, r.state.value, r.required_level.value,
                                                         r.decision_id, ",".join(r.reason_codes)))
        return 0
    from action_sentinel.governance import ReviewError
    try:
        if args.action == "approve":
            out = s.approve_review(args.review_id, args.reviewer, args.reason)
            print("APPROVED; authorization token: %s" % out.authorization.token)
        elif args.action == "reject":
            print("REJECTED: %s" % s.reject_review(args.review_id, args.reviewer, args.reason).review_id)
        elif args.action == "override":
            out = s.override_review(args.review_id, args.reviewer, args.reason, json.loads(args.changes))
            print("OVERRIDDEN; new decision %s (%s); token: %s" % (out.decision.decision_id,
                                                                   out.decision.status.value, out.authorization.token))
    except ReviewError as exc:
        print("%s: %s" % (exc.code, exc), file=sys.stderr)
        return 1
    return 0


def cmd_audit_export(args) -> int:
    from action_sentinel.audit import export_jsonl
    s = _service(args)
    n = export_jsonl(s.store.list_events(correlation_id=args.correlation, limit=args.limit), args.out)
    print("exported %d sanitized events to %s" % (n, args.out))
    return 0


def cmd_demo(args) -> int:
    from action_sentinel.demo import run_demo
    return run_demo(db_path=args.db, quiet=args.quiet)


def cmd_serve(args) -> int:
    import uvicorn

    from action_sentinel.api.app import create_app
    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


# ------------------------------------------------------------------ Vega decision lab
def cmd_vega_check(args) -> int:
    import importlib.util
    ok = True
    for mod in ("vegaml", "torch", "transformers", "pyarrow"):
        found = importlib.util.find_spec(mod) is not None
        ver = getattr(__import__(mod), "__version__", "?") if found else "-"
        print("%-13s %-10s %s" % (mod, "installed" if found else "MISSING", ver))
        ok &= found or mod == "pyarrow"
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for repo in ("models--nandakishorm--vega-08b-public-intents", "models--Qwen--Qwen3.5-0.8B"):
        print("%-48s %s" % (repo, "cached" if (hub / repo).exists() else "not cached (downloads on first load)"))
    if args.load:
        from action_sentinel.vegalab.vega_core import LiveVegaRunner
        r = LiveVegaRunner(args.model, device=args.device)
        r.load()
        _print(r.info())
    return 0 if ok else 1


def _vega_question(args) -> dict:
    opts = []
    for item in args.option:
        label, _, desc = item.partition("=")
        opts.append({"label": label.strip(), "description": desc.strip() or None})
    return {"question_id": "answer", "type": args.type, "instructions": args.instructions, "options": opts}


def cmd_vega_decide(args) -> int:
    from action_sentinel.vegalab.contracts import VegaInferenceError
    from action_sentinel.vegalab.vega_core import LiveVegaRunner, VegaCore
    state = json.loads(args.state) if args.state.strip().startswith("{") else args.state
    runner = LiveVegaRunner(args.model, device=args.device)
    core = VegaCore(runner, model_id="vegaml-%s" % args.model)
    try:
        runner.load()
        res = core.decide({"state": state, "questions": [_vega_question(args)]}, adapter_mode=args.adapters)
    except VegaInferenceError as exc:
        print("%s: %s" % (exc.error.code, exc.error.message), file=sys.stderr)
        return 1
    d = res.decisions["answer"]
    u = d.uncertainty
    print("decision: %s   confidence %s   abstain=%s unbound=%s   set=%s   adapter=%s   %.0f ms"
          % (d.selected_label, "%.3f" % u.confidence if u.confidence is not None else "n/a", u.abstain, u.unbound,
             u.prediction_set, u.adapter, res.wall_ms))
    if args.json:
        print(res.model_dump_json(indent=2))
    return 0


def cmd_vega_fit(args) -> int:
    """Fit VegaML's few-shot head on labelled actions and save it for `ttt_head_path`."""
    from action_sentinel.contracts import DatasetExample
    from action_sentinel.providers import ProviderFailure
    from action_sentinel.providers.vegaml_provider import VegaMLDecisionProvider
    if args.split == "test":
        print("refusing to fit on the test split: it is for measuring, never for fitting", file=sys.stderr)
        return 2
    from action_sentinel.resources import resolve
    args.dataset = str(resolve(Path.cwd(), args.dataset))
    try:
        provider = VegaMLDecisionProvider("vegaml-head", checkpoint=args.model, device=args.device,
                                          adapter_mode="off", framing=args.framing, readout="ttt",
                                          ttt_fit_dataset=args.dataset, ttt_fit_split=args.split)
        provider.warm_up()
        out = provider.save_head(args.out)
    except ProviderFailure as exc:
        print("%s: %s" % (exc.code, exc), file=sys.stderr)
        return 1
    rep = provider.fit_report or {}
    cv = rep.get("cv_accuracy")
    print("fitted on %s %s examples; cross-validated accuracy %s (chance %s)" % (
        rep.get("examples"), args.split, "%.3f" % cv if cv is not None else "unknown (too few examples)",
        rep.get("chance")))
    print("saved %s" % out)
    if args.evaluate:
        from action_sentinel.contracts import DecisionRequest
        rows = [DatasetExample.model_validate_json(line) for line in
                Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [r for r in rows if r.split == args.evaluate]
        correct = answered = 0
        for r in rows:
            res = provider.decide(DecisionRequest(correlation_id="fit-eval", action=r.situation, question=r.question))
            if res.label is not None:
                answered += 1
                correct += res.label == r.expected_label
        print("%s split: %d/%d correct (%s)" % (args.evaluate, correct, len(rows),
                                                "%.3f" % (correct / len(rows)) if rows else "no examples"))
    print("serve it with: providers -> vegaml-head -> config: {readout: ttt, framing: %s, ttt_head_path: %s}"
          % (args.framing, out))
    return 0


def cmd_vega_data(args) -> int:
    from action_sentinel.vegalab.datasets import build_datasets, check_dataset
    if args.action == "build":
        folder = build_datasets(args.version, args.root, tasks=args.tasks.split(",") if args.tasks else None,
                                n_test=args.n_test, n_cal=args.n_cal, n_dev=args.n_dev, seed=args.seed)
        print("built %s" % folder)
    else:
        folder = Path(args.root) / args.version
    rep = check_dataset(folder)
    for task, r in rep.items():
        print("%-20s n=%-5d %s  overlap=%s" % (task, r["n"], "ok" if r["passed"] else "FAILED",
                                               r["test_overlap_with_other_splits"]))
    return 0 if rep and all(r["passed"] for r in rep.values()) else 1


def cmd_vega_run(args) -> int:
    from action_sentinel.vegalab.experiment import run_experiment
    out = run_experiment(args.dataset, args.out, systems=args.systems.split(",") if args.systems else None,
                         tasks=args.tasks.split(",") if args.tasks else None, splits=tuple(args.splits.split(",")),
                         limit=args.limit, model=args.model, device=args.device)
    print("predictions in %s; next: action-sentinel vega analyze %s" % (out, out))
    return 0


DEFAULT_COMPARE = [("vega-off", "backbone-optscore"), ("vega-auto", "vega-off"), ("vega-off", "majority")]


def cmd_vega_analyze(args) -> int:
    from action_sentinel.vegalab.analysis import analyze
    from action_sentinel.vegalab.report import write_reports
    compare = [tuple(c.split(":", 1)) for c in args.compare] if args.compare else DEFAULT_COMPARE
    res = analyze(args.run_dir, split=args.split, compare=compare)  # type: ignore[arg-type]
    for path in write_reports(args.run_dir, res, args.title or "Vega lab results"):
        print("wrote %s" % path)
    return 0


def cmd_vega_export(args) -> int:
    from action_sentinel.vegalab.analysis import analyze
    from action_sentinel.vegalab.report import summary_rows, to_csv
    res = analyze(args.run_dir, split=args.split)
    text = to_csv(res) if args.format == "csv" else json.dumps(summary_rows(res), indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("wrote %s" % args.out)
    else:
        print(text)
    return 0


def _failure_kind(kind: str, min_conf: float):
    if kind == "error":
        return lambda r: r["status"] == "error"
    if kind == "abstained":
        return lambda r: r.get("abstain") is True
    if kind == "wrong":
        return lambda r: r["status"] == "ok" and r["predicted_label"] != r["gold_label"]
    return lambda r: (r["status"] == "ok" and r["predicted_label"] != r["gold_label"]
                      and (r.get("confidence") or 0) >= min_conf)


def cmd_vega_failures(args) -> int:
    from action_sentinel.vegalab.analysis import load_predictions
    rows = [r for r in load_predictions(args.run_dir) if r["system"] == args.system and r["split"] == args.split
            and (not args.task or r["task"] == args.task)]
    keep = _failure_kind(args.kind, args.min_confidence)
    sel = [r for r in rows if keep(r)]
    print("%d of %d %s predictions are %s" % (len(sel), len(rows), args.system, args.kind))
    for r in sel[: args.limit]:
        conf = "%.3f" % r["confidence"] if r.get("confidence") is not None else "n/a"
        print("%-36s gold=%-14s pred=%-14s conf=%s abstain=%s %s" % (
            r["example_id"], r["gold_label"], r.get("predicted_label"), conf, r.get("abstain"),
            ((r.get("error") or {}).get("message") or "")[:80]))
    return 0


def cmd_init(args) -> int:
    """Copy the shipped settings, rule packs, dataset and sample actions into a project, to edit."""
    from action_sentinel.resources import copy_bundled
    written, skipped = copy_bundled(args.dir, force=args.force)
    for f in written:
        print("wrote   %s" % f)
    for f in skipped:
        print("exists  %s (kept; --force overwrites)" % f)
    print()
    print("Next: edit examples/rule_packs/core.yaml for your agent's actions and targets, then")
    print("  action-sentinel validate-rules")
    print("  action-sentinel demo")
    print("config/sentinel.yaml uses publicly known development API keys: replace the principals before "
          "exposing the API.")
    return 0


def cmd_dashboard(args) -> int:
    import subprocess
    app = Path(__file__).parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--server.address", "127.0.0.1", "--", "--config", str(args.config or "")]
    return subprocess.call(cmd)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ds_default = "examples/evaluation_datasets/sentinel_synthetic_v1.jsonl"
    p = argparse.ArgumentParser(prog="action-sentinel", description="Sentinel decision governance")
    p.add_argument("--config", help="settings YAML (default: config/sentinel.yaml or $SENTINEL_CONFIG)")
    p.add_argument("--db", help="override the SQLite database path")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("validate-rules", help="validate rule packs")
    a.add_argument("paths", nargs="*")
    a.set_defaults(fn=cmd_validate_rules)

    sub.add_parser("providers", help="list providers and their health").set_defaults(fn=cmd_providers)

    a = sub.add_parser("evaluate", help="evaluate action proposals from a JSON/JSONL file")
    a.add_argument("file")
    a.add_argument("--providers", help="comma-separated provider names (default: active_providers)")
    a.add_argument("--strategy", choices=["single", "sequential", "parallel"])
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(fn=cmd_evaluate)

    lab = sub.add_parser("lab", help="Confidence Lab").add_subparsers(dest="lab_cmd", required=True)
    a = lab.add_parser("suite", help="run the standard comparison and ablations")
    a.add_argument("--dataset", default=ds_default)
    a.add_argument("--model", default="mock-a", help="the decision provider under test")
    a.add_argument("--second", default="mock-b", help="second provider for multi-provider configs ('' for none)")
    a.add_argument("--extra-naive", default="rules", help="other providers to run as naive baselines")
    a.add_argument("--splits", default="test")
    a.add_argument("--limit", type=int)
    a.add_argument("--out")
    a.add_argument("--title")
    a.set_defaults(fn=cmd_lab_suite)
    a = lab.add_parser("run", help="run one configuration")
    a.add_argument("--name", default="custom")
    a.add_argument("--dataset", default=ds_default)
    a.add_argument("--providers", default="rules")
    a.add_argument("--splits", default="test")
    a.add_argument("--mode", choices=["sentinel", "naive"], default="sentinel")
    a.add_argument("--strategy", choices=["single", "sequential", "parallel"], default="parallel")
    for flag in ("--no-validation", "--no-disagreement", "--no-uncertainty", "--no-review"):
        a.add_argument(flag, action="store_true")
    a.add_argument("--limit", type=int)
    a.add_argument("--out")
    a.set_defaults(fn=cmd_lab_run)
    a = lab.add_parser("show", help="list runs, or show one run's metrics")
    a.add_argument("run_id", nargs="?")
    a.set_defaults(fn=cmd_lab_show)
    a = lab.add_parser("export", help="export stored runs as a markdown comparison")
    a.add_argument("run_ids", nargs="+")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_lab_export)
    a = lab.add_parser("calibrate", help="measure a provider's calibration on the calibration split")
    a.add_argument("--provider", required=True)
    a.add_argument("--dataset", default=ds_default)
    a.add_argument("--splits", default="calibration")
    a.add_argument("--limit", type=int)
    a.set_defaults(fn=cmd_lab_calibrate)

    a = sub.add_parser("reviews", help="list or act on human reviews (local operator tool)")
    a.add_argument("action", choices=["list", "approve", "reject", "override"])
    a.add_argument("review_id", nargs="?")
    a.add_argument("--reviewer")
    a.add_argument("--reason")
    a.add_argument("--changes", default="{}", help="JSON object of overridden fields")
    a.add_argument("--state", default="PENDING")
    a.set_defaults(fn=cmd_reviews)

    a = sub.add_parser("audit-export", help="export sanitized audit events as JSONL")
    a.add_argument("--out", required=True)
    a.add_argument("--correlation")
    a.add_argument("--limit", type=int, default=100_000)
    a.set_defaults(fn=cmd_audit_export)

    a = sub.add_parser("init", help="copy the shipped settings and rule packs into a directory, to edit")
    a.add_argument("dir", nargs="?", default=".")
    a.add_argument("--force", action="store_true", help="overwrite files that already exist")
    a.set_defaults(fn=cmd_init)

    a = sub.add_parser("demo", help="run the offline end-to-end demonstration")
    a.add_argument("--db", dest="db", default=None, help="database for the demo (default: a temporary file)")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(fn=cmd_demo)

    a = sub.add_parser("serve", help="start the API server")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8000)
    a.set_defaults(fn=cmd_serve)

    vega = sub.add_parser("vega", help="Vega decision lab (VegaML)").add_subparsers(dest="vega_cmd", required=True)
    a = vega.add_parser("check", help="check installation and cached weights; --load also loads the model")
    a.add_argument("--load", action="store_true")
    a.add_argument("--model", default="0.8b")
    a.add_argument("--device", default="cpu")
    a.set_defaults(fn=cmd_vega_check)
    a = vega.add_parser("decide", help="one typed decision with the real model")
    a.add_argument("--state", required=True, help="text, or a JSON object")
    a.add_argument("--type", choices=["choice", "score", "boolean"], default="choice")
    a.add_argument("--instructions", required=True)
    a.add_argument("--option", action="append", required=True, help="label=description (repeat)")
    a.add_argument("--adapters", choices=["off", "auto", "on"], default="auto")
    a.add_argument("--model", default="0.8b")
    a.add_argument("--device", default="cpu")
    a.add_argument("--json", action="store_true", help="print the full result, native output included")
    a.set_defaults(fn=cmd_vega_decide)
    a = vega.add_parser("fit", help="fit VegaML's few-shot head on your labelled actions and save it")
    a.add_argument("--dataset", required=True, help="JSONL of labelled actions (Confidence Lab dataset format)")
    a.add_argument("--split", default="calibration", help="split to fit on (dev or calibration; never test)")
    a.add_argument("--out", required=True, help="where to write the head file (.npz)")
    a.add_argument("--framing", choices=["compact", "rich"], default="compact")
    a.add_argument("--evaluate", choices=["dev", "test"], help="also report accuracy on this split (measure only)")
    a.add_argument("--model", default="0.8b")
    a.add_argument("--device", default="cpu")
    a.set_defaults(fn=cmd_vega_fit)
    a = vega.add_parser("data", help="build or check a public evaluation dataset")
    a.add_argument("action", choices=["build", "check"])
    a.add_argument("--version", default="pub-v1")
    a.add_argument("--root", default="data/vegalab")
    a.add_argument("--tasks")
    a.add_argument("--n-test", type=int, default=100)
    a.add_argument("--n-cal", type=int, default=60)
    a.add_argument("--n-dev", type=int, default=20)
    a.add_argument("--seed", type=int, default=20261015)
    a.set_defaults(fn=cmd_vega_data)
    a = vega.add_parser("run", help="run systems over a dataset (inference; resumable)")
    a.add_argument("--dataset", default="data/vegalab/pub-v1")
    a.add_argument("--out")
    a.add_argument("--systems", help="comma list of majority,vega-off,vega-auto,backbone-optscore")
    a.add_argument("--tasks")
    a.add_argument("--splits", default="test,calibration")
    a.add_argument("--limit", type=int)
    a.add_argument("--model", default="0.8b")
    a.add_argument("--device", default="cpu")
    a.set_defaults(fn=cmd_vega_run)
    a = vega.add_parser("analyze", help="compute metrics; writes metrics.json, summary.csv, REPORT.md")
    a.add_argument("run_dir")
    a.add_argument("--split", default="test")
    a.add_argument("--compare", nargs="*", help="pairs systemA:systemB")
    a.add_argument("--title")
    a.set_defaults(fn=cmd_vega_analyze)
    a = vega.add_parser("export", help="export per-task metrics as CSV or JSON")
    a.add_argument("run_dir")
    a.add_argument("--format", choices=["csv", "json"], default="csv")
    a.add_argument("--split", default="test")
    a.add_argument("--out")
    a.set_defaults(fn=cmd_vega_export)
    a = vega.add_parser("failures", help="inspect failed, abstained or wrong predictions")
    a.add_argument("run_dir")
    a.add_argument("--system", default="vega-off")
    a.add_argument("--kind", choices=["error", "abstained", "wrong", "confident-wrong"], default="confident-wrong")
    a.add_argument("--task")
    a.add_argument("--split", default="test")
    a.add_argument("--min-confidence", type=float, default=0.9)
    a.add_argument("--limit", type=int, default=30)
    a.set_defaults(fn=cmd_vega_failures)

    a = sub.add_parser("dashboard", help="start the Streamlit dashboard")
    a.add_argument("--port", type=int, default=8501)
    a.set_defaults(fn=cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
