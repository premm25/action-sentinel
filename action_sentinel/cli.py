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

    a = sub.add_parser("demo", help="run the offline end-to-end demonstration")
    a.add_argument("--db", dest="db", default=None, help="database for the demo (default: a temporary file)")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(fn=cmd_demo)

    a = sub.add_parser("serve", help="start the API server")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8000)
    a.set_defaults(fn=cmd_serve)

    a = sub.add_parser("dashboard", help="start the Streamlit dashboard")
    a.add_argument("--port", type=int, default=8501)
    a.set_defaults(fn=cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
