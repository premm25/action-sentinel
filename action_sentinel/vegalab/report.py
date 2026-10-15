"""Render analysis results (analysis.analyze output) as Markdown and CSV. Only measured values; n/a when undefined."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


def _v(x: Any, d: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, dict) and "value" in x:
        return "n/a" if x["value"] is None else "%.*f [%d/%d]" % (d, x["value"], x["k"], x["n"])
    if isinstance(x, float):
        return "%.*f" % (d, x)
    return str(x)


def summary_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for system, tasks in res["per_task"].items():
        for task, m in tasks.items():
            rc = res["recalibrated"].get(system, {}).get(task, {})
            rows.append({
                "system": system, "task": task, "n": m["n"], "errors": m["errors"],
                "accuracy": m["accuracy"]["value"], "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"], "ece": m.get("ece"), "brier": m.get("brier"), "log_loss": m.get("log_loss"),
                "aurc": m.get("aurc"), "mean_confidence": m.get("mean_confidence"),
                "high_conf_errors": m.get("high_conf_errors"),
                "abstain_rate": (m.get("abstain_rate") or {}).get("value"), "coverage": m.get("coverage"),
                "selective_accuracy": (m.get("selective_accuracy") or {}).get("value"),
                "conformal_coverage": (m.get("conformal_coverage") or {}).get("value"),
                "mean_set_size": m.get("mean_set_size"), "unbound_rate": (m.get("unbound_rate") or {}).get("value"),
                "ordinal_mae": m.get("ordinal_mae"), "latency_p50_ms": m["latency_ms"]["p50"],
                "latency_p95_ms": m["latency_ms"]["p95"], "throughput_per_s": m.get("throughput_per_s"),
                "recal_temperature": rc.get("temperature"), "recal_ece": rc.get("ece"), "recal_log_loss": rc.get("log_loss"),
            })
    return rows


def to_csv(res: dict[str, Any]) -> str:
    rows = summary_rows(res)
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()


def render(res: dict[str, Any], experiment: dict[str, Any] | None = None, title: str = "Vega lab results") -> str:
    systems, tasks = res["systems"], res["tasks"]
    L = ["# %s" % title, ""]
    if experiment:
        L += ["Run `%s` · dataset `%s` · split `%s` · model `%s` on `%s`." % (
            experiment.get("run_id"), experiment.get("dataset_version"), res["split"], experiment.get("model"),
            experiment.get("device")), ""]
    L += ["Values are `value [k/n]` where a count applies. All numbers are computed from the saved predictions "
          "in this folder.", "", "## Accuracy by task", "",
          "| task | " + " | ".join(systems) + " |", "|---|" + "---:|" * len(systems)]
    for t in tasks:
        L.append("| %s | %s |" % (t, " | ".join(_v(res["per_task"].get(s, {}).get(t, {}).get("accuracy")) for s in systems)))
    L.append("| **macro mean** | %s |" % " | ".join(_v(res["macro"].get(s, {}).get("accuracy")) for s in systems))
    for title_, key in (("Balanced accuracy", "balanced_accuracy"), ("Expected calibration error (10 bins) ↓", "ece"),
                        ("Brier score ↓", "brier"), ("AURC (risk–coverage) ↓", "aurc")):
        L += ["", "## %s" % title_, "", "| task | " + " | ".join(systems) + " |", "|---|" + "---:|" * len(systems)]
        for t in tasks:
            L.append("| %s | %s |" % (t, " | ".join(_v(res["per_task"].get(s, {}).get(t, {}).get(key)) for s in systems)))
    vega = [s for s in systems if s.startswith("vega-")]
    if vega:
        L += ["", "## VegaML abstention, conformal sets, unbound (native signals)", "",
              "| system | task | abstain rate | coverage | selective acc. | acc. on abstained | conformal coverage "
              "(α=0.1) | mean set size | unbound |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in vega:
            for t in tasks:
                m = res["per_task"].get(s, {}).get(t)
                if not m:
                    continue
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    s, t, _v(m.get("abstain_rate")), _v(m.get("coverage")), _v(m.get("selective_accuracy")),
                    _v(m.get("accuracy_on_abstained")), _v(m.get("conformal_coverage")), _v(m.get("mean_set_size"), 2),
                    _v(m.get("unbound_rate"))))
    if res["recalibrated"]:
        L += ["", "## Post-hoc temperature (fitted on the calibration split, applied to test)", "",
              "| system | task | T | ECE before | ECE after | log loss before | log loss after |",
              "|---|---|---:|---:|---:|---:|---:|"]
        for s, tm in res["recalibrated"].items():
            for t, rc in tm.items():
                m = res["per_task"][s][t]
                L.append("| %s | %s | %.2f | %s | %s | %s | %s |" % (s, t, rc["temperature"], _v(m.get("ece")),
                                                                    _v(rc.get("ece")), _v(m.get("log_loss")),
                                                                    _v(rc.get("log_loss"))))
    if res["paired"]:
        L += ["", "## Paired accuracy differences (bootstrap 95% CI, same examples)", "",
              "| comparison | task | n | difference | 95% CI | only A correct | only B correct |",
              "|---|---|---:|---:|---|---:|---:|"]
        for comp, tm in res["paired"].items():
            for t, p in tm.items():
                if p:
                    L.append("| %s | %s | %d | %+.3f | [%+.3f, %+.3f] | %d | %d |" % (
                        comp, t, p["n"], p["diff"], p["ci95"][0], p["ci95"][1], p["a_only_correct"], p["b_only_correct"]))
    L += ["", "## Latency per example (wall clock, ms)", "", "| task | " + " | ".join(systems) + " |",
          "|---|" + "---:|" * len(systems)]
    for t in tasks:
        L.append("| %s | %s |" % (t, " | ".join(_v(res["per_task"].get(s, {}).get(t, {}).get("latency_ms", {}).get("p50"), 0)
                                               for s in systems)))
    L += ["", "## Slices (all test tasks pooled)", ""]
    for s in systems:
        for key, groups in res["slices"].get(s, {}).items():
            L.append("- **%s** by %s: %s" % (s, key, "; ".join("%s %s" % (g, _v(v["accuracy"])) for g, v in groups.items())))
    return "\n".join(L) + "\n"


def write_reports(run_dir: str | Path, res: dict[str, Any], title: str) -> list[Path]:
    run_dir = Path(run_dir)
    exp_f = run_dir / "experiment.json"
    experiment = json.loads(exp_f.read_text(encoding="utf-8")) if exp_f.exists() else None
    paths = [run_dir / "metrics.json", run_dir / "summary.csv", run_dir / "REPORT.md"]
    paths[0].write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    paths[1].write_text(to_csv(res), encoding="utf-8", newline="")
    paths[2].write_text(render(res, experiment, title), encoding="utf-8")
    return paths
