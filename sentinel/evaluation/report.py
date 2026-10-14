"""Markdown rendering of Confidence Lab reports. Only measured values are printed; undefined is 'n/a'."""

from __future__ import annotations

from typing import Any

from sentinel.contracts import EvaluationReport


def _f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, dict) and "value" in v:
        if v["value"] is None:
            return "n/a (n=0)"
        return "%.*f [%d/%d]" % (digits, v["value"], v["k"], v["n"])
    if isinstance(v, float):
        return "%.*f" % (digits, v)
    return str(v)


def _ci(v: dict) -> str:
    ci = v.get("ci95") if isinstance(v, dict) else None
    return "%.2fâ€“%.2f" % tuple(ci) if ci else "n/a"


GOV_ROWS = [
    ("outcome accuracy", "outcome_accuracy"),
    ("false approval rate â†“", "false_approval_rate"),
    ("hard violations blocked", "hard_violation_blocked"),
    ("policy-violation detection (expected BLOCK)", "policy_violation_detection_rate"),
    ("false block rate â†“", "false_block_rate"),
    ("missed escalation rate â†“", "missed_escalation_rate"),
    ("over-escalation of safe cases", "over_escalation_rate"),
    ("auto-approval of safe cases", "auto_approval_coverage"),
    ("human escalation rate", "human_escalation_rate"),
    ("abstention rate", "abstention_rate"),
    ("error rate", "error_rate"),
    ("disagreement rate", "disagreement_rate"),
]


def render_comparison(reports: list[EvaluationReport], title: str = "Confidence Lab comparison") -> str:
    if not reports:
        return "# %s\n\nNo runs.\n" % title
    r0 = reports[0]
    lines = ["# %s" % title, "",
             "Dataset `%s` (sha256 `%sâ€¦`), splits %s, %d examples per configuration."
             % (r0.dataset_version, r0.dataset_sha256[:12], ", ".join(r0.config.splits), r0.n_examples),
             "Policy `%s`. Sentinel %s, Python %s." % (r0.environment.get("policy_version"),
                                                      r0.environment.get("sentinel"), r0.environment.get("python")),
             "", "Values are `rate [k/n]`. Lower is better where marked â†“. 95% Wilson intervals are in the JSON.", "",
             "## Governance outcomes", ""]
    header = "| metric | " + " | ".join(r.config.name for r in reports) + " |"
    lines += [header, "|---|" + "---:|" * len(reports)]
    for label, key in GOV_ROWS:
        lines.append("| %s | %s |" % (label, " | ".join(_f(r.governance_metrics.get(key)) for r in reports)))
    lines.append("| latency p50 ms (pipeline) | %s |" % " | ".join(
        _f(r.governance_metrics["latency_ms"]["p50"], 1) for r in reports))
    lines += ["", "Configurations:", ""]
    for r in reports:
        c = r.config
        lines.append("- **%s**: providers %s; validation=%s, disagreement=%s, uncertainty=%s, review=%s, mode=%s"
                     % (c.name, c.providers, c.use_validation, c.use_disagreement, c.use_uncertainty, c.use_review,
                        c.governance_mode))

    lines += ["", "## Provider predictions (model accuracy, independent of governance)", ""]
    seen: set[str] = set()
    lines += ["| provider | kind | accuracy | coverage | macro F1 | ECE | Brier | AURC | confâ‰¥0.9 errors | "
              "abstain | errors+timeouts | p50 ms |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        for name, pm in r.provider_metrics.items():
            if name in seen:
                continue
            seen.add(name)
            cls = pm["classification"]
            cal = pm.get("calibration")
            ps = pm.get("proper_scores")
            rc = pm.get("risk_coverage")
            err = pm["error_rate"]["k"] + pm["timeout_rate"]["k"]
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d | %s |" % (
                name, r.environment["provider_kinds"].get(name), _f(cls["accuracy"]), _f(cls["coverage"]),
                _f(cls["macro_f1"]), _f(cal["ece"]) if cal else "n/a", _f(ps["brier"]) if ps else "n/a",
                _f(rc["aurc"]) if rc else "n/a", pm["high_confidence_errors"], _f(pm["abstention_rate"]), err,
                _f(pm["latency_ms"]["p50"], 1)))

    full = next((r for r in reports if r.config.name.startswith("6_")), reports[-1])
    lines += ["", "## Error analysis: %s" % full.config.name, ""]
    fa = [x for x in full.records if x.outcome.value == "APPROVE" and x.expected_outcome.value != "APPROVE"]
    fb = [x for x in full.records if x.outcome.value == "BLOCK" and x.expected_outcome.value != "BLOCK"]
    mis = [x for x in full.records if x.outcome != x.expected_outcome]
    lines.append("- False approvals: %s" % (", ".join("%s (%s)" % (x.example_id, "/".join(x.tags)) for x in fa)
                                             or "none"))
    lines.append("- False blocks: %s" % (", ".join(x.example_id for x in fb) or "none"))
    by_kind: dict[str, int] = {}
    for x in mis:
        k = "%sâ†’%s" % (x.expected_outcome.value, x.outcome.value)
        by_kind[k] = by_kind.get(k, 0) + 1
    lines.append("- All outcome mismatches (expectedâ†’actual): %s" % (
        ", ".join("%s Ã—%d" % kv for kv in sorted(by_kind.items())) or "none"))
    lines += ["", "## Notes and limitations", ""]
    notes = list(dict.fromkeys(n for r in reports for n in r.notes))
    lines += ["- %s" % n for n in notes]
    lines.append("- Expected outcomes encode the reference policy's intent, written by the dataset author. "
                 "Agreement with them shows consistency with that policy, not real-world correctness.")
    return "\n".join(lines) + "\n"
