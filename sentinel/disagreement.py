"""Detect disagreement between providers, and between providers, policy and evidence.

Disagreement is recorded, never resolved by voting. The governance engine decides what to do with it,
and a hard policy violation always takes precedence.
"""

from __future__ import annotations

from collections.abc import Collection

from sentinel.contracts import (
    DisagreementRecord,
    DisagreementReport,
    ProviderResult,
    Severity,
    ValidationReport,
)


def detect_disagreement(results: list[ProviderResult], validation: ValidationReport | None, strategy: str,
                        permissive_labels: Collection[str] = ("safe",),
                        restrictive_labels: Collection[str] = ("unsafe",),
                        confidence_gap: float = 0.3) -> DisagreementReport:
    records: list[DisagreementRecord] = []
    ok = [r for r in results if r.succeeded]

    for r in results:
        if r.failed:
            records.append(DisagreementRecord(
                kind="missing_result", severity=Severity.LOW, providers=[r.provider_name],
                detail="%s returned no result (%s)" % (r.provider_name, r.error.code if r.error else r.status),
                observed={"status": r.status}, recommended_handling="informational"))

    labels = {r.provider_name: r.label for r in ok}
    distinct = set(labels.values())
    if len(distinct) > 1:
        severe = bool(distinct & set(permissive_labels)) and bool(distinct & set(restrictive_labels))
        records.append(DisagreementRecord(
            kind="category", severity=Severity.HIGH if severe else Severity.MEDIUM, providers=sorted(labels),
            detail="providers chose different categories: %s" % ", ".join(
                "%s=%s" % (k, v) for k, v in sorted(labels.items())),
            observed={k: v for k, v in labels.items()}, recommended_handling="human_review"))
    elif len(distinct) == 1:
        confs = {r.provider_name: r.confidence for r in ok if r.confidence is not None}
        if len(confs) > 1 and max(confs.values()) - min(confs.values()) > confidence_gap:
            records.append(DisagreementRecord(
                kind="confidence", severity=Severity.LOW, providers=sorted(confs),
                detail="providers agree on the category but their confidence differs by more than %.2f "
                       "(scales may not be comparable)" % confidence_gap,
                observed={k: v for k, v in confs.items()}, recommended_handling="informational"))

    if validation is not None:
        permissive = sorted(n for n, lbl in labels.items() if lbl in permissive_labels)
        restrictive = sorted(n for n, lbl in labels.items() if lbl in restrictive_labels)
        if validation.hard_violations and permissive:
            records.append(DisagreementRecord(
                kind="model_vs_policy", severity=Severity.HIGH, providers=permissive,
                detail="policy blocks the action (%s) but providers rate it permissible; policy takes precedence"
                       % ", ".join(f.code for f in validation.hard_violations),
                observed={"failed_rules": [f.code for f in validation.hard_violations]},
                recommended_handling="policy_precedence"))
        if not validation.findings and restrictive:
            records.append(DisagreementRecord(
                kind="model_vs_policy", severity=Severity.MEDIUM, providers=restrictive,
                detail="no policy finding, but providers rate the action unsafe",
                recommended_handling="human_review"))
        if validation.evidence.status == "contradicted" and permissive:
            records.append(DisagreementRecord(
                kind="evidence_vs_model", severity=Severity.HIGH, providers=permissive,
                detail="evidence contradicts the action but providers rate it permissible",
                observed={"contradicting": validation.evidence.contradicting},
                recommended_handling="human_review"))
        if any(f.code == "POLICY_CONFLICT" for f in validation.findings):
            records.append(DisagreementRecord(
                kind="contradictory_findings", severity=Severity.MEDIUM,
                detail="rule packs produced conflicting findings for this action",
                recommended_handling="human_review"))

    agreement = not any(r.kind in ("category", "model_vs_policy", "evidence_vs_model", "contradictory_findings")
                        for r in records)
    return DisagreementReport(agreement=agreement, records=records, strategy=strategy)
