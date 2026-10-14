"""Uncertainty assessment over provider results.

The separate concepts are kept apart in the output:

* model confidence     -> ProviderUncertainty.confidence / confidence_source
* calibration          -> ProviderUncertainty.calibration_status / calibration_ece
* predictive ambiguity -> prediction sets with more than one member, low confidence
* evidence sufficiency -> UncertaintyAssessment.evidence (computed by validation, not by a model)
* provider health      -> provider status (failed, abstained)

Confidence values from different providers are never compared with each other or averaged.
Each is checked only against the thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from action_sentinel.contracts import EvidenceAssessment, ProviderResult, ProviderUncertainty, UncertaintyAssessment
from action_sentinel.contracts.common import StrictModel


class UncertaintyThresholds(StrictModel):
    approve_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    abstain_below: float = Field(default=0.5, ge=0.0, le=1.0)
    require_measured_calibration: bool = False
    max_calibration_ece: float = Field(default=0.1, ge=0.0, le=1.0)


def assess_uncertainty(results: list[ProviderResult], evidence: EvidenceAssessment,
                       thresholds: UncertaintyThresholds,
                       calibration: Mapping[str, Mapping[str, Any]] | None = None) -> UncertaintyAssessment:
    calibration = calibration or {}
    per: list[ProviderUncertainty] = []
    reasons: list[str] = []
    for r in results:
        notes: list[str] = []
        profile = calibration.get(r.provider_name)
        ece = float(profile["ece"]) if profile and profile.get("ece") is not None else None
        cal_status = "measured" if profile else r.calibration_status
        usable = False
        if r.failed:
            notes.append("PROVIDER_%s:%s" % (r.status.upper(), r.error.code if r.error else "UNKNOWN"))
        elif r.status == "abstained":
            notes.append("PROVIDER_ABSTAINED")
        elif r.provider_kind == "deterministic":
            usable = True
        else:
            usable = True
            if r.confidence is None:
                notes.append("CONFIDENCE_UNAVAILABLE")
            elif r.confidence < thresholds.abstain_below:
                usable = False
                notes.append("CONFIDENCE_BELOW_ABSTAIN_THRESHOLD")
            elif r.confidence < thresholds.approve_min_confidence:
                notes.append("CONFIDENCE_BELOW_APPROVAL_THRESHOLD")
            if r.prediction_set is not None and len(r.prediction_set) > 1:
                notes.append("AMBIGUOUS_PREDICTION_SET")
            if cal_status != "measured":
                notes.append("CALIBRATION_UNKNOWN")
                if thresholds.require_measured_calibration:
                    notes.append("UNCALIBRATED_CONFIDENCE_NOT_ACCEPTED")
            elif ece is not None and ece > thresholds.max_calibration_ece:
                notes.append("POORLY_CALIBRATED")
        per.append(ProviderUncertainty(
            provider_name=r.provider_name, status=r.status, confidence=r.confidence,
            confidence_source=r.confidence_source, calibration_status=cal_status, calibration_ece=ece,
            usable=usable, notes=notes))
        reasons.extend("%s:%s" % (r.provider_name, n) for n in notes)

    status = _overall(per)
    return UncertaintyAssessment(status=status, providers=per, evidence=evidence, reasons=reasons,
                                 thresholds={"approve_min_confidence": thresholds.approve_min_confidence,
                                             "abstain_below": thresholds.abstain_below,
                                             "max_calibration_ece": thresholds.max_calibration_ece})


_UNCERTAIN_NOTES = {"CONFIDENCE_BELOW_APPROVAL_THRESHOLD", "AMBIGUOUS_PREDICTION_SET", "POORLY_CALIBRATED",
                    "UNCALIBRATED_CONFIDENCE_NOT_ACCEPTED"}


def _overall(per: list[ProviderUncertainty]) -> str:
    if not per or all(p.status in ("error", "timeout", "unavailable") for p in per):
        return "failed"
    usable = [p for p in per if p.usable]
    if not usable:
        return "abstained"
    if any("CONFIDENCE_UNAVAILABLE" in p.notes for p in usable):
        return "unknown"
    if len(usable) < len(per):  # some provider failed or abstained: the picture is partial
        return "uncertain"
    if any(set(p.notes) & _UNCERTAIN_NOTES for p in usable):
        return "uncertain"
    return "confident"
