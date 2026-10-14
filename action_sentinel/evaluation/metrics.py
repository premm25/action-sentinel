"""Evaluation metrics. Pure Python; undefined values are returned as None, never as 0.

Probabilistic metrics are computed only when their assumptions hold:
* ECE / reliability / risk-coverage need a top-label confidence for every answered example.
* Brier score and log loss need a full probability distribution over all categories.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None when n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict[str, Any]:
    return {"value": safe_div(k, n), "k": k, "n": n, "ci95": wilson(k, n)}


def classification(y_true: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str]) -> dict[str, Any]:
    """Accuracy over all examples (no answer = wrong) and over answered ones, plus per-class P/R/F1."""
    n = len(y_true)
    answered = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if p is not None]
    correct = sum(1 for t, p in answered if t == p)
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in answered:
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    per_class = {}
    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in labels if t != c)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p != c)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = None if prec is None or rec is None else (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(1 for t in y_true if t == c)}
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    return {
        "n": n, "answered": len(answered),
        "coverage": safe_div(len(answered), n),
        "accuracy": rate(correct, n),
        "selective_accuracy": rate(correct, len(answered)),
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else None,
        "macro_f1_classes": len(f1s),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(labels), "rows_true_cols_pred": [[cm[t][p] for p in labels]
                                                                             for t in labels]},
    }


def calibration(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> dict[str, Any] | None:
    """Top-label expected calibration error with equal-width bins, and reliability-diagram data."""
    if not confidences:
        return None
    table = []
    ece = 0.0
    n = len(confidences)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            table.append({"bin": [lo, hi], "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mc = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        ece += len(idx) / n * abs(acc - mc)
        table.append({"bin": [lo, hi], "count": len(idx), "mean_confidence": mc, "accuracy": acc})
    return {"ece": ece, "bins": bins, "n": n, "reliability": table,
            "mean_confidence": sum(confidences) / n, "accuracy": sum(correct) / n}


def risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> dict[str, Any] | None:
    """Selective risk at each coverage level when keeping the most confident answers; AURC summarizes it."""
    if not confidences:
        return None
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    curve = []
    errors = 0
    for k, i in enumerate(order, start=1):
        errors += 0 if correct[i] else 1
        curve.append({"coverage": k / len(order), "risk": errors / k, "threshold": confidences[i]})
    aurc = sum(p["risk"] for p in curve) / len(curve)
    return {"aurc": aurc, "curve": curve}


def brier_and_log_loss(probs: Sequence[dict[str, float]], y_true: Sequence[str], labels: Sequence[str],
                       eps: float = 1e-12) -> dict[str, Any] | None:
    """Multiclass Brier score and log loss. Requires full distributions; returns None otherwise."""
    if not probs or any(p is None or set(p) != set(labels) for p in probs):
        return None
    brier = sum(sum((p[c] - (1.0 if c == t else 0.0)) ** 2 for c in labels)
                for p, t in zip(probs, y_true, strict=True)) / len(probs)
    clipped = sum(1 for p, t in zip(probs, y_true, strict=True) if p[t] < eps)
    ll = -sum(math.log(max(p[t], eps)) for p, t in zip(probs, y_true, strict=True)) / len(probs)
    return {"brier": brier, "log_loss": ll, "n": len(probs), "log_loss_clipped_examples": clipped}


def percentiles(values: Sequence[float], ps: Sequence[int] = (50, 95, 99)) -> dict[str, float | None]:
    if not values:
        return {"p%d" % p: None for p in ps}
    s = sorted(values)
    out = {}
    for p in ps:
        k = (len(s) - 1) * p / 100
        lo, hi = math.floor(k), math.ceil(k)
        out["p%d" % p] = s[lo] + (s[hi] - s[lo]) * (k - lo)
    return out
