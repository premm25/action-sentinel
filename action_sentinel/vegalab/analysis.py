"""Metrics from saved predictions (no inference here). Reuses action_sentinel.evaluation.metrics.

Definitions (also in docs/vegalab/METHODOLOGY.md):
* accuracy: correct / all examples; an example with an inference error counts as not correct and is
  reported separately in `errors`.
* balanced accuracy: mean recall over the gold classes present in the split.
* ECE: top-label, 10 equal-width confidence bins, weighted by bin count.
* Brier / log loss: multiclass, over the full option distribution; only when every evaluated example has
  a complete distribution. Log loss clips probabilities at 1e-12 and reports how many were clipped.
* abstention (VegaML native flag at the checkpoint's default risk): coverage = share not abstained;
  selective accuracy = accuracy among non-abstained.
* risk-coverage / AURC: examples sorted by confidence, from the system's own confidence.
* conformal: empirical coverage = gold in VegaML's prediction_set; mean set size.
* recalibration: one temperature T fitted on the calibration split (minimizing NLL over a log grid),
  applied to test-split log-probabilities. The test split is never used to fit.
* paired difference: bootstrap (2,000 resamples, seeded) over examples both systems answered.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from action_sentinel.evaluation import metrics as M


def load_predictions(run_dir: str | Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(Path(run_dir).glob("*.jsonl")):
        rows += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def balanced_accuracy(gold: list[str], pred: list[str | None]) -> float | None:
    classes = sorted(set(gold))
    if not classes:
        return None
    recalls = []
    for c in classes:
        idx = [i for i, g in enumerate(gold) if g == c]
        recalls.append(sum(1 for i in idx if pred[i] == c) / len(idx))
    return sum(recalls) / len(recalls)


def _temper(probs: dict[str, float], T: float) -> dict[str, float]:
    logs = {k: math.log(max(v, 1e-12)) / T for k, v in probs.items()}
    m = max(logs.values())
    ex = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def fit_temperature(records: list[dict[str, Any]]) -> float | None:
    usable = [r for r in records if r["status"] == "ok" and r.get("probabilities")]
    if len(usable) < 10:
        return None
    grid = [math.exp(x / 40) for x in range(-120, 121)]   # T in [0.05, 20]

    def nll(T: float) -> float:
        return -sum(math.log(max(_temper(r["probabilities"], T)[r["gold_label"]], 1e-12)) for r in usable) / len(usable)
    return min(grid, key=nll)


def group_metrics(records: list[dict[str, Any]], temperature: float | None = None) -> dict[str, Any]:
    n = len(records)
    ok = [r for r in records if r["status"] == "ok"]
    labels = records[0]["labels"] if records else []
    gold = [r["gold_label"] for r in records]
    pred = [r.get("predicted_label") if r["status"] == "ok" else None for r in records]
    out: dict[str, Any] = {"n": n, "errors": n - len(ok)}
    cls = M.classification(gold, pred, labels)
    out["accuracy"] = cls["accuracy"]
    out["balanced_accuracy"] = balanced_accuracy(gold, pred)
    out["macro_f1"] = cls["macro_f1"]
    out["confusion_matrix"] = cls["confusion_matrix"]
    wall = [r["wall_ms"] for r in ok if r.get("wall_ms") is not None]
    out["latency_ms"] = M.percentiles(wall)
    out["throughput_per_s"] = (len(wall) / (sum(wall) / 1000)) if wall and sum(wall) > 0 else None

    probs_ok = [r for r in ok if r.get("probabilities") and set(r["probabilities"]) == set(labels)]
    if ok and len(probs_ok) == len(ok):
        P = [(_temper(r["probabilities"], temperature) if temperature else r["probabilities"]) for r in probs_ok]
        top = [max(p, key=p.get) for p in P]
        conf = [p[t] for p, t in zip(P, top, strict=True)]
        correct = [t == r["gold_label"] for t, r in zip(top, probs_ok, strict=True)]
        cal = M.calibration(conf, correct)
        out["ece"] = cal["ece"] if cal else None
        out["reliability"] = cal["reliability"] if cal else None
        out["mean_confidence"] = cal["mean_confidence"] if cal else None
        ps = M.brier_and_log_loss(P, [r["gold_label"] for r in probs_ok], labels)
        out["brier"] = ps["brier"] if ps else None
        out["log_loss"] = ps["log_loss"] if ps else None
        out["log_loss_clipped"] = ps["log_loss_clipped_examples"] if ps else None
        rc = M.risk_coverage(conf, correct)
        out["aurc"] = rc["aurc"] if rc else None
        out["risk_coverage"] = [p for i, p in enumerate(rc["curve"]) if i % max(1, len(rc["curve"]) // 20) == 0] \
            if rc else None
        out["high_conf_errors"] = sum(1 for c, k in zip(conf, correct, strict=True) if c >= 0.9 and not k)
    else:
        out["ece"] = out["brier"] = out["log_loss"] = out["aurc"] = None
        out["probabilistic_note"] = "not computed: full probability distributions unavailable"

    flagged = [r for r in ok if isinstance(r.get("abstain"), bool)]
    if ok and len(flagged) == len(ok):
        kept = [r for r in flagged if not r["abstain"]]
        out["abstain_rate"] = M.rate(len(flagged) - len(kept), len(flagged))
        out["coverage"] = len(kept) / len(flagged)
        out["selective_accuracy"] = M.rate(sum(1 for r in kept if r["predicted_label"] == r["gold_label"]), len(kept))
        ab = [r for r in flagged if r["abstain"]]
        out["accuracy_on_abstained"] = M.rate(sum(1 for r in ab if r["predicted_label"] == r["gold_label"]), len(ab))
    else:
        out["abstain_rate"] = out["coverage"] = out["selective_accuracy"] = None
    sets = [r for r in ok if isinstance(r.get("prediction_set"), list)]
    if ok and len(sets) == len(ok):
        out["conformal_coverage"] = M.rate(sum(1 for r in sets if r["gold_label"] in r["prediction_set"]), len(sets))
        out["mean_set_size"] = sum(len(r["prediction_set"]) for r in sets) / len(sets)
    if ok and all(isinstance(r.get("unbound"), bool) for r in ok):
        out["unbound_rate"] = M.rate(sum(1 for r in ok if r["unbound"]), len(ok))
    if records and records[0]["labels"] == [str(i) for i in range(len(labels))] and len(labels) > 2:
        diffs = [abs(int(p) - int(g)) for p, g in zip(pred, gold, strict=True) if p is not None]
        out["ordinal_mae"] = sum(diffs) / len(diffs) if diffs else None
        out["within_1"] = (sum(1 for d in diffs if d <= 1) / len(diffs)) if diffs else None
    if temperature:
        out["temperature"] = temperature
    return out


def paired_difference(a: list[dict], b: list[dict], resamples: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """Accuracy(a) - accuracy(b) on examples present in both, with a bootstrap 95% interval."""
    bi = {r["example_id"]: r for r in b}
    pairs = [(int(r["status"] == "ok" and r["predicted_label"] == r["gold_label"]),
              int(bi[r["example_id"]]["status"] == "ok" and bi[r["example_id"]]["predicted_label"] == r["gold_label"]))
             for r in a if r["example_id"] in bi]
    if not pairs:
        return None
    d = [x - y for x, y in pairs]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(resamples))
    return {"n": len(d), "diff": sum(d) / len(d),
            "ci95": [boots[int(0.025 * resamples)], boots[int(0.975 * resamples)]],
            "a_only_correct": sum(1 for x, y in pairs if x and not y),
            "b_only_correct": sum(1 for x, y in pairs if y and not x)}


def analyze(run_dir: str | Path, split: str = "test", slice_keys: tuple[str, ...] = ("difficulty", "length", "label_space"),
            compare: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    rows = load_predictions(run_dir)
    by: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["system"], r["task"], r["split"])].append(r)
    systems = sorted({k[0] for k in by})
    tasks = sorted({k[1] for k in by if k[2] == split})
    result: dict[str, Any] = {"run_dir": str(run_dir), "split": split, "systems": systems, "tasks": tasks,
                              "per_task": {}, "recalibrated": {}, "slices": {}, "paired": {}, "macro": {}}
    for s in systems:
        for t in tasks:
            recs = by.get((s, t, split))
            if not recs:
                continue
            result["per_task"].setdefault(s, {})[t] = group_metrics(recs)
            T = fit_temperature(by.get((s, t, "calibration"), []))
            if T is not None:
                result["recalibrated"].setdefault(s, {})[t] = {
                    k: v for k, v in group_metrics(recs, T).items()
                    if k in ("temperature", "ece", "brier", "log_loss", "mean_confidence", "aurc")}
        all_recs = [r for (sy, t, sp), rs in by.items() if sy == s and sp == split and t in tasks for r in rs]
        for key in slice_keys:
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in all_recs:
                groups[r.get("slices", {}).get(key, "n/a")].append(r)
            result["slices"].setdefault(s, {})[key] = {
                g: {"n": len(v), "accuracy": M.rate(sum(1 for r in v if r["status"] == "ok"
                                                       and r["predicted_label"] == r["gold_label"]), len(v))}
                for g, v in sorted(groups.items())}
        accs = [m["accuracy"]["value"] for m in result["per_task"].get(s, {}).values()
                if m["accuracy"]["value"] is not None]
        eces = [m["ece"] for m in result["per_task"].get(s, {}).values() if m.get("ece") is not None]
        result["macro"][s] = {"tasks": len(accs), "accuracy": sum(accs) / len(accs) if accs else None,
                              "ece": sum(eces) / len(eces) if eces else None}
    for a, b in compare or []:
        for t in tasks:
            if (a, t, split) in by and (b, t, split) in by:
                result["paired"].setdefault("%s vs %s" % (a, b), {})[t] = paired_difference(by[(a, t, split)],
                                                                                            by[(b, t, split)])
    return result
