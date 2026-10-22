"""Extra analysis for a binary vega-lab run: AUROC, confusion counts, and a calibration-fitted threshold.

    python scripts/threshold_check.py reports/vegalab/agentharm-v1_cpu_0.8b [--positive harmful]

A system that ranks well but sits on the wrong side of 0.5 scores at the majority level on plain
accuracy. To compare decision methods fairly, each system's threshold on P(positive) is chosen on the
CALIBRATION split only (highest accuracy, ties broken toward 0.5) and then applied once to TEST.
Systems with no calibration predictions (the few-shot head, fitted on that split) keep 0.5.
Reads only the saved predictions; writes threshold_check.json next to them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def auroc(scores: list[float], positive: list[bool]) -> float:
    pos = [s for s, y in zip(scores, positive, strict=True) if y]
    neg = [s for s, y in zip(scores, positive, strict=True) if not y]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def confusion(rows: list[dict[str, Any]], positive: str, th: float) -> dict[str, int]:
    out = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    for r in rows:
        pred = r["probabilities"][positive] >= th
        gold = r["gold_label"] == positive
        out[("t" if pred == gold else "f") + ("p" if pred else "n")] += 1
    return out


def accuracy(rows: list[dict[str, Any]], positive: str, th: float) -> float:
    c = confusion(rows, positive, th)
    return (c["tp"] + c["tn"]) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--positive", default="harmful")
    args = ap.parse_args()
    run = Path(args.run_dir)
    result: dict[str, Any] = {"positive": args.positive, "systems": {}}
    for f in sorted(run.glob("*.jsonl")):
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [r for r in rows if r["status"] == "ok"]
        test = [r for r in rows if r["split"] == "test"]
        cal = [r for r in rows if r["split"] == "calibration"]
        scores = [r["probabilities"][args.positive] for r in test]
        entry: dict[str, Any] = {
            "n_test": len(test), "n_calibration": len(cal),
            "auroc": auroc(scores, [r["gold_label"] == args.positive for r in test]),
            "at_0.5": {"accuracy": accuracy(test, args.positive, 0.5), **confusion(test, args.positive, 0.5)}}
        if cal:
            cands = sorted({r["probabilities"][args.positive] for r in cal})
            th = max(cands, key=lambda c: (accuracy(cal, args.positive, c), -abs(c - 0.5)))
            entry["calibrated"] = {"threshold": th, "calibration_accuracy": accuracy(cal, args.positive, th),
                                   "accuracy": accuracy(test, args.positive, th),
                                   **confusion(test, args.positive, th)}
        result["systems"][f.stem] = entry
        print(f.stem, json.dumps(entry))
    (run / "threshold_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
