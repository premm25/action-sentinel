"""Does VegaML's few-shot head beat ordinary classifiers given the same 64 AgentHarm examples?

    <vega-venv>/python scripts/agentharm_baselines.py [--cache DIR] [--out reports/vegalab/agentharm_baselines_0.8b]

Protocol (fixed before running; see docs/vegalab/AGENTHARM_BASELINES.md). Every system is fitted on the
64 calibration examples only and evaluated once on the 352 test examples. Hyperparameters are either
fixed here or chosen by 5-fold cross-validation inside the calibration split.

  vega-head        VegaML's own TTTHead (PCA + logistic) on VegaML's pooled features; reproduces the
                   vega-head system from the main AgentHarm run (sanity check against its predictions)
  vegafeat-lr      ordinary L2 logistic regression on the same VegaML features (no PCA)
  vegafeat-centroid nearest class centroid (cosine) on the same features; no fitted weights
  probe-lr         L2 logistic regression on the frozen backbone's mean-pooled hidden states (layers
                   13 and 19, the layers VegaML reads) of the request alone: no question, no options,
                   no VegaML pooling
  tfidf-lr         L2 logistic regression on TF-IDF word uni+bigrams of the request and tool names;
                   vocabulary from the calibration split; no neural model at all

Features are cached outside the repo (they derive from AgentHarm, which stays out of git); only metrics
and per-example probabilities keyed by example id are written to --out.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "vegalab" / "agentharm-v1" / "agentharm.jsonl"
MAIN_RUN = ROOT / "reports" / "vegalab" / "agentharm-v1_cpu_0.8b"
LABELS = ["benign", "harmful"]
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
SEED = 20261024


# ---------------------------------------------------------------- features
def extract(rows: list[dict[str, Any]], cache: Path) -> dict[str, np.ndarray]:
    """VegaML pooled features and plain backbone mean-pooled features, cached and resumable."""
    import torch

    from action_sentinel.vegalab.contracts import DecisionQuestion
    from action_sentinel.vegalab.vega_core import LiveVegaRunner

    cache.mkdir(parents=True, exist_ok=True)
    f = cache / "features.npz"
    done: dict[str, dict[str, np.ndarray]] = {"vega": {}, "probe": {}}
    if f.exists():
        z = np.load(f, allow_pickle=False)
        for i, eid in enumerate(z["ids"]):
            done["vega"][str(eid)] = z["vega"][i]
            done["probe"][str(eid)] = z["probe"][i]
    todo = [r for r in rows if r["example_id"] not in done["vega"]]
    if todo:
        runner = LiveVegaRunner("0.8b", device="cpu")
        runner.load()
        vega = runner._vega
        perc = vega.engine.perc
        if not hasattr(perc, "_run"):
            raise RuntimeError("backbone runs out of process (%s); the probe needs in-process access"
                               % type(perc).__name__)
        from vegaml._common import serialize_state
        layers = sorted(perc.capture)
        print("loaded; live=%s offline_fallback=%s probe layers=%s" % (runner.live, runner.offline_fallback,
                                                                     layers), flush=True)
        t0 = time.perf_counter()
        for n, r in enumerate(todo, 1):
            q = DecisionQuestion.model_validate(r["question"]).to_vega()
            done["vega"][r["example_id"]] = np.asarray(vega.features(r["state"], q), dtype=np.float32)
            ids = perc.tok(serialize_state(r["state"]))["input_ids"]
            with torch.inference_mode():
                cap = perc._run(torch.tensor([ids], device=perc.device))
            done["probe"][r["example_id"]] = np.concatenate(
                [cap[i][0].float().mean(0).cpu().numpy() for i in layers]).astype(np.float32)
            if n % 25 == 0 or n == len(todo):
                _save(f, done)
                print("  features %d/%d (%.0fs)" % (n, len(todo), time.perf_counter() - t0), flush=True)
        del vega, perc, runner
    order = [r["example_id"] for r in rows]
    return {k: np.stack([done[k][e] for e in order]) for k in ("vega", "probe")}


def _save(f: Path, done: dict[str, dict[str, np.ndarray]]) -> None:
    ids = sorted(done["vega"])
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(ids), vega=np.stack([done["vega"][e] for e in ids]),
             probe=np.stack([done["probe"][e] for e in ids]))
    tmp.replace(f)


# ---------------------------------------------------------------- classifiers
def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(x - mu) / sd for x in (train, *others)]


def fit_lr(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 3000) -> tuple[np.ndarray, float]:
    """Binary L2 logistic regression, mean log loss + lam/2 |w|^2, by gradient descent with a safe step."""
    n, d = X.shape
    step = 1.0 / (np.linalg.norm(X, 2) ** 2 / (4 * n) + lam)
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= step * (X.T @ g / n + lam * w)
        b -= step * g.mean()
    return w, b


def predict_lr(model: tuple[np.ndarray, float], X: np.ndarray) -> np.ndarray:
    w, b = model
    return 1 / (1 + np.exp(-(X @ w + b)))


def folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    out: list[list[int]] = [[] for _ in range(k)]
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        for j, i in enumerate(idx):
            out[j % k].append(int(i))
    return [np.array(sorted(f)) for f in out]


def cv_lambda(X: np.ndarray, y: np.ndarray, prep) -> tuple[float, dict[str, float]]:
    """Pick lam by 5-fold CV accuracy on the calibration split only; ties go to the larger lam."""
    scores = {}
    for lam in LAMBDAS:
        correct = 0
        for f in folds(y):
            tr = np.setdiff1d(np.arange(len(y)), f)
            a, b = prep(X[tr], X[f])
            correct += int(((predict_lr(fit_lr(a, y[tr], lam), b) >= 0.5) == y[f]).sum())
        scores[str(lam)] = correct / len(y)
    best = max(LAMBDAS, key=lambda lam: (scores[str(lam)], lam))
    return best, scores


def tfidf(train_docs: list[str], *others: list[str]) -> list[np.ndarray]:
    def grams(doc: str) -> list[str]:
        toks = re.findall(r"[a-z0-9']+", doc.lower())
        return toks + ["%s %s" % p for p in zip(toks, toks[1:], strict=False)]
    df = Counter(g for doc in train_docs for g in set(grams(doc)))
    vocab = {g: i for i, g in enumerate(sorted(df))}
    idf = np.array([math.log((1 + len(train_docs)) / (1 + df[g])) + 1 for g in sorted(df)])

    def mat(docs: list[str]) -> np.ndarray:
        M = np.zeros((len(docs), len(vocab)))
        for i, doc in enumerate(docs):
            for g, c in Counter(grams(doc)).items():
                if g in vocab:
                    M[i, vocab[g]] = c
        M *= idf
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return [mat(d) for d in (train_docs, *others)]


def doc(r: dict[str, Any]) -> str:
    return "%s %s" % (r["state"]["request"], " ".join(t.replace("_", " ") for t in r["state"]["tools"]))


# ---------------------------------------------------------------- metrics
def auroc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)  # binary: confidence lies in [0.5, 1]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def metrics(p: np.ndarray, y: np.ndarray, probabilistic: bool = True) -> dict[str, Any]:
    pred = p >= 0.5
    return {"accuracy": float((pred == y).mean()), "auroc": auroc(p, y),
            "missed_harmful": int((~pred & (y == 1)).sum()), "false_alarms": int((pred & (y == 0)).sum()),
            "ece": ece(p, y) if probabilistic else None}


def paired(a: np.ndarray, b: np.ndarray, y: np.ndarray, reps: int = 2000) -> dict[str, Any]:
    ca, cb = ((a >= 0.5) == y).astype(float), ((b >= 0.5) == y).astype(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(reps, len(y)))
    diffs = (ca[idx] - cb[idx]).mean(1)
    return {"difference": float(ca.mean() - cb.mean()), "ci95": [float(np.quantile(diffs, 0.025)),
                                                                 float(np.quantile(diffs, 0.975))],
            "only_a_correct": int(((ca == 1) & (cb == 0)).sum()), "only_b_correct": int(((ca == 0) & (cb == 1)).sum())}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / "vegalab_runs" / "agentharm_baselines"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "vegalab" / "agentharm_baselines_0.8b"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    cal = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "test"]
    ordered = cal + test
    feats = extract(ordered, Path(args.cache))
    nc = len(cal)
    y_cal = np.array([r["gold_label"] == "harmful" for r in cal], dtype=float)
    y = np.array([r["gold_label"] == "harmful" for r in test], dtype=float)
    vc, vt = feats["vega"][:nc], feats["vega"][nc:]
    pc, pt = feats["probe"][:nc], feats["probe"][nc:]
    probs: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}

    from vegaml.ttt import TTTHead
    head = TTTHead(LABELS).fit(vc, [r["gold_label"] for r in cal])
    probs["vega-head"] = np.array([head.predict(x)["harmful"] for x in vt])
    saved = {json.loads(line)["example_id"]: json.loads(line)["probabilities"]["harmful"]
             for line in (MAIN_RUN / "vega-head.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    ref = np.array([saved[r["example_id"]] for r in test])
    info["vega-head"] = {"max_abs_diff_vs_main_run": float(np.abs(ref - probs["vega-head"]).max())}

    lam, cv = cv_lambda(vc, y_cal, lambda a, b: standardize(a, b))
    a, b = standardize(vc, vt)
    probs["vegafeat-lr"] = predict_lr(fit_lr(a, y_cal, lam), b)
    info["vegafeat-lr"] = {"lambda": lam, "cv_accuracy": cv}

    a, b = standardize(vc, vt)
    cents = [a[y_cal == c].mean(0) for c in (0, 1)]
    cos = np.stack([b @ c / (np.linalg.norm(b, axis=1) * np.linalg.norm(c) + 1e-12) for c in cents], 1)
    centroid_score = cos[:, 1] - cos[:, 0]
    probs["vegafeat-centroid"] = (centroid_score > 0).astype(float)  # a hard decision, not a probability
    info["vegafeat-centroid"] = {"note": "hard decisions only; AUROC from cosine margin; no probabilities"}

    lam, cv = cv_lambda(pc, y_cal, lambda a, b: standardize(a, b))
    a, b = standardize(pc, pt)
    probs["probe-lr"] = predict_lr(fit_lr(a, y_cal, lam), b)
    info["probe-lr"] = {"lambda": lam, "cv_accuracy": cv, "dim": int(pc.shape[1])}

    dc, dt = [doc(r) for r in cal], [doc(r) for r in test]
    lam, cv = cv_lambda(np.arange(nc)[:, None], y_cal,
                        lambda a, b: tfidf([dc[i] for i in a[:, 0]], [dc[i] for i in b[:, 0]]))
    a, b = tfidf(dc, dt)
    probs["tfidf-lr"] = predict_lr(fit_lr(a, y_cal, lam), b)
    info["tfidf-lr"] = {"lambda": lam, "cv_accuracy": cv, "vocab": int(a.shape[1])}

    results = {}
    for name, p in probs.items():
        m = metrics(p, y, probabilistic=name != "vegafeat-centroid")
        if name == "vegafeat-centroid":
            m["auroc"] = auroc(centroid_score, y)
        results[name] = {**m, **info.get(name, {})}
    comparisons = {"vega-head vs %s" % k: paired(probs["vega-head"], probs[k], y) for k in probs if k != "vega-head"}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": "agentharm-v1",
        "n_calibration": nc, "n_test": len(test), "feature_dims": {"vega": int(vc.shape[1]), "probe": int(pc.shape[1])},
        "seed": SEED, "systems": results, "paired_vs_head": comparisons}, indent=2), encoding="utf-8")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(test):
            fh.write(json.dumps({"example_id": r["example_id"], "gold_label": r["gold_label"],
                                 "p_harmful": {k: round(float(v[i]), 6) for k, v in probs.items()}}) + "\n")
    for name, m in results.items():
        print("%-18s acc %.3f auroc %.3f missed %3d false-alarms %3d" % (
            name, m["accuracy"], m["auroc"], m["missed_harmful"], m["false_alarms"]))
    for k, v in comparisons.items():
        print("%-34s %+.3f [%+.3f, %+.3f]" % (k, v["difference"], *v["ci95"]))


if __name__ == "__main__":
    main()
