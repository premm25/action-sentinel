"""Inference runs: one system over dataset examples, raw predictions to JSONL, run metadata to JSON.

Inference only. Metrics are computed separately (analysis.py) from the saved files, so they can be
recomputed without re-running models. Runs resume: examples already present in the output file are skipped.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from action_sentinel import __version__

from .contracts import EvaluationRecord, InferenceError, VegaInferenceError


def peak_rss_mb() -> float | None:
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t)] + \
                           [(n, ctypes.c_size_t) for n in ("a", "b", "c", "d", "e", "f")]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return round(pmc.PeakWorkingSetSize / 2**20, 1)
            return None
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 / (1024 if sys.platform == "darwin" else 1), 1)
    except Exception:
        return None


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(),
                           "processor": platform.processor(), "cpu_count": os.cpu_count(),
                           "action_sentinel": __version__}
    for mod in ("torch", "transformers", "vegaml", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["cuda"] = torch.cuda.is_available()
        env["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        env["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return env


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("status") == "ok":
                out.add(rec["example_id"])
    return out


def run_system(system: Any, examples: Iterable[dict[str, Any]], out_dir: str | Path, run_id: str,
               dataset_manifest: dict[str, Any] | None = None, keep_native: bool = True,
               progress_every: int = 25) -> Path:
    """Run `system.predict` over examples; append EvaluationRecords to <out_dir>/<system>.jsonl."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / ("%s.jsonl" % system.name)
    done = _done_ids(pred_path)
    examples = [e for e in examples if e["example_id"] not in done]
    started = datetime.now(timezone.utc).isoformat()
    t_all = time.perf_counter()
    n_ok = n_err = 0
    # Rewrite the file without previous error lines, so a retry replaces them.
    if pred_path.exists():
        kept = [ln for ln in pred_path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and json.loads(ln).get("status") == "ok"]
        pred_path.write_text("".join(ln + "\n" for ln in kept), encoding="utf-8")
    with pred_path.open("a", encoding="utf-8", newline="\n") as fh:
        for i, ex in enumerate(examples, 1):
            labels = [o["label"] for o in ex["question"]["options"]]
            base = dict(run_id=run_id, system=system.name, task=ex["task"], split=ex["split"],
                        example_id=ex["example_id"], gold_label=ex["gold_label"], labels=labels,
                        slices=ex.get("slices", {}))
            t0 = time.perf_counter()
            try:
                p = system.predict(ex)
                wall = round(1000 * (time.perf_counter() - t0), 2)
                rec = EvaluationRecord(status="ok", wall_ms=wall, predicted_label=p["predicted_label"],
                                       probabilities=p.get("probabilities"), confidence=p.get("confidence"),
                                       abstain=p.get("abstain"), prediction_set=p.get("prediction_set"),
                                       unbound=p.get("unbound"), settle_steps=p.get("settle_steps"),
                                       adapter=p.get("adapter"), native=p.get("native") if keep_native else None,
                                       **base)
                n_ok += 1
            except VegaInferenceError as exc:
                rec = EvaluationRecord(status="error", error=exc.error, **base)
                n_err += 1
            except Exception as exc:
                rec = EvaluationRecord(status="error", error=InferenceError(
                    code="INFERENCE_FAILED", message="%s: %s" % (type(exc).__name__, str(exc)[:2000])), **base)
                n_err += 1
            fh.write(rec.model_dump_json() + "\n")
            fh.flush()
            if progress_every and i % progress_every == 0:
                print("  %s %d/%d (%.1fs elapsed)" % (system.name, i, len(examples), time.perf_counter() - t_all),
                      flush=True)
    meta_path = out / ("%s.run.json" % system.name)
    history = json.loads(meta_path.read_text(encoding="utf-8"))["sessions"] if meta_path.exists() else []
    history.append({"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                    "examples_attempted": n_ok + n_err, "ok": n_ok, "errors": n_err, "skipped_already_done": len(done),
                    "wall_seconds": round(time.perf_counter() - t_all, 2), "peak_rss_mb": peak_rss_mb()})
    meta = {"run_id": run_id, "system": system.name, "system_info": _safe_info(system), "environment": environment(),
            "dataset": _dataset_ref(dataset_manifest), "predictions_file": pred_path.name,
            "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(), "sessions": history}
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return pred_path


def _safe_info(system: Any) -> dict[str, Any]:
    try:
        return system.info()
    except Exception as exc:
        return {"error": str(exc)}


def _dataset_ref(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    return {"version": manifest.get("version"), "seed": manifest.get("seed"),
            "task_sha256": {k: v.get("sha256") for k, v in manifest.get("tasks", {}).items()}}
