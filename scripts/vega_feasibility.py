"""Phase 1 feasibility: run genuine VegaML inference and record the native output untouched.

    <vega-venv>/python scripts/vega_feasibility.py [--model 0.8b] [--device cpu] [--out reports/vega_feasibility]

What it does, and nothing more:
  1. records the environment (versions, CPU, threads);
  2. loads the checkpoint with vegaml.load(...) (downloads weights from the Hugging Face Hub on first run);
  3. runs one decision of each native question type (choice, boolean, score) through Vega.decide(..., mode="engine");
  4. saves the exact dictionaries returned by vegaml, plus wall-clock timings and process memory, to JSON.

No output field is renamed, filled in or post-processed. The examples are hand-written illustrations,
not an evaluation set.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def rss_mb() -> float | None:
    """Resident memory of this process (Windows: working set; elsewhere: ru_maxrss as peak)."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = ctypes.c_void_p   # a 64-bit pseudo-handle; the default int truncates it
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.WorkingSetSize / 2**20, 1)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


EXAMPLES = [
    {
        "name": "choice_routing",
        "state": {"from": "billing@acme.com", "subject": "Invoice overdue", "body": "Third notice. Pay now."},
        "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "payments and invoices", "technical": "product faults",
                                            "sales": "new business"}}},
    },
    {
        "name": "boolean_churn",
        "state": "I was billed twice this month. Refund me today or I am cancelling my subscription.",
        "questions": {"churn": {"type": "boolean", "instructions": "Is this customer at risk of leaving?",
                                "criteria": {"true": "shows intent to cancel", "false": "no such signal"}}},
    },
    {
        "name": "score_urgency",
        "state": "Production checkout is returning HTTP 500 for every customer since 10 minutes ago.",
        "questions": {"urgency": {"type": "score", "instructions": "How urgent is this incident?",
                                  "criteria": ["not urgent", "can wait a day", "same day", "immediately"]}},
    },
    {
        "name": "agent_action_disposition",
        "state": {"agent": "infra-agent", "action_type": "delete_database", "target": "prod/orders",
                  "payload": {"database": "orders", "reason": "free disk space"}},
        "questions": {"disposition": {"type": "choice",
                                      "instructions": "How should this proposed agent action be handled?",
                                      "criteria": {"safe": "routine, low impact, reversible",
                                                   "needs_review": "impactful or ambiguous; a human should decide",
                                                   "unsafe": "destructive or irreversible; must not run"}}},
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.8b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--adapters", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--out", default="reports/vega_feasibility")
    args = ap.parse_args()

    import torch
    import transformers
    import vegaml

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 1 feasibility: genuine VegaML inference, native output preserved",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "transformers": transformers.__version__, "vegaml": vegaml.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "model_request": {"model": args.model, "device": args.device, "mode": "engine", "adapters": args.adapters},
        "memory_mb": {"before_load": rss_mb()},
        "runs": [],
    }
    t0 = time.perf_counter()
    v = vegaml.load(args.model, device=args.device, mode="engine", adapters=args.adapters)
    record["load_seconds"] = round(time.perf_counter() - t0, 2)
    record["memory_mb"]["after_load"] = rss_mb()
    eng = v.engine
    record["checkpoint"] = {
        "model_dir": str(eng.model_dir), "backbone": str(eng.backbone),
        "backbone_dtype": str(next(eng.perc.model.parameters()).dtype),
        "feature_layers": eng.cfg.get("feature_layers"), "option_layers": eng.cfg.get("option_layers"),
        "n_steps": eng.cfg.get("n_steps"), "latent_dim": eng.cfg.get("latent_dim"),
        "max_len": eng.cfg.get("max_len"), "default_risk": eng.risk,
        "adapters": [a["name"] for a in eng.adapters], "engine_sha256": eng.base_sha256(),
        "repr": repr(v),
    }
    print("loaded in %.1fs: %s" % (record["load_seconds"], record["checkpoint"]["repr"]), flush=True)

    for ex in EXAMPLES:
        for attempt in ("cold" if not record["runs"] else "warm", "repeat"):
            t = time.perf_counter()
            out = v.decide(ex["state"], ex["questions"], mode="engine")
            wall = round(1000 * (time.perf_counter() - t), 1)
            record["runs"].append({"example": ex["name"], "attempt": attempt, "wall_ms": wall,
                                   "input": {"state": ex["state"], "questions": ex["questions"]},
                                   "native_output": out})
            qid = next(iter(ex["questions"]))
            a = out["answers"][qid]
            shown = a.get("choice", a.get("score", a.get("boolean")))
            print("%-26s %-6s %7.1f ms  -> %s  conf=%.3f abstain=%s unbound=%s set=%s steps=%s" % (
                ex["name"], attempt, wall, shown, a["confidence"], a["abstain"], a["unbound"],
                a.get("prediction_set"), a.get("settle_steps")), flush=True)
    record["memory_mb"]["after_runs"] = rss_mb()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("feasibility_%s_adapters-%s_%s.json" % (args.model, args.adapters,
                                                                datetime.now().strftime("%Y%m%d_%H%M%S")))
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
