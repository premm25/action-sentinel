"""Run a controlled experiment: the same examples through every requested system, one after another.

Order: majority (instant) -> vega-off -> vega-auto (one loaded model, adapter mode switched at runtime)
-> backbone-optscore (loaded only after VegaML is released, so the two models never share memory).
"""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .datasets import check_dataset, load_examples
from .runner import run_system

SYSTEMS = ("majority", "vega-off", "vega-auto", "vega-head", "backbone-optscore")


def run_experiment(dataset_dir: str | Path, out_dir: str | Path | None = None, systems: list[str] | None = None,
                   tasks: list[str] | None = None, splits: tuple[str, ...] = ("test", "calibration"),
                   limit: int | None = None, model: str = "0.8b", device: str = "cpu") -> Path:
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    checks = check_dataset(dataset_dir)
    if not all(c["passed"] for c in checks.values()):
        raise RuntimeError("dataset checks failed: %s" % {k: v for k, v in checks.items() if not v["passed"]})
    systems = systems or list(SYSTEMS)
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        raise ValueError("unknown systems %s (known: %s)" % (sorted(unknown), SYSTEMS))
    task_files = sorted(f for f in dataset_dir.glob("*.jsonl") if tasks is None or f.stem in tasks)
    examples: list[dict[str, Any]] = []
    fit: list[dict[str, Any]] = []
    for f in task_files:
        rows = load_examples(f)
        per_task = [r for r in rows if r["split"] in splits]
        if limit:
            per_task = [r for s in splits for r in [x for x in per_task if x["split"] == s][:limit]]
        examples += per_task
        fit += [r for r in rows if r["split"] in ("calibration", "dev")]
    run_id = "vegalab-%s-%s" % (manifest["version"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    out = Path(out_dir) if out_dir else Path("reports/vegalab") / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.json").write_text(json.dumps({
        "run_id": run_id, "dataset_dir": str(dataset_dir), "dataset_version": manifest["version"],
        "systems": systems, "tasks": [f.stem for f in task_files], "splits": list(splits), "limit": limit,
        "model": model, "device": device, "n_examples_per_system": len(examples)}, indent=2), encoding="utf-8")

    if "majority" in systems:
        from .systems import MajorityBaseline
        print("system majority", flush=True)
        run_system(MajorityBaseline(fit), examples, out, run_id, manifest, keep_native=False)
    vega_modes = [s.split("-", 1)[1] for s in systems if s in ("vega-off", "vega-auto")]
    if vega_modes or "vega-head" in systems:
        from .systems import VegaHeadSystem, VegaSystem
        from .vega_core import LiveVegaRunner, VegaCore
        runner = LiveVegaRunner(model, device=device)
        runner.load()
        core = VegaCore(runner, model_id="vegaml-%s" % model)
        for mode in vega_modes:
            print("system vega-%s" % mode, flush=True)
            run_system(VegaSystem(core, mode), examples, out, run_id, manifest)  # type: ignore[arg-type]
        if "vega-head" in systems:
            print("system vega-head", flush=True)
            # Head examples come from the calibration split; predictions on that same split would be
            # in-sample, so the head is only run on the other splits.
            head_examples = [e for e in examples if e["split"] != "calibration"]
            run_system(VegaHeadSystem(core, fit), head_examples, out, run_id, manifest)
        core.close()
        del core, runner
        gc.collect()
    if "backbone-optscore" in systems:
        from .systems import BackboneOptionScorer
        bb = BackboneOptionScorer(model, device=device)
        bb.load()
        print("system backbone-optscore", flush=True)
        run_system(bb, examples, out, run_id, manifest)
        del bb
        gc.collect()
    return out
