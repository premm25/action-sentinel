# Vega Decision Intelligence Lab: architecture

The lab lives inside the `action-sentinel` repository as `action_sentinel/vegalab/`. **VegaML is the
primary decision engine.** Sentinel's governance layer is an optional consumer of VegaML decisions.

## What is VegaML's and what is ours

| VegaML native (vegaml 0.9.0, unchanged) | Ours (this repository) |
|---|---|
| Loading (`vegaml.load`), frozen Qwen3.5 perception, prompt construction (`build_prompt`) | `LiveVegaRunner`: loads once, switches adapter mode at runtime |
| The physics decision engine (wells, damped dynamics, settling) | nothing: the engine is never reimplemented or modified |
| Thermal calibration, conformal sets, abstain and unbound flags, adapters and gates | `VegaCore`: validation, timeouts, typed errors, normalization that keeps the native output |
| `decide(state, questions, alpha, risk, views)` and its answer fields | Contracts: `DecisionRequest`, `CandidateOption`, `DecisionQuestion`, `DecisionEvidence`, `UncertaintyMetadata`, `QuestionDecision`, `DecisionResult`, `InferenceError`, `EvaluationRecord` |
| | Decision service and `/vega/*` API, Confidence Lab v2, baselines, CLI, dashboard tab, Sentinel provider |

## Components

```
                    caller / agent
                          │  DecisionRequest (typed, validated)
                          ▼
   /vega/decide ──► VegaDecisionService (one resident model, readiness, accounting)
                          │
                          ▼
                    VegaCore ── validate ─► runner.decide (under a lock, with a timeout)
                          │                        │
                          │              LiveVegaRunner ──► vegaml.Vega.engine.decide  (REAL inference)
                          │              FakeVegaRunner ──► scripted dicts, live=False (tests only)
                          ▼
                    DecisionResult: normalized fields + `native` (exact vegaml output) + live_inference flag
                          │
             ┌────────────┴─────────────┐
             ▼                          ▼
   Confidence Lab v2              Sentinel (optional): VegaMLDecisionProvider → validation, governance,
   datasets → runner → analysis    review, authorization, audit. Hard rules always win.
   → report / CSV / dashboard
```

| module | responsibility |
|---|---|
| `vegalab/contracts.py` | project-level contracts; unknown fields are None, never 0; abstention never hidden |
| `vegalab/vega_core.py` | `VegaRunner` protocol, `LiveVegaRunner`, `VegaCore` |
| `vegalab/testing.py` | `FakeVegaRunner` (test double, `live = False`) |
| `vegalab/service.py`, `vegalab/api.py` | decision service and FastAPI router (`/vega/health`, `/vega/ready`, `/vega/model`, `/vega/decide`) |
| `vegalab/tasks.py`, `vegalab/datasets.py` | public task registry and exclusions; dataset building, provenance, quality and leakage checks |
| `vegalab/systems.py` | `VegaSystem`, `BackboneOptionScorer`, `MajorityBaseline` |
| `vegalab/runner.py`, `vegalab/experiment.py` | inference runs to raw JSONL plus run metadata; controlled multi-system experiments |
| `vegalab/analysis.py`, `vegalab/report.py` | metrics from saved predictions; Markdown, CSV and JSON reports |
| `providers/vegaml_provider.py` | VegaML as a Sentinel provider (one shared model for `vegaml-off` and `vegaml-auto`) |

## Model lifecycle

- One model per process, loaded once (eagerly or on first request). Calls are serialized by a lock
  because vegaml keeps per-call state on the model, as its own server documents.
- Adapter mode is switched per call with `engine.set_adapters(mode)`. A live test verifies that `off`
  after loading with adapters reproduces a checkpoint loaded with `adapters="off"` (probabilities equal
  within 1e-4).
- A timeout returns `TIMEOUT` to the caller. The underlying call can't be killed, and the next request
  waits for it.
- The `/vega/*` routes exist only when `vegalab.enabled: true` (config) or a service is injected, so the
  default API never loads a 3.5 GB model unexpectedly.

## Error model

`INVALID_INPUT` (422) · `INPUT_TOO_LONG` (413) · `MODEL_NOT_READY` / `MODEL_LOAD_FAILED` (503) ·
`TIMEOUT` (504) · `INFERENCE_FAILED` (500) · `MALFORMED_NATIVE_OUTPUT` (502: VegaML returned something
outside the requested options or invalid probabilities).
