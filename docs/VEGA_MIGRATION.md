# Vega Decision Intelligence Lab: audit and migration plan

Status: **Phases 0–8 implemented. Phase 5 has a small-sample CPU run (20 test examples per task, `docs/vegalab/RESULTS.md`); the full GPU benchmark is pending (notebook ready).** The project is VegaML-centred: the lab's decision engine is real VegaML inference through `VegaCore`. This document separates three kinds of statement:
*VegaML native* (verified in the vegaml source), *verified here* (observed in a run on this machine) and
*ours* (implemented or planned in this repository).

## Phase 0: repository audit (2026-10-14)

**Repository:** `action-sentinel` 0.1.0, commits `7d258c4` and `ff202db` on `main`, clean working tree.
**Baseline:** `pytest`: 203 passed, 1 deselected (optional live Ollama test). `ruff` clean.

| module | role today | reuse for Vega lab |
|---|---|---|
| `contracts/` | strict pydantic contracts (proposal, provider result, validation, uncertainty, governance, review, grant, audit, evaluation) | **reuse**: evaluation and governance contracts; add Vega-specific decision contracts |
| `providers/vegaml_provider.py` | VegaML adapter written against the inspected 0.9.0 API, **tested only against a stub**, single-question `choice`, disposition-only | **replace** with a full core adapter (all three types, multiple questions, native output preserved) |
| `providers/ollama.py`, `rule_based.py`, `mock.py`, `registry.py`, `runner.py` | judges for the governance pipeline | keep as **baselines** (deterministic, general LLM) and test doubles |
| `uncertainty.py`, `disagreement.py`, `governance/`, `execution/`, `audit.py`, `storage/` | Sentinel governance | keep as an **optional layer**, separated from the model evaluation |
| `evaluation/metrics.py` | accuracy, P/R/F1, confusion, ECE + reliability, risk–coverage/AURC, Brier, log loss, Wilson CIs, percentiles | **reuse**; add balanced accuracy and per-slice breakdowns |
| `evaluation/lab.py`, `datasets.py`, `report.py` | evaluation of the *governance pipeline* on a disposition dataset | **generalize**: a model-evaluation lab for arbitrary typed tasks, with governance evaluation as one task family |
| `api/`, `cli.py`, `dashboard/` | Sentinel interfaces | extend with model-centred endpoints, commands and pages |

**Main gaps for a VegaML-centred lab:** no real VegaML inference so far; contracts are shaped around one
3-way disposition question; the lab measures governance, not the model; there's no backbone baseline;
the dataset is one synthetic disposition task; no resource or latency profiling of model loading.

## VegaML facts established from the source (vegaml 0.9.0, commit `7803fcf`)

These are **VegaML native**.

- **Loading.** `vegaml.load(repo_or_path=None, device=None, adapters="auto", mode="auto", revision=None,
  token=None, max_len=None)` → `Vega`. `hub.fetch` downloads only `vega_config.json`,
  `engine.safetensors`, `adapters/*` and `config.json` from `nandakishorm/vega-08b-public-intents` (the
  root holds 0.8B; `4b/` holds 4B). No downloaded code is executed. The backbone is then resolved from
  `vega_config.json` (`backbone_name`) and loaded with `transformers.AutoModelForCausalLM`
  (`attn_implementation="sdpa"`), **float32 on CPU/MPS** and bf16/fp16 on CUDA.
- **Checkpoints (published config):** 0.8B → backbone `Qwen/Qwen3.5-0.8B`, `d_model` 1024, 24 layers,
  feature layers [13, 19], option layer [19], engine 14.3M params (57 MB). 4B → `Qwen/Qwen3.5-4B`, engine
  124 MB. Both ship adapters `public_intents` and `typed_decisions`.
- **Backbone usage.** The LM is frozen and cut off by a forward hook after the deepest captured layer
  (`StopForward`), so the LM head never runs and **no text is generated**. The LM is not an Ollama-style
  generator and can't be served through Ollama.
- **Input schema.** `decide(state, questions, alpha="0.1", risk=None, views=1)`. `state`: a string, or
  JSON-serializable data (serialized compactly). `questions`: `{qid: {"type", "instructions", "criteria"}}`
  where type ∈ {`choice`, `score`, `boolean`}. `choice` criteria: a dict label→description (or a list of
  labels). `score`: an ordered list of level descriptions. `boolean`: optional `{"true","false"}`
  descriptions.
- **Prompt.** One sequence per question: `Observation:\n{state}\n\nMeasurement ({type}): {instructions}\n
  Possible outcomes:\n* {option}…\nOutcome:`. Option texts are part of the input.
- **Mechanism (`VegaEngine`).** Pooled spans (state, question, final token, each option) from the frozen
  layers → world latent `s` and imagined next state `s_next` → particle position `z0` and momentum `p0` in
  a 64-d space. Each option becomes a Gaussian well (centre, depth, width; `score` levels sit on a 1-d
  rail). Damped symplectic-Euler dynamics with a selective-SSM "thermostat" (learned step size, friction
  and a bounded memory force) run for `n_steps` = 12, with per-item early halting once settled. Logits =
  negative well energies `-(‖z_T−c_k‖²/2σ_k² − log a_k)`.
- **Probabilities.** `softmax(logits / τ)` with `log τ = b + w · [log1p(residual kinetic energy),
  log1p(distance to nearest well), fraction of steps used, log #options]` per question type (`theta`
  from the checkpoint's `calibration` block, fitted by the authors on their data). Clamped to τ ∈ [0.2, 20].
- **Output per question.** `probs` (label → p), `confidence` (= max p), `prediction_set` (split-conformal
  LAC set at `alpha`, top option always kept; `None` if the checkpoint has no table), `abstain`
  (`confidence < abstain_threshold[risk]`; default risk "0.1" → threshold 0.596 for 0.8B), `unbound`
  (distance to the nearest well > `unbound_threshold`, 1.269 for 0.8B), `settle_steps`, `energy`, and
  `choice` / `score` + `expected_score` / `boolean` (p of true). With adapters: `adapter`, `gate`. With
  `views` > 1: `view_agreement`. With `mode="auto"` and fitted TTT heads: `readout`. Top level:
  `latency_ms`, `n_tokens`, `views`, `mode`.
- **Other serving that exists natively:** an HTTP server (`vegaml-serve`, `/v1/decide`), an MCP stdio
  server (`vegaml-mcp-server`), the engine-only MLX (Apple silicon) and ONNX exports (these need pooled
  features from the backbone; they don't replace it), and a vision path (`decide_image`).
- **Hardware (published):** 0.8B ~1.77 GB backbone, 22–74 ms per decision on a T4 GPU; 4B ~9.3 GB.
  CPU timing is **not published**; Phase 1 measures it here.

## Phase 1: feasibility, verified here (2026-10-14)

Command (separate environment outside OneDrive, so torch isn't synced):

```bash
python -m venv C:/Users/sasi/.venvs/vega-lab
C:/Users/sasi/.venvs/vega-lab/Scripts/pip install "vegaml==0.9.0" "transformers==5.19.0" torch
C:/Users/sasi/.venvs/vega-lab/Scripts/python scripts/vega_feasibility.py --adapters auto   # and --adapters off
```

| item | measured |
|---|---|
| packages | vegaml 0.9.0, transformers 5.19.0, torch 2.14.1+cpu, Python 3.14.6 |
| hardware | Intel i7-8665U (4 cores / 8 threads), 15.7 GB RAM, no GPU; torch used 4 threads |
| downloads | engine checkpoint 61 MB (`nandakishorm/vega-08b-public-intents`, snapshot `42ac8d4`), backbone `Qwen/Qwen3.5-0.8B` 1.7 GB; first run 1,119 s including download |
| load time (cached) | 14.4 s (adapters auto), 10.2 s (adapters off) |
| memory (working set) | 227 MB before load → about 3.42 GB after load → about 3.46 GB after 8 decisions; backbone float32 on CPU |
| latency per decision | about 0.8–1.6 s wall-clock on this CPU (76–118 prompt tokens), single question, views=1 |
| early settling | every example used all 12 integration steps (`settle_steps` = 12) |
| determinism | repeated calls returned identical probabilities (to the printed precision) |

Notes:
- transformers logs that Qwen3.5's `causal_conv1d` and `chunk_gated_delta_rule` fall back to slower
  reference PyTorch implementations. CPU latency would likely improve with optimized kernels.
- The default `vegaml.load()` uses `adapters="auto"`. Both shipped adapters (`public_intents`,
  `typed_decisions`) loaded, and their gates route questions per call.
- Native quirk preserved for the adapter: for `boolean` questions, `probs` keys are the strings
  `"False"`/`"True"` while `prediction_set` holds booleans.

**Example outputs (4 hand-written illustrations, not an evaluation):**

| example | adapters auto | adapters off |
|---|---|---|
| route an overdue-invoice email | billing, 0.999 | billing, 0.999 |
| "Refund me today or I am cancelling" → churn risk? | **false**, p(true) = 0.038, confidence 0.962, not abstained | same |
| production checkout down, urgency score 0–3 | 2, confidence 0.508, **abstain**, set {1,2,3} | same |
| agent proposes deleting `prod/orders` | **needs_review**, 0.604, **abstain**, adapter `typed_decisions` (gate 0.998) | **unsafe**, 0.901 |

Two observations worth testing systematically, not conclusions: (1) a confident answer that looks
wrong to a human reader on the churn example; (2) adapter routing changes the decision on the agent
action. Raw native outputs are in `reports/vega_feasibility/`.

## Target architecture

```
                    ┌────────────── Vega Decision Intelligence Lab ──────────────┐
 caller / agent ──► │ DecisionService (Python + FastAPI)                         │
                    │   └─ VegaCore adapter ──► vegaml.Vega.decide (real engine) │  primary decision engine
                    │        native output preserved + normalized DecisionResult │
                    │ Confidence Lab: datasets → inference runs (raw JSONL)       │
                    │        → metrics → reports/dashboards (separate stages)     │
                    │ Baselines: backbone-only (same frozen Qwen, option scoring),│
                    │        deterministic rules, optional general LLM (Ollama)   │
                    └────────────────────────────┬───────────────────────────────┘
                                                 │ optional: decision signal
                                     Sentinel governance layer (existing): hard rules,
                                     review, action-bound authorization, audit
```

## Migration plan

| phase | deliverable | entry criterion |
|---|---|---|
| 0 | audit (this document), baseline tests | — (done) |
| 1 | real VegaML inference on this machine, native output saved, resources measured | packages install, checkpoint downloads |
| 2 | `vega/` core: adapter (injectable runner), contracts (`DecisionRequest`, `CandidateOption`, `DecisionResult`, `UncertaintyMetadata`, `InferenceError`, …), unit tests | Phase 1 shows real output |
| 3 | `DecisionService` + API (`/health`, `/ready`, `/model`, `/decide`), single resident model, timeouts | Phase 2 contracts stable |
| 4 | Confidence Lab v2: dataset schema with provenance and slices, leakage/duplicate checks, raw-prediction JSONL, metrics, run metadata | Phase 3 service usable offline |
| 5 | benchmarks: VegaML (engine, adapters off and auto) vs backbone-only vs rules vs (optional) Ollama, identical items | Phase 4 runner |
| 6 | Sentinel as an optional consumer of Vega decisions; governance evaluated separately | Phase 5 numbers exist |
| 7 | CLI and dashboard reading saved runs only | Phases 4–6 |
| 8 | hardening, docs, measured results | all |

## Decisions that need the owner

1. **Benchmark data.** Synthetic data can't answer RQ1–RQ5 about real performance. Public labelled
   sets used by VegaML's own authors (banking77, clinc150, emotion, ag_news…) overlap with its training
   and adapter data (`public_intents`), so they risk leakage. Options: (a) choose public tasks VegaML
   reports it did not train on, with that caveat; (b) build a small independently labelled set.
2. **Compute.** This machine is CPU-only with 16 GB RAM. The 0.8B model is feasible; the 4B model
   (≈9.3 GB float32 backbone) is not practical here. RQ6 across sizes needs a GPU (for example a free
   Kaggle T4 session, as VegaML's authors used).
3. **Project identity.** Keep the repository name `action-sentinel` with the lab inside it, or rename
   the project to the lab and keep Sentinel as a subpackage.
