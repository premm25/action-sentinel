# Vega lab results

**Status: small-sample verification run, not a benchmark.** All numbers come from real inference on
2026-10-18. The raw predictions and run metadata are in `reports/vegalab/pub-v1-n20_cpu_0.8b/` and
`reports/vegalab/rq7_governance_vegaml_0.8b/`. Re-run `action-sentinel vega analyze <dir>` to recompute.

| | |
|---|---|
| model | VegaML 0.8B (`nandakishorm/vega-08b-public-intents`, engine sha256 `09c4c5fe…`), backbone `Qwen/Qwen3.5-0.8B`, float32 |
| software | vegaml 0.9.0, transformers 5.19.0, torch 2.14.1+cpu, Python 3.14.6 |
| hardware | Intel i7-8665U, 4 cores, 15.7 GB RAM, no GPU (other programs running; see latency caveat) |
| data | `pub-v1`, first **20 test + 20 calibration examples per task** (12 task files, 420 predictions per system), seed 20261015 |
| completeness | 4 systems × 420 = 1,680 predictions, **0 errors**; both VegaML systems recorded `live=True` |

With n = 20 per task, one example is 5 percentage points, and most per-task differences have wide
intervals. Only differences whose paired 95% interval excludes zero are called out below.

## RQ1: accuracy among predefined choices (test split)

| task | options | VegaML off | VegaML auto | backbone LM | majority |
|---|---:|---:|---:|---:|---:|
| sst2 | 2 | **0.95** | **0.95** | 0.50 | 0.60 |
| dbpedia14 | 14 | **0.85** | **0.85** | 0.30 | 0.00 |
| dbpedia14_k2 / k4 / k8 | 2 / 4 / 8 | 0.95 / 0.90 / 0.85 | same | 0.85 / 0.65 / 0.65 | 0.40 / 0.00 / 0.00 |
| yelp_review_stars (5 levels) | 5 | **0.65** | **0.65** | 0.20 | 0.15 |
| prompt_injections | 2 | 0.70 | 0.70 | 0.55 | 0.55 |
| glue_rte | 2 | 0.65 | 0.65 | 0.50 | 0.50 |
| boolq | 2 | 0.60 | 0.60 | 0.65 | 0.65 |
| superglue_cb | 3 | 0.55 | 0.55 | 0.30 | 0.55 |
| massive_intent_en | 60 | 0.50 | 0.45 | 0.05 | 0.00 |
| anli_r3 (adversarial) | 3 | 0.25 | 0.25 | **0.50** | 0.30 |
| **macro mean (12 files)** | | **0.700** | 0.696 | 0.475 | 0.308 |

- VegaML was strongest on topic, sentiment and star-rating tasks, and weakest on **entailment and
  reading** (anli_r3 0.25, at or below the majority baseline; boolq and superglue_cb at majority level;
  balanced accuracy 0.33 on superglue_cb, meaning it effectively predicted one class).
- On anli_r3 the backbone baseline did better (0.50 vs 0.25), though the paired interval includes zero.

## RQ4: VegaML vs its own frozen backbone (same tokens in, different decision method)

Paired differences, VegaML-off minus backbone-optscore, where the 95% bootstrap interval excludes zero:
dbpedia14 **+0.55** [+0.35, +0.75], massive_intent_en **+0.45** [+0.25, +0.65], sst2 **+0.45** [+0.20, +0.70],
yelp **+0.45** [+0.10, +0.75], dbpedia14_k8 +0.20 [+0.05, +0.40]. No task showed a significant difference
in the other direction (anli_r3 −0.25 [−0.60, +0.10]).

**Interpretation, with its limit.** The backbone receives exactly the prompt tokens VegaML reads, so
these gains come from the decision mechanism, VegaML's trained engine, relative to zero-shot option
likelihood. They don't show that the *physics* formulation is what helps: any trained head on the same
features might also beat zero-shot scoring, and that comparison wasn't run. The baseline itself is
weak (macro 0.475) and showed a bias toward particular options (for example, it predicted
`alarm_query`, the first intent, on most massive examples), so a stronger zero-shot prompting
method would narrow the gap.

## RQ2: do the probabilities match correctness?

| | VegaML off | VegaML auto | backbone LM | majority |
|---|---:|---:|---:|---:|
| macro ECE (10 bins) ↓ | 0.250 | 0.241 | 0.118 | 0.125 |

- **VegaML was the best classifier but the worst calibrated.** Its confidence was higher than its accuracy:
  ECE 0.41 on anli_r3, 0.46 on dbpedia14 (that one *under*-confident: 0.85 accurate, mostly below the
  abstain threshold), 0.24 on glue_rte. The backbone's low ECE is mostly an artefact of near-uniform
  probabilities matching its low accuracy, not of being informative.
- **Ranking quality was mostly good:** VegaML's AURC (lower is better) beat the backbone on 8 of the 9
  base tasks, for example sst2 0.011 vs 0.307, dbpedia14 0.049 vs 0.735. The exception was anli_r3
  (0.810 vs 0.544). Its confidence *orders* examples well even where its absolute values are off.
- **A temperature fitted on 20 calibration examples helped where VegaML was badly overconfident**
  (anli_r3 ECE 0.41 → 0.13, superglue_cb 0.32 → 0.14, glue_rte 0.24 → 0.11, dbpedia14 0.46 → 0.13) and
  hurt on yelp (0.17 → 0.26). With 20 calibration examples per task these are indications only.
- VegaML's calibration was fitted by its authors on their data; these results show it doesn't
  transfer uniformly to other tasks.

## RQ3: abstention and conformal sets (VegaML native signals, default risk 0.1)

- **Abstention helped when it fired on genuinely hard items:** prompt_injections (abstained 2, accuracy
  on accepted 0.78 vs 0.70 overall), glue_rte (0.71 accepted vs 0.65), massive with adapters auto (0.64
  accepted vs 0.45). On dbpedia14_k2/k4/k8, accepted items were **100%** correct.
- **It was poorly matched to some tasks:** on dbpedia14 VegaML abstained on 18 of 20 examples even
  though it was right on 15 of those 18 (coverage 0.10). On yelp it abstained on 15 of 20 (it was right on 9).
  The threshold (0.596) is a fixed checkpoint value, so tasks with many options or ordinal answers
  produce lower top-option confidence and abstain more.
- **The conformal guarantee did not transfer:** at α = 0.1 the sets should contain the right answer about
  90% of the time. Measured coverage: sst2 1.00, dbpedia14_k2/k4 1.00, yelp 0.95, dbpedia14 0.85, glue_rte,
  boolq and prompt_injections 0.75, superglue_cb 0.55, anli_r3 0.50, massive 0.45–0.50. The conformal
  thresholds were fitted on VegaML's own data, so these sets are not valid guarantees on new tasks.
- **Unbound** (far from every well) fired on 35% of sst2 and dbpedia14 and 20–45% of the dbpedia variants
  without corresponding errors, so on these tasks it isn't a reliable out-of-distribution signal.

## Adapters: off vs auto

The learned gates routed **only massive_intent_en** through an adapter (`public_intents`, all 40
examples); every other task used the base engine, so auto and off were identical on 11 of 12 task
files. On massive, auto scored 0.45 vs off 0.50 (difference −0.05, interval [−0.25, +0.15]) and abstained
on 9 instead of 18 examples. This is the gating behaving as designed: the adapter applies only to
intent-style questions.

## RQ5: number of options, difficulty, length

Same 20 dbpedia test examples, options = gold plus random distractors:

| options | 2 | 4 | 8 | 14 |
|---|---:|---:|---:|---:|
| VegaML | 0.95 | 0.90 | 0.85 | 0.85 |
| backbone LM | 0.85 | 0.65 | 0.65 | 0.30 |

VegaML degraded gently as options were added; the zero-shot backbone collapsed at 14. By difficulty
(pooled): standard tasks 0.78, adversarial NLI 0.25, wide label space (60) 0.50. By input length:
short 0.70, medium 0.72, long 0.53 (n = 17).

## RQ6: latency and memory (0.8B, CPU)

| | VegaML | backbone LM |
|---|---:|---:|
| peak memory (working set) | 3.70 GB | 5.45 GB |
| median latency, 2-option tasks | 1.0–3.0 s | 2.5–4.8 s |
| median latency, 14 options (dbpedia14) | 3.7 s | 10.1 s |
| median latency, 60 options (massive) | 7.1 s | 37.2 s |

The backbone scores every option separately, so its cost grows with the number of options; VegaML
reads them in one pass. Caveat: the machine was memory-constrained and other programs were running.
The VegaML-auto timings are the cleanest; VegaML-off's anli and boolq predictions partly ran
during an earlier memory-starved session (higher latency). The 4B size and GPU timings were not
measured: they need the Kaggle notebook.

## RQ7: governance independent of model quality

Sentinel's synthetic agent-action set (44 test actions; synthetic, so expected outcomes follow the
reference policy, see `docs/evaluation/METHODOLOGY.md`):

- **VegaML abstained on all 44 actions** in both adapter modes. Its top-option confidence was 0.34–0.42
  on a 3-way question (chance 0.33), and its top answer matched the expected label on 13 of 44
  (about chance). The abstain flag correctly reported that it didn't know. Used naively, it would
  approve nothing and block nothing.
- **Full Sentinel with VegaML still blocked 10 of 10 hard policy violations, with 0 of 34 false approvals
  and 0 false blocks**, because the rules don't depend on the model. The cost: none of the 10 safe
  actions was auto-approved, because an abstaining model means uncertainty and the case goes to review.
- So on this task VegaML 0.8B, as prompted here, added no useful signal, and governance kept the system
  safe regardless. This matches the earlier Llama result in `docs/evaluation/RESULTS.md`.
- In the Phase 1 feasibility check, the same model rated a production database delete `unsafe` at 0.90
  when the action was given as a compact dictionary. The provider's richer state (evidence list,
  source, context) produced near-uniform outputs. The input format matters. It was **not** tuned
  here, because tuning against this test set would leak.

### Follow-up: few-shot head (2026-10-22, `AGENT_FRAMING.md`)

Re-framing the input alone did not help (more confident errors). A VegaML few-shot head (`Vega.fit`)
fitted on the 28 calibration actions raised test accuracy from 0.34 to 0.75. Behind Sentinel's rules, it
auto-approved 8/10 safe actions (previously 0/10) with **0/34 false approvals** and 10/10 hard
violations blocked. Alone, without the rules, it approved 2 unsafe actions. The data is synthetic and
templated, so these numbers are optimistic.

## RQ8: real agent-request safety data (AgentHarm, 2026-10-22, `AGENTHARM.md`)

352 test requests (176 harmful, 176 benign twins), with the head fitted on 64 calibration requests
from disjoint base behaviours; 0 errors, all VegaML systems live.

- **The few-shot head held up on real data:** accuracy 0.841, AUROC 0.920, ECE 0.063, 18 false
  alarms and 38 missed harmful requests. It is +0.105 [+0.065, +0.148] over VegaML-off.
- **VegaML vs backbone was mostly a threshold effect:** at 0.5 the backbone called everything benign
  (0.500 vs VegaML-off 0.736). With thresholds fitted on calibration it reached 0.716 vs 0.705;
  VegaML ranked somewhat better (AUROC 0.831 vs 0.784).
- Adapters (`typed_decisions`) fired on 325/352 in auto mode, with no net accuracy change.
  `unbound` fired on 95% of items in off mode.

### Follow-ups (2026-10-24): `AGENTHARM_BASELINES.md`, `AGENTHARM_SENTINEL.md`

- **Head vs ordinary classifiers**, all fitted on the same 64 calibration examples:

  | classifier | accuracy |
  |---|---:|
  | VegaML's head | 0.841 |
  | logistic regression on VegaML's features | 0.841 (difference +0.000 [−0.023, +0.023]) |
  | nearest centroid on VegaML's features | 0.832 |
  | plain mean-pooled probe of the backbone | 0.781 (head +0.060 [+0.020, +0.099]) |
  | TF-IDF | 0.611 |

  The gain comes from VegaML's prompt-conditioned features, not from its head or engine.
- **Through full Sentinel:** 0/176 harmful requests approved (the head alone: 24/176) and 95/176 benign
  requests auto-approved, with 73% of requests sent to review. Every harmful request the head called
  safe had confidence below the 0.85 threshold; the highest was 0.842, so the margin was thin.

## What these results do not show

- Nothing at benchmark precision: 20 test examples per task. The Kaggle notebook
  (`notebooks/vegalab_kaggle_gpu.ipynb`) runs 500 per task and adds the 4B checkpoint.
- Nothing about tasks VegaML may have seen: its base-engine training data is undisclosed.
- Nothing about whether the physics formulation specifically matters (see RQ4).
- Nothing about real agent traffic. The RQ7 data is synthetic, and AgentHarm (RQ8) is
  benchmark-written single requests, not multi-step agent trajectories.
