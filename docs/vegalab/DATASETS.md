# Datasets

## `pub-v1` (local CPU run)

Built with `action-sentinel vega data build --version pub-v1` (seed `20261015`). Files:
`data/vegalab/pub-v1/*.jsonl` plus `manifest.json`, which records sha256 per file, counts, class
distributions, dropped rows and the check results.

| task | source (HF) | eval split → test | fit split → calibration, dev | type | options | notes |
|---|---|---|---|---|---:|---|
| sst2 | stanfordnlp/sst2 | validation | train | choice | 2 | movie-review sentiment |
| dbpedia14 | fancyzhx/dbpedia_14 | test | train | choice | 14 | encyclopedia topic |
| massive_intent_en | mteb/amazon_massive_intent (en) | test | train | choice | 60 | wide label space |
| glue_rte | nyu-mll/glue (rte) | validation | train | boolean | 2 | entailment |
| boolq | google/boolq | validation | train | boolean | 2 | reading comprehension |
| superglue_cb | aps/super_glue (cb) | validation (all 56) | train | choice | 3 | entailment; small |
| prompt_injections | deepset/prompt-injections | test | train | boolean | 2 | safety; English and German |
| yelp_review_stars | Yelp/yelp_review_full | test | train | score | 5 | ordinal (1–5 stars as levels 0–4) |
| anli_r3 | facebook/anli | test_r3 | train_r3 | choice | 3 | adversarial NLI |
| dbpedia14_k2/k4/k8 | derived | same 100 dbpedia14 test examples | none | choice | 2/4/8 | RQ5: gold plus random distractors, canonical order |

Sizes: 100 test (56 for superglue_cb), 60 calibration and 20 dev per task, sampled uniformly at
random by row index.

**Label definitions.** Each task has one fixed question and fixed option descriptions (see
`action_sentinel/vegalab/tasks.py`), written by us and shown identically to every system. Gold labels
are mapped from the dataset's own label column. Labels outside the defined range (for example `-1`
for hidden labels) are rejected, never wrapped.

## Exclusions and leakage

- **Excluded** because VegaML's shipped adapters were trained on them (`LocalLLaMA/typed-decisions`):
  banking77, clinc150, apibank, appliances, esci, when2call. **Excluded** because VegaML's fine-tuning
  example uses them: emotion, ag_news. trec was unavailable as parquet.
- **Not ruled out:** VegaML's *base* engine training corpus (about 1.2M items) is not publicly documented,
  so overlap with any public dataset is possible. All results carry this caveat.
- **Checks** (`action-sentinel vega data check`; run before every experiment, which refuses to start
  if any fail): unique example ids; every gold label among the options; no calibration or dev input
  text equal to any test input (normalized); one fixed question per task, so no per-example wording
  can carry the label; no label field copied into any input.
- Calibration data is used only to fit the post-hoc temperature in the analysis. Nothing (prompts,
  options, thresholds, baselines) was tuned on test data.

## Larger GPU dataset

The Kaggle notebook builds `pub-n500-s20261015` with the same builder and seed (500 / 200 / 40 per
task). Larger samples are drawn independently, so they aren't a superset of `pub-v1`.
