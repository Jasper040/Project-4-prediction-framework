# Claude Code Prompts — Coding + Writing Tracks

Copy-paste each prompt into Claude Code from the `novabank_retention/` folder.
Each prompt is self-contained — Claude Code starts cold each time, so the
context is repeated.

**Workflow per artefact:**
1. Pull the latest from git so you don't overwrite your partner's work.
2. Activate the venv: `source .venv/bin/activate` (or Windows equivalent).
3. Paste the prompt below into Claude Code.
4. Review the diff before accepting.
5. Run the notebook top-to-bottom from a clean kernel (coding tracks only).
6. Verify the testable outcomes.
7. Commit the artefact + any output files.
8. Add a one-paragraph entry to `ai_usage_log.md`.

The coding-track prompts (notebooks 02–09) come first. The writing-track
prompts (notebook 01, executive memo, slide appendix) are at the end —
run those after the coding work is finished and stable.

---

## Prompt — Notebook 02: Data Readiness

```
Project context: I'm building "NovaBank — Predictive Retention via Targeted
Engagement Campaigns" — a Quantic MSBA analytics project. The dataset is
the UCI Bank Marketing dataset (bank-additional-full.csv). Target `y` =
subscribed to a term deposit (treated here as a retention signal, not
literal churn).

The project is already scaffolded. From the project root:
- `src/config.py` defines paths, seeds, business assumptions.
- `src/data.py` has `load_raw`, `light_clean`, `split_train_test`,
  `save_interim`, `load_interim`, `feature_columns`.
- `notebooks/02_data_readiness.ipynb` is a stub with section headers
  and starter cells.

Your job: fill in `notebooks/02_data_readiness.ipynb` so it runs
top-to-bottom from a clean kernel and produces the locked train/test
split that all downstream notebooks depend on.

Required behaviour:
1. Use the helpers in `src/data.py` — do NOT reimplement loading,
   cleaning, or splitting inline.
2. After loading raw data, print: shape, dtypes, memory usage, null
   counts per column, target distribution (counts and proportions).
3. Apply `data.light_clean` and verify: target is 0/1 integer; pdays
   sentinel (999) is replaced with NaN; `was_previously_contacted` flag
   added; whitespace stripped from string columns.
4. Run the stratified 80/20 split via `data.split_train_test`.
5. Verify both splits have positive rates within 0.1 percentage points
   of each other and within 0.2 of the overall ~11.3%.
6. Save both splits to `data/interim/` via `data.save_interim`.
7. Leave the data dictionary markdown cell as-is (it's already drafted).
8. Add a final markdown cell summarising: total rows, train/test sizes,
   class balance, any data quality concerns observed.

Testable outcomes:
- `data/interim/train.parquet` exists, ~32,950 rows, target proportion
  between 0.111 and 0.115.
- `data/interim/test.parquet` exists, ~8,238 rows, target proportion
  between 0.111 and 0.115.
- All notebook cells have outputs (don't strip them).
- The notebook runs end-to-end from a fresh kernel without errors.

Constraints:
- Do not modify files outside `notebooks/02_data_readiness.ipynb`.
- Do not import anything that's not already in `requirements.txt`.
- Do not impute, scale, or encode features here — that happens inside
  the modelling pipeline.
```

---

## Prompt — Notebook 03: EDA

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
Dataset is the UCI Bank Marketing dataset, target `y` = subscribed to
term deposit (~11.3% positive rate). Notebooks 01 and 02 are done; the
locked train/test split is at `data/interim/`.

The project is scaffolded:
- `src/config.py`, `src/data.py`, `src/features.py`, `src/viz.py`
  are available.
- `viz.py` provides a consistent matplotlib/seaborn style on import.
- `notebooks/03_eda.ipynb` is a stub with section headers.

Your job: fill in `notebooks/03_eda.ipynb` to surface 3–5 slide-ready
insights from the **train set only** (do not load test data).

Required behaviour:
1. Load only the train split via `data.load_interim()` and ignore the
   test return value. Test data must NOT be opened in this notebook.
2. Confirm class balance with both a count and a proportion plot.
3. Univariate distributions:
   - Histograms for numeric columns (`age`, `campaign`, `previous`,
     `euribor3m`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`,
     `nr.employed`).
   - Bar charts of value counts for categorical columns. Flag any
     near-constant columns or unusually high `unknown` rates.
4. Bivariate vs. target — for each candidate predictor, plot
   subscription rate by category/decile:
   - Categorical: barplot of mean target per level, sorted descending.
   - Numeric: bin into deciles and plot conversion rate per decile.
   - Highlight `poutcome`, `contact`, `month`, `euribor3m`, and `age` —
     these are likely the strongest signals.
5. Time/macro view: plot mean subscription rate by month, and the
   macro indicators (`euribor3m`, `emp.var.rate`, `nr.employed`) by
   month. Note the 2008 crisis signature in the text.
6. Save every figure to `outputs/figures/` via `viz.save_fig` with a
   descriptive name (e.g. `03_eda_target_balance.png`,
   `03_eda_poutcome_lift.png`). These figures will be reused in slides.
7. Add a final markdown cell with 3–5 bulleted "headline insights"
   that are defensible from the charts above. Each bullet must
   reference a specific saved figure by filename.

Testable outcomes:
- At least 8 PNGs saved to `outputs/figures/` with the `03_eda_` prefix.
- The notebook never reads `data/interim/test.parquet`.
- The final markdown cell has 3–5 insights, each with a figure reference.
- The notebook runs end-to-end from a clean kernel.

Constraints:
- Do not load or peek at the test set.
- Do not modify `src/` modules.
- Do not engineer features here (that happens in `src/features.py` and
  the modelling notebooks).
```

---

## Prompt — Notebook 04: Baseline Model (Logistic Regression)

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
Dataset has been split into train/test in `data/interim/`. The target
is `y` (binary, ~11.3% positive). We are building two logistic
regression baselines — one with the leaky `duration` feature and one
without — to demonstrate the leakage gap explicitly.

Available helpers:
- `src/data.py` — `load_interim`, `feature_columns(include_leaky=False)`.
- `src/models.py` — `make_logreg_pipeline()` (returns a Pipeline with
  the preprocessor + LogisticRegression(class_weight='balanced')).
- `src/metrics.py` — `classification_summary` (returns dict with
  ROC-AUC, PR-AUC, F1, Brier, confusion counts, precision, recall).
- `src/viz.py` — `plot_roc`, `plot_pr`, `plot_confusion`, `save_fig`.
- `src/config.py` — paths, RANDOM_SEED.

Your job: fill in `notebooks/04_baseline_model.ipynb` so it trains both
logistic regressions, compares them, and persists the deployable model
+ test-set predictions for downstream notebooks.

Required behaviour:
1. Load train/test via `data.load_interim()`.
2. Build feature matrices for the leaky benchmark (
   `feature_columns(include_leaky=True)`) and the deployable model
   (`feature_columns(include_leaky=False)`).
3. Fit `make_logreg_pipeline()` on each. Predict probabilities on the
   matching test feature set.
4. Run `metrics.classification_summary` at threshold 0.5 for both
   models. Combine into a 2-row DataFrame indexed
   ['Leaky benchmark (with duration)', 'Deployable baseline (no duration)']
   and save to `outputs/tables/baseline_comparison.csv`.
5. Plot ROC, PR, and confusion matrix for the deployable baseline in a
   single 1×3 figure. Save to `outputs/figures/04_baseline_curves.png`.
6. Persist the deployable pipeline + its test-set probabilities via
   joblib to `outputs/models/baseline_logreg.joblib` as a dict
   {'model': <pipeline>, 'proba_test': <np.ndarray>}.
7. Add a final markdown cell with a 2–3 sentence interpretation of
   the leaky-vs-deployable gap, suitable for a slide.

Testable outcomes:
- `outputs/tables/baseline_comparison.csv` exists with two rows.
- The leaky benchmark's ROC-AUC is at least 0.10 higher than the
  deployable baseline's. (Expect leaky ~0.93, deployable ~0.78–0.80.)
- `outputs/figures/04_baseline_curves.png` exists.
- `outputs/models/baseline_logreg.joblib` exists and `joblib.load`
  returns a dict with keys 'model' and 'proba_test'.
- The notebook runs end-to-end from a clean kernel.

Constraints:
- Use the helpers in `src/`. Do not duplicate preprocessing inline.
- Do NOT use the leaky benchmark's predictions for any downstream
  artefact; only the deployable model gets persisted to
  `outputs/models/`.
- Do not tune hyperparameters here — this is the baseline.
```

---

## Prompt — Notebook 05: Improved Model (LightGBM, calibrated)

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
Notebook 04 has produced a logistic regression baseline at
`outputs/models/baseline_logreg.joblib`. Your job is to train a
LightGBM classifier that beats it on PR-AUC while producing
well-calibrated probabilities (the decision framework needs them).

Available helpers:
- `src/data.py` — `load_interim`, `feature_columns(include_leaky=False)`.
- `src/models.py` — `make_lgbm_pipeline(params)`, `calibrate(pipe, X, y, method='isotonic')`,
  `positive_class_weight(y)`.
- `src/metrics.py` — `classification_summary`, `calibration_table`.
- `src/viz.py` — `plot_calibration`, `save_fig`.
- `src/config.py` — `LIGHTGBM_DEFAULTS`, `N_TUNING_TRIALS`,
  `RANDOM_SEED`.

Your job: fill in `notebooks/05_improved_model.ipynb`.

Required behaviour:
1. Load train/test via `data.load_interim()`. Build the deployable
   feature matrices (no `duration`).
2. Run an Optuna study with `N_TUNING_TRIALS` (default 50) trials,
   maximising 5-fold stratified CV `average_precision` on the train
   set. Tune at minimum: `learning_rate`, `num_leaves`,
   `min_child_samples`, `reg_alpha`, `reg_lambda`. Always include
   `scale_pos_weight = positive_class_weight(y_train)`.
3. Refit the best pipeline on the full train set, then wrap in
   `models.calibrate(..., method='isotonic')`. The calibration is
   isotonic, 5-fold CV, on the train set.
4. Predict probabilities on the test set.
5. Build a side-by-side comparison table: load the baseline
   probabilities from `outputs/models/baseline_logreg.joblib` and
   compute `classification_summary` for both. Save the resulting
   DataFrame to `outputs/tables/baseline_vs_improved.csv` with rows
   indexed ['Logistic baseline', 'LightGBM (calibrated)'].
6. Build the calibration table on the test set and plot it via
   `viz.plot_calibration`. Save the plot to
   `outputs/figures/05_calibration.png`.
7. Persist the calibrated model artefact to
   `outputs/models/improved_lgbm.joblib` as a dict {'model':
   <CalibratedClassifierCV>, 'proba_test': <np.ndarray>,
   'best_params': <dict>}.
8. Final markdown cell: 2–3 sentences on (a) where LightGBM improved
   over baseline, (b) what the calibration plot tells you, (c) any
   red flags (instability, suspicious feature dependence).

Testable outcomes:
- `outputs/tables/baseline_vs_improved.csv` shows LightGBM PR-AUC
  ≥ baseline PR-AUC. (Expect baseline ~0.43, LightGBM ~0.50–0.55.)
- `outputs/models/improved_lgbm.joblib` exists and `joblib.load`
  returns a dict with keys 'model', 'proba_test', 'best_params'.
- `outputs/figures/05_calibration.png` exists.
- The Brier score of the LightGBM model is ≤ that of the baseline
  (better calibration).
- The notebook runs end-to-end from a clean kernel.

Constraints:
- Use `scale_pos_weight` for class imbalance, NOT SMOTE or other
  oversampling.
- The test set is for final evaluation only — never use it in tuning
  or calibration.
- Calibration is isotonic, not Platt — Platt scaling underperforms on
  tree models.
- Use `optuna.logging.set_verbosity(optuna.logging.WARNING)` to keep
  output readable.
```

---

## Prompt — Notebook 06: Customer Segmentation (k-means)

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
The improved model is at `outputs/models/improved_lgbm.joblib`. We
want to layer customer segmentation on top so business teams can
interpret model output as actionable customer archetypes.

Available helpers:
- `src/data.py` — `load_interim`, `feature_columns(include_leaky=False)`.
- `src/features.py` — `build_preprocessor(scale_numeric=True)`.
- `src/config.py` — `KMEANS_K_RANGE`, `RANDOM_SEED`, `TARGET_COL`.

Your job: fill in `notebooks/06_segmentation.ipynb`.

Required behaviour:
1. Load train/test via `data.load_interim()`. Build the deployable
   feature matrix (no `duration`, no target).
2. Fit `features.build_preprocessor(scale_numeric=True)` on the train
   set. Transform both train and test. Cache feature names via
   `preprocessor.get_feature_names_out()`.
3. K-search: for each k in `config.KMEANS_K_RANGE`, fit
   `KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)` on the
   transformed train set. Record inertia and silhouette score
   (silhouette can use a 5,000-row sample for speed). Save the
   diagnostic table to `outputs/tables/kmeans_k_search.csv`.
4. Plot inertia and silhouette vs. k in a single 1×2 figure. Save to
   `outputs/figures/06_kmeans_diagnostic.png`. Pick the final k
   (likely 4 or 5) and document the choice in a markdown cell.
5. Refit KMeans at the chosen k. Assign segment labels to train AND
   test rows.
6. Profile each segment on the train set. Aggregate: size, base
   conversion rate, mean age, dominant `job`, dominant `contact`
   channel, mean `euribor3m`. Save to
   `outputs/tables/segment_profiles.csv`, sorted by base rate
   descending.
7. Score the LightGBM model within each test segment: report n,
   base rate, PR-AUC (only if both classes present in segment).
   Save to `outputs/tables/segment_model_perf.csv`.
8. Final markdown: one-line persona description per segment (e.g.
   "Segment 0 — Affluent retirees: older, cellular, prior success,
   conversion 35%, model PR-AUC 0.65 → high-confidence targeting").

Testable outcomes:
- `outputs/tables/kmeans_k_search.csv` has one row per k in the
  search range.
- `outputs/figures/06_kmeans_diagnostic.png` exists.
- `outputs/tables/segment_profiles.csv` has K rows, sorted by
  conversion rate descending.
- `outputs/tables/segment_model_perf.csv` has K rows.
- The notebook runs end-to-end from a clean kernel.
- The chosen k has a justification in markdown.

Constraints:
- Cluster on the SAME preprocessed feature space the model uses.
- Do not include the target in the clustering features.
- `n_init=10` minimum for stability.
- If a segment has only one target class in the test set, report
  PR-AUC as NaN — do not crash.
```

---

## Prompt — Notebook 07: Decision Framework

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
The calibrated LightGBM model is at `outputs/models/improved_lgbm.joblib`.
We need to translate calibrated probabilities into a contact policy
that operations can execute.

Business assumptions (from `src/config.py`, sensitivity-tested in
notebook 08):
- COST_PER_CONTACT_EUR = 5.0 (operations cost per call)
- VALUE_PER_CONVERSION_EUR = 120.0 (margin per successful sub)
- CONTACT_BUDGET_FRACTION = 0.20 (top 20% of customers per cycle)

Available helpers:
- `src/data.py` — `load_interim`.
- `src/metrics.py` — `expected_value`, `ev_curve`, `decile_lift`.
- `src/decision.py` — `threshold_for_top_k`, `threshold_max_ev`,
  `build_decision_rule` (returns a `DecisionRule` dataclass with
  `.describe()`).
- `src/viz.py` — `plot_ev_curve`, `plot_decile_lift`, `save_fig`.

Your job: fill in `notebooks/07_decision_framework.ipynb`.

Required behaviour:
1. Load test data via `data.load_interim()` and the LightGBM
   probabilities via `joblib.load('outputs/models/improved_lgbm.joblib')['proba_test']`.
2. Print the cost / value / budget assumptions clearly.
3. Compute the EV curve via `metrics.ev_curve(y_test, y_proba)` and
   save to `outputs/tables/ev_curve.csv`.
4. Build the budget-constrained decision rule via
   `decision.build_decision_rule(y_test, y_proba)`. Print
   `rule.describe()`.
5. Compute the unconstrained EV-max threshold via
   `decision.threshold_max_ev` and report its EV. State why the
   constrained rule is the recommendation (real ops budgets are
   fixed).
6. Compute the decile lift table via `metrics.decile_lift(y_test,
   y_proba)`. Save to `outputs/tables/decile_lift.csv`.
7. Plot a 1×2 figure: EV curve (with the chosen threshold marked) and
   decile lift bars. Save to `outputs/figures/07_decision_framework.png`.
8. Final markdown — pilot plan with explicit sections:
   - Scope (which segment, which cycle).
   - Treatment vs. control: randomise within top decile.
   - Primary KPI: subscription rate uplift in treatment vs. control.
   - Timeline: 30 days build / 60 days run / 30 days analyse.
   - Rollout decision criteria: scale if uplift >20%, iterate if
     10–20%, kill if <10%.

Testable outcomes:
- `outputs/tables/ev_curve.csv` exists with 99 rows (thresholds
  0.01 to 0.99).
- `outputs/tables/decile_lift.csv` exists with 10 rows.
- `outputs/figures/07_decision_framework.png` exists.
- Decile 1 lift_vs_base ≥ 3.0 (model concentrates positives in the
  top decile).
- The recommended decision rule reports a positive expected_value_eur.
- The notebook prints both the constrained and unconstrained
  thresholds, with the recommendation justified.
- The pilot plan markdown has all 5 sections listed above.

Constraints:
- Use the test set (not train) for the EV / lift calculations —
  this is a final evaluation.
- Recommend the budget-constrained policy; only mention EV-max for
  contrast.
- The pilot must include a control group; without it, downstream
  causal claims fail.
```

---

## Prompt — Notebook 08: Sensitivity Analysis

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
The decision rule from notebook 07 depends on three assumptions:
cost-per-contact (€5), value-per-conversion (€120), and contact-budget
(top 20%). Plus an implicit assumption that the base subscription
rate stays around 11%. This notebook stress-tests all of them.

Available helpers:
- `src/data.py` — `load_interim`.
- `src/decision.py` — `build_decision_rule`, `sensitivity_sweep`.
- `src/metrics.py` — `expected_value`.
- `src/config.py` — `RANDOM_SEED`, default cost/value/budget.

Your job: fill in `notebooks/08_sensitivity_analysis.ipynb`.

Required behaviour:
1. Load test data + LightGBM probabilities (same as notebook 07).
2. Scenario A — cost/value sweep:
   Use `decision.sensitivity_sweep` with:
   - cost_per_contact_grid = [2.5, 5.0, 10.0, 20.0]
   - value_per_conversion_grid = [60, 120, 200, 300]
   Save the resulting long DataFrame to
   `outputs/tables/sensitivity_cost_value.csv`. Pivot into a
   cost × value heatmap of EV and save the heatmap to
   `outputs/figures/08_sensitivity_heatmap.png`.
3. Scenario B — contact-budget sweep:
   For each fraction in [0.05, 0.10, 0.20, 0.30, 0.50], call
   `decision.build_decision_rule(..., contact_budget_fraction=frac)`.
   Tabulate threshold, EV, contacts, successful, wasted, missed.
   Save to `outputs/tables/sensitivity_budget.csv`. Plot EV and
   precision-at-budget vs. budget fraction in a 1×2 figure. Save to
   `outputs/figures/08_sensitivity_budget.png`.
4. Scenario C — recession (base rate halves):
   Downsample positives in the test set to 50% of original count
   (use `np.random.default_rng(RANDOM_SEED)`), keep all negatives,
   reindex y_proba accordingly. Build the decision rule on this
   recession sample. Compare EV, threshold, contacts, and conversion
   to the baseline scenario. Save the comparison to
   `outputs/tables/sensitivity_recession.csv`.
5. Final markdown — "what holds vs. what shifts":
   - 2 bullets on what's robust (the *direction* of the
     recommendation).
   - 2 bullets on what's sensitive (specific thresholds, specific EVs).
   - 1 bullet on actionable risk to flag in the memo.

Testable outcomes:
- `outputs/tables/sensitivity_cost_value.csv` has 16 rows
  (4 costs × 4 values).
- `outputs/tables/sensitivity_budget.csv` has 5 rows.
- `outputs/tables/sensitivity_recession.csv` exists with at least
  baseline + recession rows.
- `outputs/figures/08_sensitivity_heatmap.png` and
  `outputs/figures/08_sensitivity_budget.png` exist.
- The recession scenario shows lower EV than baseline but the rule
  structure (top-K with positive EV) still holds.
- The "what holds vs. what shifts" markdown has the 5 bullets.

Constraints:
- Do not retrain any model. Sensitivity is on the assumptions, not
  the model.
- Use a fixed random seed for the recession downsampling.
- Pivot the cost/value sweep correctly (cost on one axis, value on
  the other) for the heatmap.
```

---

## Prompt — Notebook 09: Explainability + Fairness

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
The calibrated LightGBM is at `outputs/models/improved_lgbm.joblib`.
Compliance officers and front-line ops need to defend the model's
recommendations. This notebook produces SHAP global + local
explanations, a human-readable surrogate ruleset, and a fairness
sanity-check.

Available helpers:
- `src/data.py` — `load_interim`, `feature_columns(include_leaky=False)`.
- `src/features.py` — `build_preprocessor`, `add_engineered_features`
  (adds `age_band`, `economy_contracting`, `high_contact_campaign`).
- `src/config.py` — paths, TARGET_COL.

The calibrated model is a `CalibratedClassifierCV`. To run SHAP, pull
out the inner LightGBM:
```
calibrated.calibrated_classifiers_[0].estimator
```
Then access `.named_steps['preprocessor']` and `.named_steps['classifier']`.

Your job: fill in `notebooks/09_explainability_and_fairness.ipynb`.

Required behaviour:
1. Load train/test, the deployable feature columns, and the
   LightGBM artefact.
2. Pull out the inner pipeline → preprocessor + LGBM. Transform the
   test set. Get `feature_names = preprocessor.get_feature_names_out()`.
3. Run `shap.TreeExplainer` on the LightGBM. Compute SHAP values on
   the transformed test set. If the result is a list (multiclass-
   shaped), pick index 1 for the positive class.
4. Save `shap.summary_plot` (default beeswarm) to
   `outputs/figures/09_shap_summary.png`. Save a SHAP bar plot to
   `outputs/figures/09_shap_bar.png`.
5. Pick TWO test customers deliberately:
   - One with predicted probability > 0.85 (strong contact rec).
   - One with predicted probability < 0.05 (strong skip rec).
   For each, generate a SHAP waterfall plot. Save to
   `outputs/figures/09_shap_local_recommend.png` and
   `outputs/figures/09_shap_local_skip.png`.
6. Surrogate decision tree: fit a `DecisionTreeRegressor(max_depth=3,
   random_state=RANDOM_SEED)` on (X_test_transformed, y_proba). Export
   the tree as text via `sklearn.tree.export_text(..., feature_names=...)`.
   Save to `outputs/tables/surrogate_rules.txt`.
7. Fairness check:
   - Augment the test set with `features.add_engineered_features` to
     get `age_band`.
   - For each value of `age_band` and each value of `education`,
     report: n, base_rate, mean_predicted, PR-AUC (only if both
     classes present), top-20%-contact-rate.
   - Save to `outputs/tables/fairness_by_age.csv` and
     `outputs/tables/fairness_by_education.csv`.
   - Plot a 1×2 figure: contact-rate-by-age-band and
     contact-rate-by-education. Save to
     `outputs/figures/09_fairness.png`.
8. Final markdown — 3-bullet summary for the compliance slide:
   - Top 3 features the model relies on (from SHAP).
   - Any group disparities worth flagging, with the structural reason
     (e.g. "older customers genuinely convert more — disparity is
     justifiable but should be monitored").
   - The fallback policy: if the model is offline, use the depth-3
     surrogate rules as a documented backup.

Testable outcomes:
- 4 figures saved with `09_` prefix (summary, bar, local_recommend,
  local_skip, fairness — actually 5).
- `outputs/tables/surrogate_rules.txt` exists and contains a tree
  representation with at least 4 leaves.
- `outputs/tables/fairness_by_age.csv` has rows for each age_band
  level present in the test data.
- `outputs/tables/fairness_by_education.csv` has rows for each
  education level present in the test data.
- The notebook runs end-to-end from a clean kernel.
- The 3-bullet compliance summary references SHAP outputs and the
  surrogate rules concretely.

Constraints:
- Do not refit the LightGBM. Use the persisted calibrated model.
- For SHAP, use the inner LightGBM, not the calibration wrapper.
- For the local explanations, pick ONE customer above 0.85 and ONE
  below 0.05 — these contrast cleanly. If no test customer hits
  >0.85, lower the threshold to the 99th percentile.
- Frame fairness findings as observations, not verdicts. Do not
  claim the model "is fair" or "is unfair" categorically.
```

---

## After all 8 notebooks are green

Final reproducibility check (do this BEFORE writing the memo / slides):

```
Project context: NovaBank Predictive Retention project. All 9 notebooks
have been filled in. I want to verify the project is reproducible
from scratch.

Your job:
1. Delete every file under `data/interim/`, `data/processed/`,
   `outputs/figures/`, `outputs/models/`, `outputs/tables/`.
2. Restart the kernel and run notebooks 02 through 09 in order, top
   to bottom, no skipped cells.
3. After each notebook, list the new files produced. Confirm they
   match the testable outcomes from the prompts.
4. If any notebook fails, fix the bug in `src/` (not inline) and rerun
   from that notebook onwards.
5. Report the final inventory of files under `outputs/`.

This is the rubric line "reproducible notebook" — if a grader pulls
the repo and can't get the same numbers, the score caps at 4.
```

---

# Writing Track

Run these AFTER notebooks 02–09 are complete and the `outputs/` folder
is stable. Memo and slides are synthesis — they should not introduce
new analysis. If you find yourself wanting to compute something new
mid-memo, go fix the relevant notebook first.

---

## Prompt — Notebook 01: Problem Framing (writing-only)

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
The reframe — "Predictive Retention via Targeted Engagement Campaigns"
— is documented in `README.md`. Notebook 01 is the framing notebook;
it sets up the narrative every other notebook builds on. It contains
ONLY markdown cells. No code.

Files to read for context before writing:
- `README.md` — the canonical project framing.
- `Analytics Methods and Frameworks Project Overview.pdf` (in the
  parent folder) — the Quantic project description and rubric.
- `notebooks/01_problem_framing.ipynb` — the stub with section
  headers already in place.

Your job: fill in the markdown cells of
`notebooks/01_problem_framing.ipynb` with executive-ready content.
The reader is a NovaBank exec, not a TA.

Required behaviour:
1. Section 1.1 — Five-sentence problem brief covering, in order:
   context, decision to be made, who cares, timeframe, constraints.
   It must be EXACTLY five sentences. No more, no fewer.
2. Section 1.2 — Business → analytics translation. State the
   business question, the analytics translation (binary
   classification of campaign response as a retention signal), and
   why we are NOT predicting churn directly (no churn label in the
   data). 4–6 sentences.
3. Section 1.3 — Success measures. Pick exactly two KPIs:
   - Primary: PR-AUC. Justify with the ~11% positive rate making
     ROC-AUC misleading.
   - Secondary: Expected value at top-K decile. Justify as the
     translation of probabilities to euros for executive comparison.
   2–3 sentences per KPI.
4. Section 1.4 — Draft action / policy. State the initial policy
   ("each cycle, contact the top 20% of customers by predicted
   subscription probability"), note that 20% is a starting budget
   that notebook 08 will sensitivity-test against 5%, 10%, and 30%.
5. Section 1.5 — What this framing buys us. Three bullets max:
   real labels (no synthetic churn), clean FP/FN trade-off,
   defensible explanation surface.

Testable outcomes:
- All five markdown sections in the stub are filled with substantive
  content (no "TODO" placeholders, no skeleton text).
- Section 1.1 is exactly five sentences.
- Section 1.3 names PR-AUC as primary and EV at top-K decile as
  secondary, with explicit justification of each.
- The notebook has zero code cells (it's pure writing).
- Total markdown word count between 500 and 900 words.
- Notebook opens cleanly in Jupyter; no parse errors.

Constraints:
- Do not add code cells.
- Do not invent statistics about the dataset — use only what is in
  `README.md` or the data documentation.
- Use the word "retention" (or "engagement"), not "churn", to keep
  framing consistent with the project README.
- No bullet points longer than 2 lines.
- Update `ai_usage_log.md` with a brief entry when done.
```

---

## Prompt — Executive Memo (1 page)

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
All 9 notebooks are complete. The artefacts in `outputs/tables/` and
`outputs/figures/` are the source of truth for every number you cite.
You are now writing the 1-page executive memo that anchors the final
PDF submission.

Files to read for context:
- `README.md` — project framing.
- `outputs/tables/baseline_vs_improved.csv` — headline model metrics.
- `outputs/tables/decile_lift.csv` — top-decile lift figures.
- `outputs/tables/sensitivity_cost_value.csv`,
  `sensitivity_budget.csv`, `sensitivity_recession.csv` — risk inputs.
- The output of `decision.build_decision_rule(...).describe()` from
  the bottom of `notebooks/07_decision_framework.ipynb` — contact
  volume and EV in euros.

Your job: create `executive_memo.md` at the project root. The memo is
written for NovaBank's Chief Operating Officer + Chief Customer
Officer — they're smart, time-poor, non-technical.

Required structure (in this order, with these exact heading names):
1. **Title + one-line elevator pitch** — single sentence at the top,
   bold, that captures the recommendation.
2. **Decision** — 2–3 sentences. Lead with the recommendation, not
   the reasoning. ("We recommend deploying a calibrated LightGBM
   targeting model to select the top 20% of customers per campaign
   cycle, projecting €X expected value with €Y avoided waste.")
3. **Rationale** — 3–4 sentences. Cite specific numbers:
   - PR-AUC of LightGBM vs. logistic baseline.
   - Decile-1 lift vs. base rate.
   - Why the budget-constrained rule beats unconstrained EV-max in
     practice.
4. **Impact** — 2–3 sentences in € terms. Use the
   `DecisionRule.describe()` numbers from notebook 07: contact
   volume, expected value, expected successful conversions, FP rate.
5. **Risks** — exactly 3 bullets, drawn from notebook 08:
   - Sensitivity to cost/value assumptions.
   - Sensitivity to base-rate (recession scenario).
   - Fairness disparities flagged in notebook 09 with proposed
     monitoring.
6. **Pilot plan** — one paragraph (~60 words). 90-day pilot, treatment
   vs. control within the top decile, primary KPI = uplift over
   control, decision criteria for full rollout (>20% uplift = scale,
   10–20% = iterate, <10% = kill).

Testable outcomes:
- `executive_memo.md` exists at the project root.
- Word count is between 400 and 550 (1 printed page).
- All six sections present with their exact heading names.
- At least three specific numbers cited, each traceable to a file in
  `outputs/tables/`. Cite the source file inline in a comment in the
  markdown source, e.g. `<!-- source: outputs/tables/baseline_vs_improved.csv -->`.
- No paragraph longer than 4 sentences.
- The word "churn" does not appear; the framing is retention.
- Markdown renders cleanly to PDF via pandoc (no broken tables,
  no missing references).

Constraints:
- No new analysis. Numbers come from the notebooks; if a number
  isn't in `outputs/`, do not cite it.
- Decision-first writing. Recommendation is in the first paragraph,
  not the last.
- € symbol for monetary figures, percentage points (pp) for rate
  changes, percentages (%) for proportions.
- No bullet points outside the Risks section.
- Do not include figures in the memo itself — figures go in the
  slide appendix. The memo is text-only.
- Update `ai_usage_log.md` with an entry covering this work.
```

---

## Prompt — Slide Appendix (8–10 slides)

```
Project context: NovaBank Predictive Retention project (Quantic MSBA).
All 9 notebooks are complete. The executive memo is written. You're
now building the slide appendix that the assignment requires (8–10
slides covering methods, evidence, scenarios, business implications,
and AI usage).

Files to read for context:
- `executive_memo.md` — the memo, for narrative consistency.
- `README.md` — project framing.
- `outputs/figures/` — every chart you can use is here.
- `outputs/tables/` — every number you can cite is here.
- `ai_usage_log.md` — source for the final slide.

Your job: create `slides.pptx` at the project root, using
`python-pptx`. Total slide count must be 8 to 10 inclusive.

Required slide order (each slide should reference at least one
artefact from `outputs/`):

1. **Title** — project name, the reframe in one sentence, team
   names. No artefact required.
2. **Problem & KPIs** — five-sentence brief boiled to 3 bullets,
   plus the two KPIs (PR-AUC + EV at top-K) with justification.
3. **Data & EDA** — top 2–3 EDA insights with figures from
   `outputs/figures/03_eda_*.png`. Use `poutcome` lift and
   `euribor3m`/macro view at minimum.
4. **Modelling approach** — baseline vs. improved + the leakage
   story. Embed `outputs/figures/04_baseline_curves.png`. One bullet
   on the leaky-vs-deployable gap, one on the LightGBM lift.
5. **Model performance** — table from
   `outputs/tables/baseline_vs_improved.csv` (rendered as a slide
   table) + `outputs/figures/05_calibration.png` next to it.
6. **Customer segmentation** — table from
   `outputs/tables/segment_profiles.csv` + 1-line personas
   from notebook 06. (Skip this slide if you need to cut to 8.)
7. **Decision rule** — `outputs/figures/07_decision_framework.png`
   + the headline numbers (threshold, contacts, EV, successful, wasted).
8. **Sensitivity analysis** — `outputs/figures/08_sensitivity_heatmap.png`
   + `08_sensitivity_budget.png`. 3 bullets: what holds, what shifts,
   risk to flag.
9. **Explainability & fairness** —
   `outputs/figures/09_shap_summary.png` (or `09_shap_bar.png`) +
   one fairness table summary. Closing bullet: surrogate-tree
   fallback policy.
10. **AI usage & limitations** — distilled from `ai_usage_log.md`
    (3 bullets: how AI was used, what was verified, key learning) +
    2 limitations of the analysis. Last slide.

Method:
- Use `python-pptx`. Build a small helper that creates each slide
  with: title, optional image (from `outputs/figures/`), optional
  table (from a `outputs/tables/` CSV rendered into a python-pptx
  table), 3–5 bullet talking points, and a footer in 9pt grey text
  that names the source notebook (e.g. "Source: notebooks/07_decision_framework.ipynb").
- Use a single consistent template: same title font, same body
  font, same accent colour throughout. Pick a NovaBank-style
  palette (a deep blue + a single accent — keep it simple).
- 16:9 aspect ratio.
- Title slide: large title, subtitle, team names. No body bullets.
- Last slide MUST be AI usage + limitations.

Testable outcomes:
- `slides.pptx` exists at the project root.
- Slide count between 8 and 10 inclusive.
- Every non-title slide has a footer naming the source notebook.
- Every image embedded comes from a file under `outputs/figures/`.
- Every numeric claim on a slide can be traced to a file under
  `outputs/tables/` (verify by spot-checking 3 random numbers).
- The first slide is the title; the last is AI usage + limitations.
- Slide deck opens cleanly in Microsoft PowerPoint and Google
  Slides without missing-image errors.
- No slide has more than 6 bullet points.
- No bullet point is longer than 2 lines.

Constraints:
- No images sourced outside `outputs/figures/`.
- No numbers invented — every number traces back to a CSV.
- Do not include the word "churn"; framing is retention.
- Do not include speaker notes longer than 60 words per slide.
- Do not exceed 10 slides (rubric explicitly says ≤10).
- Do not drop below 8 slides (rubric requires ≥8).
- Update `ai_usage_log.md` with an entry covering this work.
```

---

## Prompt — Final PDF Assembly

```
Project context: All deliverables are written: `executive_memo.md`,
`slides.pptx`, and the 9 notebooks. Quantic requires submission as
a single PDF containing the memo, with links to the slide deck and
the reproducible notebook (GitHub or Drive).

Your job: produce `NovaBank_Submission.pdf` at the project root.

Required behaviour:
1. Convert `executive_memo.md` to PDF via pandoc:
   `pandoc executive_memo.md -o executive_memo.pdf
    --pdf-engine=xelatex -V geometry:margin=1in
    -V mainfont="Helvetica" -V fontsize=11pt`
   (or equivalent if pandoc/xelatex is not available — fall back
   to a python-based markdown→PDF such as `markdown-pdf`).
2. Verify the output is exactly 1 printed page. If it overruns,
   tighten paragraphs in `executive_memo.md` (do not change
   numbers) and re-render.
3. Append a final page to the memo PDF containing:
   - **Slide deck:** [URL placeholder — to be filled with the Drive
     or GitHub link to slides.pptx]
   - **Reproducible notebook:** [URL placeholder — repo or Drive
     folder containing the full novabank_retention/ project]
   Save as `NovaBank_Submission.pdf`.

Testable outcomes:
- `NovaBank_Submission.pdf` exists at the project root.
- Page 1 is the executive memo (1 page).
- Page 2 contains the two link placeholders, clearly labelled.
- Total page count is exactly 2.
- File opens in Adobe Acrobat and Preview without errors.

Constraints:
- Do not embed the full slide deck or notebook in the PDF —
  Quantic asks for links only.
- Do not include any analysis or visualisations in the PDF beyond
  what's in the memo.
- Replace the placeholder URLs with real ones BEFORE submission;
  leaving placeholders in is a 0 on the rubric.
```