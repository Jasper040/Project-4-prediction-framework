# AI Usage Log

The Quantic rubric requires a log of how AI tools were used during this project.
Add an entry every time you use an AI assistant for something non-trivial. Both
team members log to the same file. Do this **as you go** — reconstructing it at
the end is painful and inaccurate.

Format per entry:

- **Date** — YYYY-MM-DD
- **Author** — your name
- **Tool** — Claude / ChatGPT / Cursor / Copilot / etc.
- **Task** — what you were trying to do
- **Prompt summary** — one sentence on what you asked
- **What the AI contributed** — concretely, what it produced or unblocked
- **What you learned / how you verified** — your own check or insight

---

## Entries

### 2026-05-04 — Jasper — Claude (Cowork mode)
- **Task:** Project framing + build plan.
- **Prompt summary:** Asked Claude to pick apart the project description, explain
  the gap between the stated NovaBank churn case and the UCI Bank Marketing
  dataset, and propose a framing that lets us use the real labels.
- **What the AI contributed:** Identified that the dataset is campaign-response
  data (not literal churn), proposed the "Predictive Retention via Targeted
  Engagement Campaigns" reframe, mapped each rubric line to a concrete
  deliverable, and flagged the `duration` leakage issue as a deliberate
  methodological move (build one leaky benchmark + one deployable model).
- **What I learned / how I verified:** Confirmed the leakage point against the
  dataset README (`bank-additional-names.txt`), which states `duration` is only
  known after the call ends and should not be used for realistic prediction.

### 2026-05-04 — Jasper — Claude (Cowork mode)
- **Task:** Scaffold the project.
- **Prompt summary:** Asked Claude to scaffold the full Python project structure
  for VS Code with notebook stubs, src/ modules, requirements.txt, and a README.
- **What the AI contributed:** Generated the folder layout, starter modules in
  `src/`, notebook stubs with rubric-mapped section headers, and this log
  template.
- **What I learned / how I verified:** Reviewed the structure against the
  Quantic step-by-step guidance in the project description; confirmed each step
  has a corresponding notebook.

### 2026-05-05 - Jasper Claude Code
- **Task:** fill in `notebooks/02_data_readiness.ipynb` so it runs
- **Prompt summary** Asked Claude to complete a reproducible data-readiness notebook for the UCI Bank Marketing dataset, using the existing `src/data.py` helper pipeline, and keeping strict constraints around stratified splits, validation checks, and end-to-end notebook execution in mind.
- **What the AI contributed:** Filled in notebooks/02_data_readiness.ipynb with: detailed profiling (shape, memory, dtypes, null counts, target counts+proportions),
  assertion-guarded cleaning verification, stratification checks, save+existence checks, and a final summary cell. The notebook was then executed via jupyter
  nbconvert --execute --inplace so all outputs are present.
- **What I learned / how I verified:** Confirmed the test set has a positive rate between 11.1% and 11.5% as well as the testing set. Testing set differed by only 0.0007 pp indicating a strong coverage and accurate data model. Ensured test data was not seen in the training data set.   
  - data/interim/train.parquet — 32,950 rows, positive rate 11.27% (within 0.111–0.115 ✓)                                                                       
  - data/interim/test.parquet — 8,238 rows, positive rate 11.26% (within 0.111–0.115 ✓)                                                                         
  - Train/test rate gap — 0.0007 pp (threshold: 0.1 pp ✓)                                                                                                       
  - Notebook executed end-to-end without errors, all cell outputs populated ✓

### 2026-05-06 — Jasper — Claude Code
- **Task:** Fill in `notebooks/03_eda.ipynb` with slide-ready EDA on the train set.
- **Prompt summary:** Asked Claude to complete the EDA notebook using `src/viz.py` helpers, producing univariate distributions, bivariate lift charts, a macro trends view, and 3–5 headline insights — all from the train set only, with figures saved to `outputs/figures/`.
- **What the AI contributed:** Generated all chart code (histograms, bar plots, decile conversion rates, macro timeline), saved ≥8 PNGs with the `03_eda_` prefix, and drafted a final markdown cell with bulleted insights each referencing a specific saved figure. Never opened the test parquet.
- **What I learned / how I verified:** Counted 12 figures in `outputs/figures/` with the `03_eda_` prefix ✓. Checked that `test.parquet` was not referenced anywhere in notebook cell code ✓. Read the headline insights and confirmed each cited a real figure filename ✓. The poutcome and euribor3m charts aligned with what the data dictionary predicted — prior campaign success and low interest rates are the clearest positive signals.

### 2026-05-06 — Jasper — Claude Code
- **Task:** Fill in `notebooks/04_baseline_model.ipynb` — logistic regression baselines (leaky benchmark + deployable model).
- **Prompt summary:** Asked Claude to complete the baseline notebook: load the interim split, train two logistic regressions (one with `duration`, one without), produce a side-by-side comparison CSV, save ROC/PR/confusion figure, persist the deployable pipeline + probabilities via joblib, and write a plain-language interpretation of the leakage gap.
- **What the AI contributed:**
  - Filled in the final markdown cell (§4.7) with a 2–3 sentence slide-ready interpretation of the leakage gap.
  - Diagnosed and fixed a silent bug: `build_preprocessor` used `remainder="drop"` with hardcoded column lists, so `duration` was silently discarded even when passed in `X_train_leaky` — making both models produce identical AUC (~0.80) instead of the expected ~0.94 / ~0.80 split.
  - Added an `extra_numeric_cols` parameter to `build_preprocessor` (`src/features.py`) and `make_logreg_pipeline` (`src/models.py`) so the leaky pipeline routes `duration` through the same impute + scale step as the other numeric features. Change is backward-compatible; all other notebooks call `make_logreg_pipeline()` with no args and are unaffected.
  - Updated the leaky pipeline cell to call `make_logreg_pipeline(extra_numeric_cols=['duration'])`.
  - Registered the `.venv` as the `novabank` Jupyter kernel and executed the notebook end-to-end via `nbconvert`.
  - Verified `duration` is absent from the saved deployable pipeline by inspecting `preprocessor.transformers_` and `classifier.n_features_in_` (63 features vs. 64 in the leaky model).
- **What I learned / how I verified:** Ran the following checks after execution:

  ```powershell
  python -c "import pandas as pd; df=pd.read_csv('outputs/tables/baseline_comparison.csv', index_col=0); print(df[['roc_auc','pr_auc']].to_string())"
  python -c "import joblib; m=joblib.load('outputs/models/baseline_logreg.joblib'); print(list(m.keys())); print(m['proba_test'].shape)"
  python -c "import joblib; m=joblib.load('outputs/models/baseline_logreg.joblib'); print(round(m['proba_test'].min(),3), round(m['proba_test'].max(),3))"
  Test-Path outputs/figures/04_baseline_curves.png
  Select-String -Path outputs/tables/baseline_comparison.csv -Pattern "duration"
  ```

  Results:

  ```
                                      roc_auc    pr_auc
  Leaky benchmark (with duration)    0.943863  0.622475
  Deployable baseline (no duration)  0.800945  0.460116
  ['model', 'proba_test']
  (8238,)
  0.047 0.988
  True
  outputs\tables\baseline_comparison.csv:2:Leaky benchmark (with duration),...
  outputs\tables\baseline_comparison.csv:3:Deployable baseline (no duration),...
  ```

  Outcome checklist:

  | Check | Expected | Actual | Pass? |
  |---|---|---|---|
  | Leaky ROC-AUC | ~0.93 | 0.9439 | ✓ |
  | Deployable ROC-AUC | ~0.78–0.80 | 0.8009 | ✓ |
  | AUC gap | > 0.10 | 0.143 | ✓ |
  | Leaky PR-AUC | ~0.60–0.65 | 0.6225 | ✓ |
  | Deployable PR-AUC | ~0.40–0.45 | 0.4601 — baseline to beat in NB05 | ✓ |
  | `proba_test` shape | (8238,) | (8238,) | ✓ |
  | `proba_test` range | 0.0–1.0 | 0.047–0.988 | ✓ |
  | Keys in joblib | `['model', 'proba_test']` | `['model', 'proba_test']` | ✓ |
  | `duration` in comparison CSV | No match (silence) | Row labels contain "duration" as expected; no leaky model artefact persisted to `outputs/models/` | ✓ |

### 2026-05-06 — Jasper — Claude Code
- **Task:** Fill in `notebooks/05_improved_model.ipynb` — LightGBM with Optuna tuning and isotonic calibration.
- **Prompt summary:** Asked Claude to complete the improved-model notebook: Optuna hyperparameter search (50 trials, 5-fold stratified CV PR-AUC), refit best pipeline, isotonic calibration on the train set, side-by-side comparison against the logistic baseline, calibration reliability diagram, and persist the calibrated artefact.
- **What the AI contributed:**
  - Completed all code cells (already scaffolded as stubs) including the Optuna objective, calibration refit, comparison table generation, calibration plot, and joblib persistence.
  - Wrote the §5.7 plain-language interpretation cell based on actual run results.
- **What I learned / how I verified:** Ran the validation commands below and caught two issues in the process:
  - `cross_val_score(n_jobs=-1)` caused a silent hang inside the Jupyter kernel on Windows (Proactor event loop incompatibility with ZMQ); fixed by switching to `n_jobs=1`.
  - The validation script expected column `brier_score` but `classification_summary` outputs key `brier`; fixed by adding `.rename(columns={'brier': 'brier_score'})` before saving the CSV.

  ```powershell
  python -c "import pandas as pd; df=pd.read_csv('outputs/tables/baseline_vs_improved.csv', index_col=0); print(df[['roc_auc','pr_auc','brier_score']].to_string())"
  python -c "import joblib; m=joblib.load('outputs/models/improved_lgbm.joblib'); print(list(m.keys())); print(m['best_params'])"
  python -c "import joblib; m=joblib.load('outputs/models/improved_lgbm.joblib'); print(m['proba_test'].shape); print(round(m['proba_test'].min(),3), round(m['proba_test'].max(),3))"
  Test-Path outputs/figures/05_calibration.png
  ```

  Results:

  ```
                          roc_auc    pr_auc  brier_score
  Logistic baseline      0.800945  0.460116     0.161569
  LightGBM (calibrated)  0.816889  0.494776     0.074172
  ['model', 'proba_test', 'best_params']
  {'learning_rate': 0.0108, 'num_leaves': 26, 'min_child_samples': 66, 'reg_alpha': 0.103, 'reg_lambda': 0.021, 'scale_pos_weight': 7.88}
  (8238,)
  0.0 0.945
  True
  ```

  Outcome checklist:

  | Check | Expected | Actual | Pass? |
  |---|---|---|---|
  | LightGBM PR-AUC ≥ baseline | ≥ 0.4601 | 0.4948 | ✓ |
  | LightGBM ROC-AUC ≥ baseline | ≥ 0.8009 | 0.8169 | ✓ |
  | Brier score improved | < 0.1616 | 0.0742 (54% better) | ✓ |
  | `improved_lgbm.joblib` keys | model, proba_test, best_params | ✓ | ✓ |
  | `proba_test` shape | (8238,) | (8238,) | ✓ |
  | `05_calibration.png` exists | True | True | ✓ |

### 2026-05-06 — Jasper — Claude Code
- **Task:** Fill in `notebooks/06_segmentation.ipynb` — k-means customer segmentation layered on top of the LightGBM model.
- **Prompt summary:** Asked Claude to complete the segmentation notebook: fit the same preprocessor as the model, run a k=2–10 search (inertia + silhouette), pick and justify a k, profile segments, and score the LightGBM model within each segment.
- **What the AI contributed:**
  - Built the full notebook from the scaffold: preprocessor fit/transform with `get_feature_names_out()` cached, k-search loop saving `kmeans_k_search.csv`, 1×2 diagnostic figure saved as `06_kmeans_diagnostic.png`, K=4 refit, segment profiling (n, conversion rate, avg age, dominant job/contact, mean euribor3m) saved as `segment_profiles.csv`, and per-segment model scoring (PR-AUC with NaN guard for single-class segments) saved as `segment_model_perf.csv`.
  - Wrote the §6.4 k-choice justification and §6.7 persona narratives with actual numbers from the run.
- **What I learned / how I verified:**

  ```powershell
  python -c "import pandas as pd; df=pd.read_csv('outputs/tables/segment_profiles.csv'); print(df.to_string())"
  python -c "import pandas as pd; df=pd.read_csv('outputs/tables/kmeans_k_search.csv'); print(df.to_string())"
  python -c "import pandas as pd; df=pd.read_csv('outputs/tables/segment_model_perf.csv'); print(df.to_string())"
  python -c "import pandas as pd; df=pd.read_csv('outputs/tables/segment_profiles.csv'); rates=df['conversion_rate'].values; print('spread:', round(rates.max()-rates.min(),3))"
  Test-Path outputs/figures/06_kmeans_diagnostic.png
  ```

  Results:

  ```
  segment      n  conversion_rate  avg_age  dominant_job  dominant_contact  mean_euribor3m
        3   1243         0.629123    41.76        admin.          cellular        0.976316
        1   9608         0.191715    39.36        admin.          cellular        1.200348
        0  11510         0.059427    40.19        admin.          cellular        4.751523
        2  10589         0.038153    40.22   blue-collar         telephone        4.890886

  k=2..10 rows in kmeans_k_search.csv ✓
  segment_model_perf.csv: seg 3 PR-AUC 0.794, seg 1 PR-AUC 0.463, seg 0 PR-AUC 0.070, seg 2 PR-AUC 0.108
  spread: 0.591
  True
  ```

  Outcome checklist:

  | Check | Expected | Actual | Pass? |
  |---|---|---|---|
  | `kmeans_k_search.csv` rows | 9 (k=2..10) | 9 | ✓ |
  | `06_kmeans_diagnostic.png` exists | True | True | ✓ |
  | `segment_profiles.csv` rows | 4 (K=4) | 4 | ✓ |
  | Profiles sorted by conversion rate desc | seg 3 first (63%) | ✓ | ✓ |
  | `segment_model_perf.csv` rows | 4 | 4 | ✓ |
  | No crash on single-class segment | NaN returned | all segments had both classes | ✓ |
  | Notebook runs clean from fresh kernel | No errors | ✓ | ✓ |

  **K-means k=4 choice:** The silhouette score peaks at k=2 (0.285), not k=4 (0.142), so pure cluster-quality metrics favour a coarser split. k=4 was chosen on business grounds: k=2 collapses the "warm returner" cohort (63% conversion) into the broader cellular group (19%), erasing the highest-value targeting signal. The inertia elbow inflects at k=3–4, and k=4 produces four segments each above minimum actionable campaign size (~1 200 customers in train), mapping to four campaign tiers. k=3 would merge the two low-conversion segments (cellular vs. telephone cold), eliminating a channel-routing decision that matters operationally.

  **Column rename — `conv_rate` → `conversion_rate`:** The initial profile aggregation used the alias `conv_rate`. The validation script raised a `KeyError: 'conversion_rate'`. Fixed by renaming the aggregation key in the notebook and re-executing so the CSV was regenerated with the correct column name.

<!-- Add new entries above this line. Most recent on top, or chronological — your call, just be consistent. -->
