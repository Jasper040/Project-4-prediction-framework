Here's the execution playbook for each notebook — what to build, the methods to use, the traps to avoid, and what comes out the other end. I've marked the rubric-critical moves so you can see where the score is won.  
Step 1 — Problem framing (Notebook 01\)  
What you're building. Pure writing, near-zero code. Five-sentence brief, business→analytics translation, two KPIs with justification, draft contact policy.  
Methods. None. This is framing.  
Key points not to miss.

Write for an executive, not a TA. Every sentence should pass the "would a CFO care?" test.  
Justify explicitly why subscription is a retention signal — anticipate the "this isn't churn" pushback.  
Pick PR-AUC as primary, not ROC-AUC, and state why: with \~11% positive rate, ROC-AUC inflates and PR-AUC reflects what the campaign team actually cares about (precision among contacts).  
Define the FP/FN cost asymmetry up front (FP \= €5 wasted, FN \= lost €115 net engagement value). This seeds Notebook 07\.  
State the contact-budget constraint as a constraint, not a parameter to optimise around freely.

Result. A self-contained framing document inside the notebook. The 1-page memo and slide 1 lift directly from this — if you write it well here, those deliverables write themselves.  
Step 2 — Data readiness (Notebook 02\)  
What you're building. Load → profile → light clean → stratified 80/20 split → save to data/interim/. Plus a complete data dictionary (already drafted in the stub).  
Methods. pandas profiling, sklearn.model\_selection.train\_test\_split(stratify=y), parquet persistence.  
Key points not to miss.

Test set is locked here and not opened until Notebook 04\. No EDA, no peeking. Violating this is the silent way most students cap themselves at a 3\.  
Stratify the split — non-stratified splits fluctuate the 11% positive rate enough to invalidate metric comparisons.  
pdays \== 999 is a sentinel, not a number. The cleaning step replaces it with NaN and adds a was\_previously\_contacted flag. Verify this in bank-additional-names.txt and document.  
No imputation or scaling at this stage. Those operations learn from data and must run inside the modelling pipeline (the ColumnTransformer in features.py). Fitting them on the full dataset is leakage.  
The unknown value in default/education/job is informative, not random missing — keep it as its own level.

Result. data/interim/train.parquet (\~32,950 rows), test.parquet (\~8,238 rows). Data dictionary committed in the notebook. Both teammates work from the same locked split for the rest of the project.  
Step 3 — EDA (Notebook 03\)  
What you're building. Train-only pattern discovery. Univariate distributions, bivariate-vs-target, macro indicators across time.  
Methods. matplotlib/seaborn, pandas groupbys, simple crosstabs.  
Key points not to miss.

Confirm class balance \~11.3% positive — that's the number that anchors PR-AUC as the right metric.  
Look for two kinds of signal: monotonic (logistic will catch these) and interaction-based (only LightGBM will catch). Note both — it sets up the baseline-vs-improved story.  
poutcome \== "success" is the strongest single predictor — conversion rate jumps from \~11% to \~65%. Show this on a slide.  
Cellular contact converts \~2× more than telephone — a slide-worthy single chart.  
The 2008 crisis is in the data — euribor3m drops from \~5% to \~1% over the campaign window. Show macro indicators as a time series and note that the model is being trained across regimes.  
Seasonality is real (March/September/October peak). Don't strip it out.

Result. 3–5 named EDA insights, each backed by one chart, written up in the notebook. These are slide candidates 2–4 in the appendix.  
Step 4 — Baseline model (Notebook 04\)  
What you're building. Two logistic regressions, side-by-side. Leaky benchmark with duration, deployable baseline without it.  
Methods. sklearn.linear\_model.LogisticRegression(class\_weight='balanced'), the preprocessor from features.py.  
Key points not to miss.

The leaky-vs-deployable gap IS the headline. Leaky benchmark will hit ROC-AUC \~0.93 / PR-AUC \~0.55. Deployable drops to ROC-AUC \~0.79 / PR-AUC \~0.43. That gap is the methodological story — most students never even notice the leakage.  
class\_weight='balanced' is non-negotiable. Without it, logistic regression at default threshold predicts "no" for nearly everyone and your confusion matrix looks vacuous.  
Don't only report metrics at the default 0.5 threshold — at this class imbalance, 0.5 is meaningless. Report at top-decile and EV-max thresholds too (Notebook 07 will use these).  
Save metrics: roc\_auc, pr\_auc, brier, f1, plus the full confusion matrix at chosen threshold.  
Persist outputs/models/baseline\_logreg.joblib — Notebook 05 and 07 load it.

Result. baseline\_comparison.csv with both models. Persisted model artefact. Plain-language interpretation paragraph that becomes the "why we drop duration" slide.  
Step 5 — Improved model (Notebook 05\)  
What you're building. LightGBM tuned with Optuna on cross-validated PR-AUC, then calibrated isotonically. Side-by-side vs. baseline.  
Methods. LGBMClassifier, optuna.create\_study(direction='maximize') with 50 trials, CalibratedClassifierCV(method='isotonic', cv=5).  
Key points not to miss.

Tune on train only, with 5-fold stratified CV. Test set never enters tuning.  
Use scale\_pos\_weight \= n\_neg / n\_pos, not SMOTE. Oversampling distorts the probability distribution; the decision framework needs honest probabilities.  
Calibrate after tuning. LightGBM probabilities are systematically miscalibrated (overconfident at extremes). Isotonic \> Platt scaling for tree models. The reliability diagram should hug the diagonal — if it doesn't, your decision framework is built on lies.  
Expected improvement over the deployable baseline: PR-AUC from \~0.43 → \~0.50–0.55. Don't oversell — the gain is real but modest.  
The interactions LightGBM picks up (e.g., poutcome × month, age × education, euribor3m × contact) are where the lift comes from. SHAP in Notebook 09 will surface them.  
Persist improved\_lgbm.joblib with model \+ test predictions \+ best params.

Result. baseline\_vs\_improved.csv, calibration reliability diagram, persisted calibrated model. Plain-language paragraph: "what the improvement is, where it comes from, and what's still uncertain."  
Step 6 — Segmentation (Notebook 06\)  
What you're building. K-means on the same feature space as the model (target excluded), elbow+silhouette to pick k, profile each segment.  
Methods. sklearn.cluster.KMeans with random\_state=42, silhouette\_score (sample 5,000 for speed), groupby aggregations.  
Key points not to miss.

Cluster on the transformed feature space (use features.build\_preprocessor) so segments are defined identically to how the model sees customers. Clustering on raw mixed-type data is meaningless.  
Pick k where silhouette is high AND the elbow is visible. Likely k=4 or 5\.  
The story isn't "we found segments." It's "model performance varies across segments." Some segments have strong signal (PR-AUC 0.65, lift 6×) — high-confidence targeting. Others have weak signal (PR-AUC 0.30) — flag for human review or skip entirely.  
Build personas: one-line plain-language description per segment ("Affluent retirees, prior campaign success, cellular contact"). This is the slide.  
This is the "depth via combining methods" rubric line — without it, you're capped at a 4\.

Result. segment\_profiles.csv, segment\_model\_perf.csv, 4–5 named personas. One slide that ties model performance to actionable segments.  
Step 7 — Decision framework (Notebook 07\)  
What you're building. EV curve, top-K decision rule, comparison to unconstrained EV-max, pilot plan.  
Methods. metrics.ev\_curve, decision.build\_decision\_rule, metrics.decile\_lift.  
Key points not to miss.

State the cost/value assumptions explicitly (€5/contact, €120/conversion, 20% budget) and call them assumptions. Notebook 08 stress-tests them.  
Recommend the budget-constrained rule (top 20%), not the unconstrained EV-max. Real ops teams have fixed campaign budgets. Show both, recommend the constrained one with reasoning.  
The decile lift chart is the slide that sells it. Decile 1 should hit 4–6× the base conversion rate. That's the headline number for the memo.  
Pilot plan must include a control group. Randomise within the top decile so you can measure incremental lift over current practice. Without a control, you can't claim causation and a sharp grader will spot it.  
Define rollout decision criteria up front: e.g., "scale if pilot lift \>20% over control; iterate if 10–20%; kill if \<10%."

Result. Decision rule one-pager with EV, contact volume, expected conversions, FP/FN counts. 90-day pilot plan with explicit success criteria. This becomes the memo's recommendation paragraph.  
Step 8 — Sensitivity analysis (Notebook 08\)  
What you're building. Three scenarios: cost/value sweep, contact-budget sweep, recession (base-rate halved).  
Methods. decision.sensitivity\_sweep, manual index-based subsampling for the recession scenario.  
Key points not to miss.

This is the rubric line that distinguishes a 4 from a 5\. Most students skip it or do it superficially. Do it thoroughly.  
Two sides to the story: what's robust (the direction of the recommendation — "target the top decile") and what's sensitive (the specific threshold, the specific EV). Be honest about both.  
Cost/value sweep: 4×4 grid → heatmap of EV. The recommendation should hold across the realistic ranges.  
Budget sweep (5%/10%/20%/30%/50%): show that smaller budgets concentrate lift but reduce absolute volume; this is a real ops trade-off.  
Recession scenario: halve positives in the test set, re-evaluate. EV will drop sharply, but the rule structure (target top-K) should still dominate any naive policy.  
End with a 3-bullet "what holds vs. what shifts" summary in plain language.

Result. Three CSV tables, 2–3 charts (cost/value heatmap, budget curve, recession comparison). A robustness paragraph for the memo.  
Step 9 — Explainability \+ fairness (Notebook 09\)  
What you're building. SHAP global \+ 2 local explanations, surrogate decision tree (depth 3\) for human-readable rules, fairness check across age and education bands.  
Methods. shap.TreeExplainer on the underlying LightGBM, sklearn.tree.DecisionTreeRegressor(max\_depth=3) \+ export\_text for the surrogate, groupby for fairness.  
Key points not to miss.

SHAP runs on the underlying tree, not the calibration wrapper — pull out calibrated.calibrated\_classifiers\_\[0\].estimator (the stub does this).  
Pick the 2 local SHAP customers deliberately: one strongly recommended for contact (target=yes), one strongly skipped (target=no). These are the slides that defend the model to compliance.  
The depth-3 surrogate tree gives \~8 leaf rules. Use them as a "fallback policy" — if the LightGBM goes down, the front-line team can still apply the rules. This is a defensibility win.  
Fairness check: report PR-AUC, base rate, mean predicted, and top-20% contact rate across age bands and education levels. The likely finding: contact rate skews older (because older customers genuinely convert more) but PR-AUC is consistent — that's a justifiable disparity, not a bias.  
Don't claim "the model is fair." Claim: "here are the disparities, here's the structural reason for each, here's the mitigation if needed." Compliance officers want honesty, not assertions.

Result. SHAP summary plot, 2 SHAP local plots, surrogate tree rules text, two fairness tables. A 3-bullet compliance summary that becomes the explainability slide.  
Cross-cutting moves that compound into a 5  
These aren't single steps but they have to be true throughout:  
Reproducibility. Every notebook runs top-to-bottom from a clean kernel. RANDOM\_SEED \= 42 is in config.py and propagates to every random op. If a grader can't re-run, the rubric line "reproducible notebook" is gone.  
The AI log gets updated as you go, not at the end. The rubric specifically calls out "AI usage." Two-sentence entries are fine — date, what you asked, what you took, how you verified. Don't try to reconstruct three weeks of prompts on the last day.  
Outputs/figures and outputs/tables are the bridge. Every slide in your appendix should reference a file that a notebook generated. If a chart is in the deck but not in outputs/figures/, it's an unreproducible chart and graders will sniff it.  
Memo and slides are written LAST, after Notebooks 01–09 are all green. They are synthesis, not new analysis. If you find yourself doing new modelling work while writing the memo, you've broken the framework.