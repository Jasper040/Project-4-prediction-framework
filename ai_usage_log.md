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

<!-- Add new entries above this line. Most recent on top, or chronological — your call, just be consistent. -->
