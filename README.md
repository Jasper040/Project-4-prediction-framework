# NovaBank — Predictive Retention via Targeted Engagement Campaigns

Quantic MSBA — Analytics Methods and Frameworks Project (Project 4).
Two-person team. Aiming for a 5/5 on the rubric.

---

## The reframe

The provided NovaBank case asks for a **churn prediction framework**. The dataset we
have is the UCI Bank Marketing dataset (Portuguese bank, 2008–2010), where the target
is whether a customer **subscribed to a term deposit** in response to an outbound
campaign call.

Rather than force a churn-flag onto data that doesn't have one, we reframe the
business problem as **Predictive Retention via Targeted Engagement Campaigns**:

> NovaBank's margins are under pressure. Outbound campaigns offering deposit
> products are used to deepen customer relationships and reduce attrition risk.
> Each call costs money; each successful subscription deepens the relationship and
> protects future margin. Given a fixed contact budget, predict which customers
> are most likely to engage so that we waste fewer calls and capture more
> retention value.

Subscription is treated as a **retention signal** — customers deepening their
product relationship are demonstrably less likely to churn. This framing keeps real
labels (no synthetic targets), maps cleanly to the rubric's FP/FN trade-off, and is
defensible to a non-technical stakeholder.

## Repo layout

```
novabank_retention/
├── README.md                    # this file
├── requirements.txt             # pinned Python deps
├── ai_usage_log.md              # required by the rubric
├── .gitignore
├── data/
│   ├── raw/                     # bank-additional-full.csv (immutable)
│   ├── interim/                 # cleaned, split versions
│   └── processed/               # feature-engineered, model-ready
├── notebooks/                   # numbered, run in order
│   ├── 01_problem_framing.ipynb
│   ├── 02_data_readiness.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_improved_model.ipynb
│   ├── 06_segmentation.ipynb
│   ├── 07_decision_framework.ipynb
│   ├── 08_sensitivity_analysis.ipynb
│   └── 09_explainability_and_fairness.ipynb
├── src/                         # reusable engine, imported by notebooks
│   ├── config.py                # paths, costs, seeds, hyperparams
│   ├── data.py                  # load, clean, split
│   ├── features.py              # encoding, feature engineering
│   ├── models.py                # model factory, train, evaluate
│   ├── metrics.py               # expected value, lift, calibration
│   ├── decision.py              # threshold tuning
│   └── viz.py                   # consistent plot helpers
└── outputs/
    ├── figures/                 # exported plots for memo/slides
    ├── models/                  # pickled fitted models
    └── tables/                  # summary CSVs
```

## Setup

From the `novabank_retention/` folder:

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name novabank --display-name "Python (novabank)"
```

Then open the project in VS Code, select the `Python (novabank)` kernel, and run
the notebooks in numerical order.

## Division of labour (2-person team)

| Track | Owner | Notebooks | `src/` modules |
|-------|-------|-----------|----------------|
| Modelling track | Person A | 02, 04, 05, 06 | `data.py`, `features.py`, `models.py` |
| Decision track  | Person B | 01, 03, 07, 08, 09 | `metrics.py`, `decision.py`, `viz.py` |

Both review each other's notebooks before merging. README, AI usage log, the
1-page memo, and the slide deck are joint at the end.

**Workflow rules:**

1. Notebooks import from `src/`. If you write a helper inline more than once,
   move it into `src/` so the other person can use it.
2. Re-run notebooks top-to-bottom before committing — the rubric requires
   reproducibility.
3. Keep cell outputs in committed notebooks (graders read them without running).
4. Log every meaningful AI prompt in `ai_usage_log.md` as you go (don't try to
   reconstruct it at the end).

## Rubric map

Every notebook earns a specific rubric line. Don't add work that doesn't map back
to one of these.

| Rubric line | Where it lives |
|-------------|----------------|
| KPIs + strategic alignment | 01_problem_framing |
| Data readiness, dictionary | 02_data_readiness |
| Pattern discovery / EDA | 03_eda |
| Baseline model | 04_baseline_model |
| Improved model + comparison | 05_improved_model |
| Depth via combined methods | 06_segmentation |
| Decision rule + threshold + pilot plan | 07_decision_framework |
| Sensitivity / scenario analysis | 08_sensitivity_analysis |
| Explainability + fairness for non-technical stakeholders | 09_explainability_and_fairness |
| Reproducibility | `requirements.txt` + clean `src/` + run-in-order notebooks |
| AI usage log | `ai_usage_log.md` |

## Two engineering choices that are deliberately load-bearing

1. **`duration` is dropped from the deployable model.** It's only known after
   the call ends, so including it gives an inflated, undeployable AUC. We build
   one model with it (as a leaky benchmark) and one without (the real model).
   This contrast is the headline signal of methodological maturity.
2. **Probabilities are calibrated.** The decision framework only works if a
   "0.6 probability" actually means 60% conversion. We calibrate via isotonic
   regression on a held-out fold and verify with a reliability diagram.

## Final deliverables (per the assignment)

A single PDF containing:
1. A 1-page executive memo (decision, rationale, impact, risks).
2. Link to an 8–10 slide appendix (methods, evidence, scenarios, business
   implications, AI usage).
3. Link to the reproducible notebook (this repo on GitHub or Google Drive).
