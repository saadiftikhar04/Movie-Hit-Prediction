<!-- Source notebook used for this README: :contentReference[oaicite:0]{index=0} -->

# 🎬 Movie Hit Prediction
## A Data-Driven Framework for Greenlighting Decisions

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![Notebook](https://img.shields.io/badge/Format-Jupyter%20Notebook-orange)](#)
[![Status](https://img.shields.io/badge/Project-Completed-success)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Context and Motivation](#business-context-and-motivation)
- [Dataset and Scope](#dataset-and-scope)
- [Problem Definition](#problem-definition)
- [Methodology](#methodology)
  - [Data Cleaning](#data-cleaning)
  - [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Findings](#key-findings)
- [Modeling Approach](#modeling-approach)
- [Evaluation and Results](#evaluation-and-results)
- [Business Applications](#business-applications)
- [How to Reproduce](#how-to-reproduce)
- [Repository Structure](#repository-structure)
- [References](#references)
- [Conclusion](#conclusion)

---

## Project Overview

Greenlighting a film is one of the highest-stakes decisions a studio makes. Once production begins, budgets are largely sunk, timelines are locked, and downside risk becomes hard to hedge. Despite this, many greenlight decisions still rely heavily on intuition, precedent, and informal heuristics.

In this project, we build an **end-to-end machine learning system** that predicts whether a movie will become a **box-office hit before release**, using only information that would realistically be available at the greenlighting stage. We define a *hit* as a film that earns **$100M or more in worldwide box-office revenue**, a common benchmark for blockbuster performance.

Our goal is not to replace creative judgment. We treat this system as **decision support**: it helps executives assess risk, compare competing projects, and understand which factors consistently move the odds of commercial success.

---

## Business Context and Motivation

We frame this work as if we are data scientists at **NeoStudio Pictures**, a mid-size production company operating in a competitive global film market. Each year, NeoStudio must choose which projects to fund from a pipeline of scripts, talent packages, and co-production opportunities.

The recurring questions are familiar:
- Is the proposed budget justified by the project's risk profile?
- Does attaching a particular director or cast meaningfully change expected outcomes?
- How much does release timing matter relative to content or budget?
- Which genres or production setups systematically underperform?

This project turns those questions into a **repeatable, data-driven framework**. It does not promise certainty—films will always involve creative risk—but it does help us avoid *uninformed* risk.

---

## Dataset and Scope

**Primary dataset:** TMDB 5000 Movie Dataset (Kaggle)  
**Original size:** ~5,000 movies  
**Final modeling dataset:** **3,229 movies** (after cleaning)

**Input files**
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

**Data includes**
- Budget, revenue, runtime
- Genres and production companies
- Release date
- Cast and crew metadata (including director)
- Popularity and voting signals

> **Scope constraint (important):** We restrict ourselves to **pre-release features only**, so the model mirrors real executive decision constraints. We are not using post-release reviews, ratings, or revenue trajectories as predictors.

---

## Problem Definition

We model the greenlight decision in two complementary ways:

### 1) Classification (Hit Prediction)
We define:
```text
Hit = 1 if worldwide revenue ≥ $100M, else 0
```
This produces a binary, executive-friendly signal: How likely is this project to clear the blockbuster threshold?

### 2) Regression (Revenue Forecasting)
We also predict:

```text
Target = log(1 + worldwide revenue)
```
Revenue is log-transformed to reduce skew and stabilize learning. This gives us an estimate of expected magnitude, not just a yes/no label.

**Baseline reality:** in the cleaned dataset, only about ~35% of films are hits. Any useful model must do better than always guessing "non-hit."

---

## Methodology

### Data Cleaning
Real movie data is messy. Our cleaning pipeline:

- Remove movies with missing or zero budgets/revenue

- Remove invalid release dates

- Convert numeric columns (budget, revenue, runtime) to proper numeric types

- Drop remaining nulls in key modeling fields

After cleaning:

- **3,229 movies**

- Date range: 1916–2016

**Figure 1 (final):** cleaned data distribution + context

*Caption:* Figure 1 shows the core distributions we rely on throughout the project: budgets and revenues are heavily right-skewed; revenue rises with budget but with wide variance; runtimes cluster around ~90–120 minutes.

### Feature Engineering
Raw metadata rarely becomes predictive without transformation. We engineer features that map to executive levers and industry intuition.

**Budget & Scale**
- `log_budget` (stabilizes budget range)

- `budget_millions`

- `budget_bucket`: Low / Medium / High / Very High

**Release Timing**
- `release_year`, `release_month`, `release_quarter`

- `is_summer` (May–Aug)

- `is_holiday` (Nov–Dec)

- `is_spring` (Mar–Apr)

- `years_since_2000` (trend proxy)

**Production Context**
- `has_big_studio` (major studio backing indicator)

- `num_production_companies`

**Creative Talent**
- `num_cast`

- `cast_mean_popularity` (based on available cast fields)

- Director track-record features:

  - `director_hit_rate`

  - `director_mean_log_revenue`

  - `director_num_movies` (experience proxy)

**Content Signals**
- `primary_genre`

- `num_genres`

- One-hot flags for top genres (e.g., `genre_Action`, `genre_Adventure`, etc.)

Overall, we reduce raw complexity into a model-ready dataset with **45 final features** selected for training.

---

## Exploratory Data Analysis
Before training models, we ask: what actually separates hits from non-hits?

**1) Hit vs Non-Hit distribution**

**Figure 2 (final):** class balance

*Caption:* Figure 2 highlights the baseline: hits are the minority class (~35%). This matters for evaluation because accuracy alone can be misleading; we care about precision/recall tradeoffs depending on business context.

**2) Hit patterns across budget, studio backing, season, and genre**

**Figure 3 (final):** hit rates by key categories

*Caption:* Figure 3 makes the business story visible: certain categories (big budgets, big studios, summer releases, and some genres) dramatically shift hit probability.

**3) Hits vs non-hits across continuous features**

**Figure 4 (final):** boxplot comparison

*Caption:* Figure 4 shows where separation is strongest. Budget and director track record stand out clearly; runtime is much less discriminative.

---

## Key Findings
Below are the findings we would actually highlight to leadership—because they are both data-backed and decision-relevant:

**Finding A — Budget increases probability, not certainty**

Budget and revenue are positively correlated, but the variance is huge. Spending more improves the odds of crossing $100M, but it does not guarantee it. For executives, this means budget should be treated as a lever that moves probability, not a substitute for content-market fit.

**Finding B — Director track record is one of the strongest signals**

The most correlated feature with hit status is director historical hit rate, followed by director mean revenue performance. In other words, attaching proven directing talent measurably shifts risk.

**Finding C — Big-studio backing changes the odds**

Movies with major-studio involvement show a large jump in hit rate. This likely reflects distribution reach, marketing spend, and operational maturity. As a greenlight insight: studio backing behaves like a structural advantage.

**Finding D — Release timing is strategic, not cosmetic**

Summer releases are roughly ~1.9× more likely to become hits compared to off-season releases. Timing is not just scheduling; it is strategy.

**Finding E — Genre matters, especially at the extremes**

Some genres (e.g., Animation and Adventure) show consistently high hit rates in this dataset, while others systematically lag. Genre is not destiny, but it meaningfully changes the prior probability you start with.

**Finding F — Budget buckets expose an extreme asymmetry**

Very high-budget movies have a very high hit rate in the sample, while low-budget movies have a very low hit rate. This does not mean low budgets are "bad"—many may be profitable relative to cost—but they are far riskier when the target is specifically "$100M worldwide."

---

## Modeling Approach

### Temporal Train / Validation / Test Split (realistic setup)
We avoid "training on the future" by using a time-based split:

- **Train:** movies released before 2010

- **Validation:** 2010–2012

- **Test:** 2013 and later

This setup better estimates how the model would behave when predicting upcoming projects.

### Models evaluated
**Classification**

- Logistic Regression (baseline)

- Random Forest

- Gradient Boosting

- XGBoost

- LightGBM

**Regression**

- Ridge Regression

- Tree-based regressors

We evaluate classification using:

- Accuracy (secondary)

- Precision / Recall (primary, business-dependent)

- F1

- ROC-AUC

We evaluate regression using:

- MAE

- RMSE

- R²

---

## Evaluation and Results
**Important:** Paste your final tuned metrics here (from the notebook runs you consider "final").
The README is structured so you can update values without rewriting sections.

### Classification (Hit Prediction) — Final Model
**Best model:** [PASTE MODEL NAME HERE]

- **ROC-AUC (test):** [PASTE VALUE]

- **Accuracy (test):** [PASTE VALUE]

- **Precision (hit class):** [PASTE VALUE]

- **Recall (hit class):** [PASTE VALUE]

**Figure 5 (final):** ROC curve

**Figure 6 (final):** Confusion matrix

### Regression (Revenue Forecast) — Final Model
**Best model:** [PASTE MODEL NAME HERE]

- **MAE (test):** [PASTE VALUE]

- **RMSE (test):** [PASTE VALUE]

- **R² (test):** [PASTE VALUE]

**Figure 7 (final):** Predicted vs actual revenue

### Executive Summary (Final Results Snapshot)
| Component | Output | Final Choice | Key Metric (Test) |
|-----------|--------|--------------|-------------------|
| Hit classifier | P(hit ≥ $100M) | [MODEL] | ROC-AUC: [X.XX] |
| Revenue regressor | Expected revenue | [MODEL] | MAE: [X], R²: [X] |

---

## Business Applications
This system is not an automatic greenlight machine. It is best used as structured input to decision-making.

**1) Risk triage for a slate of projects**

If the studio is choosing among 15 scripts, the classifier provides an apples-to-apples way to rank the slate by hit probability.

**2) Budget stress testing**

Executives can ask: "If we move from $30M to $60M, does the hit probability meaningfully change for this type of project?" Budget becomes a lever that can be evaluated quantitatively.

**3) Talent attachment decisions**

The director features enable a grounded discussion of whether attaching a more proven director materially reduces risk.

**4) Release-window strategy**

The seasonal features show that timing shifts hit probability. This supports conversations around whether to target summer/holiday windows versus counter-programming.

**5) Threshold tuning (depending on strategy)**

Studios can tune the probability threshold based on their appetite for:

- False positives (greenlighting a non-hit)

- False negatives (rejecting a hit)

In practice, the "right" threshold is a strategic choice, not a purely statistical one.

---

## How to Reproduce

**1) Get the data**

Download the TMDB dataset from Kaggle and place the CSVs in `data/raw/`:

- `tmdb_5000_movies.csv`

- `tmdb_5000_credits.csv`

**2) Install dependencies**
```bash
pip install -r requirements.txt
```
If you are running in Colab, the notebook also installs dependencies directly.

**3) Run the notebook**

Open and run:

- `notebooks/movie_hit_prediction_complete.ipynb`

**4) Export final figures for the README**

Save figures to:

```text
figures/
  fig_01_core_distributions.png
  fig_02_class_distribution.png
  fig_03_hit_rates_by_category.png
  fig_04_hits_vs_nonhits_boxplots.png
  fig_05_roc_curve.png
  fig_06_confusion_matrix.png
  fig_07_pred_vs_actual.png
```
Once those files exist, GitHub will render every figure inline automatically.

---

## Repository Structure
```text
├── data/
│   ├── raw/                      # Original TMDB CSV files
│   └── processed/                # Cleaned, model-ready dataset
├── figures/                       # Final exported figures used in README
├── notebooks/
│   └── movie_hit_prediction_complete.ipynb
├── README.md
└── requirements.txt               # (recommended) pinned dependencies
```

---

## Limitations and Caveats
We treat this tool with the realism it deserves:

- **Correlation ≠ causation.** The model is predictive, not causal.

- TMDB data reflects historical industry structure; dynamics change over time.

- Some metadata fields may be incomplete or inconsistently populated.

- Creative success retains irreducible uncertainty.

- **This tool reduces risk; it does not eliminate it.**

---

## References
- **[R1]** Kaggle — TMDB 5000 Movie Dataset (Movie Metadata + Credits)
- **[R2]** The Movie Database (TMDB) — data and schema conventions (genres, credits structure)
- **[R3]** Standard ML references for classification/regression evaluation (ROC-AUC, MAE/RMSE, temporal split best practices)

---

## Conclusion
Film will always be a creative business. But creative decisions do not have to be made blindly.

This project demonstrates that with disciplined cleaning, decision-relevant feature engineering, and realistic temporal evaluation, we can meaningfully improve how greenlighting decisions are made—while fully respecting the role of human judgment.

**Uncertainty is unavoidable. Uninformed risk is not.**
