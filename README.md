<!-- Source notebook used for this README: :contentReference[oaicite:0]{index=0} -->

# 🎬 Movie Hit Prediction
## A Data-Driven Framework for Greenlighting Decisions
authors: Saad Iftikhar, Talal Naveed, Ahmed Arkam Mohamed Faisaar
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
- [Streamlit Implementation](#streamlit-implementation)
- [Business Applications](#business-applications)
- [How to Reproduce](#how-to-reproduce)
- [Repository Structure](#repository-structure)
- [References](#references)
- [Conclusion](#conclusion)

---
## Project Structure:

<img width="1064" height="438" alt="image" src="https://github.com/user-attachments/assets/38f730ad-1c66-477f-8344-b865594239b9" />


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


<img width="1790" height="490" alt="hit-vs-non-hit" src="https://github.com/user-attachments/assets/4f21c418-2100-4b2b-bb55-16250f6818ef" />

Figure 1 highlights the baseline: hits are the minority class (~35%). This matters for evaluation because accuracy alone can be misleading; we care about precision/recall tradeoffs depending on business context.

**2) Hit patterns across budget, studio backing, season, and genre**

<img width="1589" height="1189" alt="hit-key-categories" src="https://github.com/user-attachments/assets/4554a712-671d-4df5-91db-2391e87335a3" />

Figure 2 makes the business story visible: certain categories (big budgets, big studios, summer releases, and some genres) dramatically shift hit probability.

**3) Hits vs non-hits across continuous features**

<img width="1790" height="975" alt="box-plot" src="https://github.com/user-attachments/assets/02475e30-8751-4bae-b399-066905a142e3" />

Figure 3 shows where separation is strongest. Budget and director track record stand out clearly; runtime is much less discriminative.

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

### Classification (Hit Prediction) — Final Model
**Best model:** LightGBM Classifier

- **ROC-AUC (test):** 0.9762

- **Accuracy (test):** 90.21%

- **Precision (hit class):** 84.49%

- **Recall (hit class):** 95.83%

**ROC Curves**

<img width="803" height="487" alt="ROC" src="https://github.com/user-attachments/assets/db1fd2d1-05cc-423b-8ec5-a94f1ca5d6dd" />


**Confusion Matrices**
<img width="1355" height="1189" alt="confusionmatrix" src="https://github.com/user-attachments/assets/8a16281d-d43c-4131-af75-4005e59d5541" />

### Regression (Revenue Forecast) — Final Model
**Best model:** XGBoost Regressor

- **MAE (test):** $65.4M (original scale) / 0.4981 (log scale)

- **RMSE (test):** 0.7471 (log scale)

- **R² (test):** 0.8577

**Predicted VS Actual Revenue**
<img width="1590" height="590" alt="predictedvsactual" src="https://github.com/user-attachments/assets/4cf279c7-a541-4f30-8fb2-66f998e53fb5" />

### Executive Summary
| Component | Output | Final Choice | Key Metric (Test) |
|-----------|--------|--------------|-------------------|
| Hit classifier | P(hit ≥ $100M) | LightGBM | ROC-AUC: 0.9762 |
| Revenue regressor | Expected revenue | XGBoost Regressor | MAE: $65.4M, R²: 0.858 |

---
## Streamlit Implementation

Implemented an interactive Streamlit-based system that predicts whether a movie project should be greenlit based on historical data and machine learning.
<img width="2226" height="1204" alt="image" src="https://github.com/user-attachments/assets/3be11089-b47f-4aff-9aee-aa47d9bb49cd" />

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

---

