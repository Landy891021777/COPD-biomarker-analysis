# COPD Lung Gene-Expression Analysis

AI-driven classification and biomarker discovery for **Chronic Obstructive Pulmonary
Disease (COPD)** from lung tissue gene-expression data — a side project modeling the
AI analysis core of a lung organ-on-chip platform.

## Objective
1. Build a disease-classification model (COPD vs. control) from lung gene expression.
2. Identify key diagnostic **biomarker genes** via model interpretation (SHAP).
3. Validate generalization on an independent cohort.

## Data
| Dataset | Role | Samples | Notes |
|---------|------|---------|-------|
| GSE47460 | Training / discovery | 328 (220 COPD + 108 CTRL) | 17,000+ genes |
| GSE151052 | External validation | 117 | independent cohort |

> Raw data lives in `data/` and is **not** committed (see `.gitignore`).
> Download via `GEOparse` — see scripts in `src/`.

## Tech Stack
Python · pandas · numpy · scikit-learn · XGBoost · imbalanced-learn (SMOTE) ·
SHAP · matplotlib / seaborn

## Structure
```
src/       analysis & modeling scripts
data/      raw datasets (git-ignored)
figures/   generated plots
results/   model outputs, metrics, biomarker tables
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
