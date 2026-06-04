# Credit Default Risk Prediction

**Dataset:** Give Me Some Credit (Kaggle)  
**Task:** Binary classification — predict whether a borrower will experience serious financial distress within two years  
**Metric:** ROC-AUC  
**Result:** 0.9603 AUC (5-fold CV)

---

## Problem

Banks and lenders need to estimate credit risk before approving loans. This project builds a classifier that, given a borrower's financial history, predicts the probability they'll default (go 90+ days past due). The output is a probability score between 0 and 1 — lower is safer.

The real challenge here isn't the modeling — it's the messy data. ~20% of income values are missing, the class is heavily imbalanced (~6.7% positive rate), and several numerical columns have clearly erroneous outliers.

---

## Dataset

| File | Description |
|------|-------------|
| `cs-training.csv` | 150,000 labeled borrower records |
| `cs-test.csv` | 101,503 unlabeled records (competition test set) |
| `Data_Dictionary.xls` | Column descriptions |

**Features:**
- `RevolvingUtilizationOfUnsecuredLines` — credit card usage as % of limit
- `age` — borrower age
- `NumberOfTime30-59/60-89DaysPastDueNotWorse` — count of mild/moderate late payments
- `NumberOfTimes90DaysLate` — count of severe late payments
- `DebtRatio` — monthly debt / monthly income
- `MonthlyIncome` — gross monthly income
- `NumberOfOpenCreditLinesAndLoans` — open credit accounts
- `NumberRealEstateLoansOrLines` — mortgage and home equity accounts
- `NumberOfDependents` — family size (excluding borrower)

---

## Approach

### 1. Cleaning
- Removed 1 record with `age = 0` (clear entry error)
- Capped `RevolvingUtilization` and `DebtRatio` at 1.0 — both are defined as ratios; values above 1 are domain violations
- Imputed `MonthlyIncome` missing values with the column median (robust to skew)
- Filled `NumberOfDependents` NaN with 0 (most likely has none)

### 2. Feature Engineering
Three new features created:
- `TotalLatePayments` — sum of all three past-due buckets (proved to be the single most predictive feature)
- `IncomePerDependent` — monthly income divided by household size + 1
- `TotalCreditLines` — open loans + real estate lines

### 3. Handling Class Imbalance
SMOTE (Synthetic Minority Oversampling) used to bring the minority class to 30% of the majority, giving the model enough signal to learn the default pattern without naively predicting all zeros.

### 4. Model Comparison (5-fold CV)

| Model | AUC |
|-------|-----|
| Logistic Regression | 0.8112 |
| Random Forest | 0.9421 |
| **XGBoost** | **0.9603** |

XGBoost was selected as the final model.

---

## Results

**CV AUC: 0.9603 ± 0.0011**

Top features by importance:
1. `TotalLatePayments` (engineered) — 0.294
2. `NumberOfTimes90DaysLate` — 0.232
3. `NumberOfTime30-59DaysPastDueNotWorse` — 0.191
4. `NumberOfTime60-89DaysPastDueNotWorse` — 0.083
5. `RevolvingUtilizationOfUnsecuredLines` — 0.067

Payment history dominates. This aligns with how credit bureaus actually compute risk scores (FICO scoring weighs payment history at ~35%).

---

## Setup

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

Run the notebook: `credit_risk_prediction.ipynb`  
Generates: `submission.csv` with default probabilities for each test record

---

## What I'd Improve

- **Hyperparameter tuning** with Optuna (learning rate, tree depth, regularization)
- **SHAP values** for feature explainability — important in lending for regulatory/compliance reasons
- **LightGBM** as an alternative to XGBoost; handles missing values natively
- **Probability calibration** (Platt scaling or isotonic regression) if scores are used directly for risk pricing
- **Fairness analysis** — check whether model performs equally well across age groups
