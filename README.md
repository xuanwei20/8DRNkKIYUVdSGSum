# Customer Happiness Prediction
Machine learning analysis of ACME Happiness Survey data to predict customer satisfaction.

## Objective
Predict whether a customer is happy (`1`) or unhappy (`0`) using their answers to 6 survey questions.

## Target
- `Y`: 0 = unhappy, 1 = happy

## Features
Each feature is a rating from 1 (least positive) to 5 (most positive):

| Feature | Description |
|---------|-------------|
| `X1` | My order was delivered on time |
| `X2` | Contents of my order were as expected |
| `X3` | I ordered everything I wanted to order |
| `X4` | I paid a good price for my order |
| `X5` | I am satisfied with my courier |
| `X6` | The app makes ordering easy for me |

## What This Code Does

The main script (`customer_happiness.py`) performs the following key steps:

### 1. Load & Split Data
- Loads the CSV dataset
- Splits the data into training and test sets (80/20 stratified split)
- Prints class distributions

### 2. Exploratory Data Analysis (EDA)
- Plot count distribution of ratings for each feature
- Plot correlation heatmap
- Compare feature values against target happiness

### 3. Feature Ranking
Feature importance is calculated using a **Random Forest classifier** to rank survey questions by their predictive strength.

### 4. Model Training & Evaluation
Several models are tested using progressively larger subsets of top features:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Each model is evaluated by:
- Test accuracy
- ROC-AUC
- 5-fold cross-validation

### 5. Save Best Model & Results
- Selects the best model based on highest test accuracy
- Saves:
  - Model object (`best_model.pkl`)
  - Feature importance values (`feature_importance.csv`)
  - Full results (`model_results.csv`)

## Insights & Conclusion

### Random Forest Feature Importance (All Features)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `X3` – I ordered everything I wanted | 0.206 |
| 2 | `X1` – My order was delivered on time | 0.176 |
| 3 | `X2` – Contents of my order were as expected | 0.175 |
| 4 | `X5` – I am satisfied with my courier | 0.166 |
| 5 | `X4` – I paid a good price for my order | 0.164 |
| 6 | `X6` – The app makes ordering easy for me | 0.113 |

### Random Forest Feature Importance (Top 3 Features After Feature Selection)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `X3` – I ordered everything I wanted | 0.496 |
| 2 | `X1` – My order was delivered on time | 0.322 |
| 3 | `X2` – Contents of my order were as expected | 0.182 |

- **Gradient Boosting** achieves or exceeds the **target accuracy of 73%** using only the top three features (`X3`, `X1`, `X2`)