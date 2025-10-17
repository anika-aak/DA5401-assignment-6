# DA5401 A6 — Imputation via Regression for Missing Data

## Overview
This project implements and compares multiple strategies for handling missing data in a credit-risk classification task using the **UCI Credit Card Default Clients Dataset**.  
The assignment demonstrates how different imputation techniques affect the performance of a downstream **Logistic Regression classifier**.

---

## Objectives
1. Introduce artificial **Missing-At-Random (MAR)** values in selected numerical features.
2. Apply various **imputation methods** — simple, linear, and non-linear.
3. Train and evaluate a **Logistic Regression model** on each resulting dataset.
4. Compare the impact of each missing-data strategy on predictive performance.

---

## Dataset
- **Source:** [UCI Credit Card Default Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Target variable:** `default.payment.next.month`
- Artificial missingness introduced (~5%) in:
  - `AGE`
  - `BILL_AMT1`
  - `PAY_AMT1`

---

## Workflow Summary

### Part A — Data Preprocessing and Imputation
| Dataset | Method | Description |
|----------|---------|-------------|
| **A** | Median Imputation | Baseline strategy using column medians for missing values. |
| **B** | Linear Regression Imputation | Predict missing values in one feature using a linear model trained on other features; fill remaining NaNs with medians. |
| **C** | KNN (Non-Linear) Imputation | Predict missing values using K-Nearest Neighbors regression; fill remaining NaNs with medians. |
| **D** | Listwise Deletion | Drop all rows containing any missing value. |

---

### Part B — Model Training and Evaluation

- Standardized all datasets using **StandardScaler**.
- Trained a **Logistic Regression** classifier on each dataset.
- Evaluated with:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**

#### Summary of Results
| Dataset | Imputation Strategy | Accuracy | F1 (Positive Class) |
|----------|--------------------|-----------|---------------------|
| A | Median Imputation | 0.8080 | 0.3557 |
| B | Linear Regression | 0.8085 | 0.3592 |
| C | KNN Regression | 0.8085 | 0.3599 |
| D | Listwise Deletion | 0.8075 | 0.3555 |

---

### Part C — Comparative Analysis and Discussion

#### Trade-off: Listwise Deletion vs. Imputation
Listwise deletion discards incomplete rows, reducing sample size and class balance, leading to a minor drop in performance (≈0.001).  
Imputation retains all data, preserving variance and improving model stability.

#### Linear vs. Non-Linear Regression Imputation
Linear regression and KNN regression yield nearly identical results.  
This suggests the relationship between predictors and the imputed feature (`BILL_AMT1`) is largely **linear**, making linear regression the more efficient choice.

#### Recommended Strategy
For this dataset (5% MAR missingness, primarily linear correlations):
- **Best method:** Linear Regression Imputation (Model B)
- **Rationale:** Retains all samples, preserves relationships, slightly outperforms median imputation, and is computationally efficient.

---

## Key Takeaways
- **Low missingness (<10%)** rarely justifies complex imputation.
- **Median imputation** offers simplicity and robustness.
- **Linear regression imputation** provides marginal but consistent improvements.
- **Listwise deletion** should be avoided under MAR conditions due to unnecessary data loss.

---

