# Customer Churn Prediction (Telco)

Predicts which telecom customers are likely to cancel their service and explains the main drivers, so a retention team can act early.

## Results (hold-out test set, 20% split, seed 42)

| Metric | Value |
|---|---|
| ROC-AUC | **0.84** (5-fold CV: 0.846 +/- 0.012) |
| Recall (churners caught) | 0.79 |
| Precision | 0.50 |
| F1 | 0.62 |

Logistic Regression, Random Forest and Gradient Boosting all tie at about 0.846 CV ROC-AUC, so the **simplest model (Logistic Regression)** was selected for being easy to explain and maintain.

## Key findings
- Overall churn is **26.5%**; customers in their first year churn at **47%**, versus **9.5%** after four years.
- **Month-to-month** contracts churn at 43% vs 3% for two-year contracts.
- Customers **without tech support** churn at 42% vs 15% with it. **Fiber optic** customers churn at 42% vs 19% for DSL. **Electronic check** payers churn at 45% vs 17% for everyone else.
- `gender` and `PhoneService` have no statistically significant relationship with churn.

## Project structure
```
data/raw/telco_churn.csv      IBM sample Telco dataset (7,043 customers)
notebooks/churn_analysis.ipynb  full analysis: cleaning -> EDA -> stats -> models -> SHAP -> recommendations
src/config.py                 paths, seed, constants
src/data_prep.py              cleaning + data-quality validation
src/features.py               feature engineering + preprocessing
src/train.py                  model comparison, evaluation, saves best model
src/predict.py                score new customers
tests/                        pytest checks for data quality and the saved model
models/churn_model.joblib     trained pipeline
reports/                      metrics.json, figures/, report.md
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train                                    # trains and saves model + metrics + figures
python -m pytest -q                                    # data-quality and model tests
python -m src.predict --demo                           # score 5 sample customers
python -m src.predict --input data/raw/telco_churn.csv --output reports/predictions.csv
jupyter notebook notebooks/churn_analysis.ipynb
```

## Approach
1. **Cleaning & validation:** 11 blank `TotalCharges` values were all new customers (tenure 0), so they were set to 0. Seven automated data-quality checks run before training.
2. **EDA & statistics:** churn by contract, tenure and charges; correlation heatmap; IQR outlier check; chi-square (with Cramer's V) and Mann-Whitney U tests.
3. **Features:** tenure group, number of services, average monthly spend, security/support flag. All built inside a scikit-learn `Pipeline` to avoid data leakage.
4. **Modelling:** 3 models, 5-fold stratified CV, `class_weight="balanced"` for the 27/73 class imbalance.
5. **Evaluation:** ROC-AUC, precision, recall, F1, confusion matrix, and a cost-based decision threshold.
6. **Interpretation:** permutation importance and SHAP.

## Limitations
- The data is a single snapshot with no time dimension, so findings are associations, not proof of cause.
- Business costs used for threshold selection (500 per lost customer, 50 per retention offer) are assumptions.
- `MonthlyCharges` is correlated with fiber optic, service count and tenure, so its model coefficient should not be read in isolation (explained in the notebook).
- Should be re-validated on a company's own data before real use.

## Data privacy
The dataset is a public sample with synthetic customer IDs and no names, addresses or contact details. `customerID` is dropped before modelling and is never used as a feature.

## Data source
IBM Telco Customer Churn sample dataset (https://github.com/IBM/telco-customer-churn-on-icp4d).
