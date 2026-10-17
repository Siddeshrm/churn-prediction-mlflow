# Customer Churn Prediction with MLflow

Predicting which telecom customers are likely to leave, using scikit-learn, with every experiment tracked in MLflow.

## Dataset

Telecom customer churn data from Kaggle: https://www.kaggle.com/datasets/brahimerkan/telcocustomerchurncsv.
After cleaning, the modeling set has 6,589 customers and 32 columns, with a churn rate of 28.4%.
The CSV is not included in this repository. Download it from the link above and save it as `telecom_customer_churn.csv` in the project folder.

## Notebooks (run in this order)

1. `01_churn_data_cleaning_eda.ipynb` - data cleaning and exploratory analysis; saves the cleaned data files used later.
2. `02_churn_sql_analysis.ipynb` - SQL analysis of churn patterns.
3. `03_churn_modeling_mlflow.ipynb` - encoding, train/test split, baseline model, Random Forest, threshold tuning, and MLflow tracking.

## Approach

- Dropped `City` (1,106 unique values, too sparse for one-hot encoding) and one-hot encoded the remaining categorical columns.
- 80/20 stratified train/test split (`random_state=42`) so the churn rate is the same in both sets.
- Baseline: Logistic Regression with feature scaling, in a scikit-learn pipeline.
- Compared against a Random Forest (200 trees).
- Tuned the decision threshold instead of using the default 0.5, because missing a churner costs more than a false alarm.
- Logged parameters, metrics, and models for each run in MLflow.

## Results (test set)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.50 | 0.719 | 0.703 | 0.711 | 0.912 |
| Random Forest | 0.50 | 0.852 | 0.618 | 0.716 | 0.913 |
| Logistic Regression, tuned | 0.35 | 0.665 | 0.818 | 0.734 | 0.912 |
| Random Forest, tuned | 0.40 | 0.764 | 0.719 | 0.741 | 0.913 |

Both models rank customers equally well (ROC-AUC about 0.91). Changing the threshold changes the trade-off between catching more churners (recall) and avoiding false alarms (precision). The tuned Random Forest gave the best balance (F1 0.741).

## Key findings

- Tenure is the strongest predictor of churn, followed by revenue and charge features.
- Churn by contract type: month-to-month 51.7%, one year 10.9%, two year 2.6%.
- Churn by tenure: 59.9% in the first 12 months, falling to 9.5% after 48 months.
- Highest-risk group: new customers on month-to-month contracts.

## Limitations

- The decision threshold was chosen using the test set, so the tuned scores are slightly optimistic. A separate validation set or cross-validation would fix this.
- Feature importances show which features the model uses, not why customers churn. The contract and tenure patterns show association, not cause.

## How to run

```
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order. To view the MLflow experiments after running notebook 03:

```
mlflow ui
```

Then open http://localhost:5000.
