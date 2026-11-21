# Customer Churn Prediction Pipeline

Supervised ML pipeline (scikit-learn) served as a REST API (FastAPI) on the IBM Telco Customer Churn dataset
(7,043 customers, ~26.5% churn).

## Files
| File | Purpose |
|---|---|
| `data/telco_churn.csv` | Raw dataset |
| `features.py` | Cleaning, feature engineering, scaling, one-hot encoding |
| `train.py` | Trains 3 models, tunes with cross-validation, evaluates, selects and saves the best |
| `app.py` | FastAPI service |
| `models/churn_model.joblib` | Saved model and decision threshold |

## How to run
```bash
pip install -r requirements.txt
python train.py                 # train and save the model
uvicorn app:app --reload        # start the API
```
Open http://127.0.0.1:8000/docs to try the API in the browser.

## Approach
- **Cleaning:** blank `TotalCharges` values (new customers) become missing and are median-imputed; `customerID` dropped; duplicates removed.
- **Encoding / scaling:** one-hot encoding for categorical columns, `StandardScaler` for numeric columns, all inside one scikit-learn `Pipeline` so training and the API use identical preprocessing.
- **Engineered features:** `tenure_group`, `avg_monthly_spend` (TotalCharges / tenure), `num_addon_services`, `charge_delta` (MonthlyCharges - avg_monthly_spend).
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, each tuned with 5-fold `GridSearchCV` (scoring: AUC-ROC) on an 80% training split.
- **Evaluation:** AUC-ROC, PR-AUC, precision and recall.
- **Model selection (business impact):** each model's decision threshold is set to maximise net retention-campaign profit on cross-validated predictions from the training data. The model with the highest profit is selected, then evaluated once on the 20% test set.
  Assumptions (editable in `train.py`): customer value $600, offer cost $50, offer success rate 30%.
  A contacted true churner yields +$130 (0.30 x 600 - 50); a contacted non-churner costs $50.

## Results
Cross-validated on the training data:

| Model | AUC-ROC | PR-AUC | Threshold | Net profit |
|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.656 | 0.47 | $99,380 |
| **Random Forest (selected)** | 0.849 | 0.660 | 0.53 | $101,430 |
| Gradient Boosting | 0.849 | 0.661 | 0.24 | $101,150 |

Test set (selected model): AUC-ROC 0.843, PR-AUC 0.660, precision 0.551, recall 0.737.

## API
`POST /predict` takes one customer's data (see the example in `/docs`) and returns:
```json
{"churn_probability": 0.8889, "churn_prediction": 1}
```
`churn_prediction` is 1 when the probability is at or above the profit-optimised threshold.
