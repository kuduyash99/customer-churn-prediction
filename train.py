"""Train, tune and compare churn models, then save the best one.  Usage: python train.py"""
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from features import RAW_COLUMNS, build_preprocessor, load_and_clean

SEED = 42
DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.joblib"

# Business assumptions: value per customer retained, cost of one retention offer, offer success rate
CUSTOMER_VALUE, OFFER_COST, SAVE_RATE = 600, 50, 0.30

MODELS = {
    "logistic_regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "random_forest": (RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
                      {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None],
                       "clf__min_samples_leaf": [5, 20]}),
    "gradient_boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                           "clf__max_depth": [2, 3]}),
}


def net_profit(y_true, proba, threshold):
    """Profit of contacting every customer whose churn probability >= threshold."""
    contacted = proba >= threshold
    true_churners = np.sum(contacted & (y_true == 1))
    false_alarms = np.sum(contacted & (y_true == 0))
    return true_churners * (SAVE_RATE * CUSTOMER_VALUE - OFFER_COST) - false_alarms * OFFER_COST


def best_threshold(y_true, proba):
    """Threshold that maximises net profit; returns (threshold, profit)."""
    thresholds = np.linspace(0.05, 0.95, 91)
    profits = [net_profit(y_true, proba, t) for t in thresholds]
    best = int(np.argmax(profits))
    return float(thresholds[best]), float(profits[best])


def main():
    df = load_and_clean(DATA_PATH)
    X, y = df[RAW_COLUMNS], df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) Tune every model with cross-validation and score it on cross-validated predictions (train only)
    candidates = {}
    for name, (estimator, grid) in MODELS.items():
        pipeline = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_train, y_train)
        cv_proba = cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold, profit = best_threshold(y_train.values, cv_proba)
        candidates[name] = {"model": search.best_estimator_, "threshold": threshold, "profit": profit}
        print(f"{name:20s} CV AUC-ROC={roc_auc_score(y_train, cv_proba):.3f}  "
              f"CV PR-AUC={average_precision_score(y_train, cv_proba):.3f}  "
              f"threshold={threshold:.2f}  CV net profit=${profit:,.0f}  params={search.best_params_}")

    # 2) Select the best model by business impact (net profit)
    best_name = max(candidates, key=lambda n: candidates[n]["profit"])
    best = candidates[best_name]
    print(f"\nSelected model: {best_name}")

    # 3) Final, one-time evaluation on the untouched test set
    proba = best["model"].predict_proba(X_test)[:, 1]
    pred = (proba >= best["threshold"]).astype(int)
    print(f"Test AUC-ROC={roc_auc_score(y_test, proba):.3f}  "
          f"PR-AUC={average_precision_score(y_test, proba):.3f}  "
          f"Precision={precision_score(y_test, pred):.3f}  Recall={recall_score(y_test, pred):.3f}  "
          f"Net profit=${net_profit(y_test.values, proba, best['threshold']):,.0f}")

    joblib.dump({"model": best["model"], "threshold": best["threshold"]}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
