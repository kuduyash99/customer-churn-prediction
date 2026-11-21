"""Data cleaning, feature engineering and preprocessing (shared by training and the API)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", *ADDON_SERVICES, "Contract", "PaperlessBilling", "PaymentMethod"]
RAW_COLUMNS = NUMERIC + CATEGORICAL

ENGINEERED_NUMERIC = ["avg_monthly_spend", "num_addon_services", "charge_delta"]
ENGINEERED_CATEGORICAL = ["tenure_group"]


def load_and_clean(path):
    """Read the CSV, fix data types and encode the target (Churn: Yes -> 1, No -> 0)."""
    df = pd.read_csv(path).drop(columns=["customerID"])
    # TotalCharges holds blank strings for brand-new customers -> convert to NaN (imputed later)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop_duplicates().reset_index(drop=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds tenure_group, avg_monthly_spend, num_addon_services and charge_delta."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["tenure_group"] = pd.cut(X["tenure"], bins=[-1, 12, 24, 48, 60, np.inf],
                                   labels=["0-12", "13-24", "25-48", "49-60", "60+"]).astype(str)
        X["avg_monthly_spend"] = (X["TotalCharges"] / X["tenure"].replace(0, np.nan)
                                  ).fillna(X["MonthlyCharges"])
        X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        X["charge_delta"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        return X


def build_preprocessor():
    """Feature engineering -> impute + scale numeric columns, one-hot encode categorical columns."""
    encoder = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC + ENGINEERED_NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
         CATEGORICAL + ENGINEERED_CATEGORICAL),
    ])
    return Pipeline([("engineer", FeatureEngineer()), ("encode", encoder)])
