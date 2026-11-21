"""REST API for churn prediction.  Usage: uvicorn app:app --reload   (docs at /docs)"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import RAW_COLUMNS

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts.update(joblib.load("models/churn_model.joblib"))
    yield


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan,
              description="Send customer data to /predict to get a churn probability and a 0/1 prediction.")

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Optional[float] = Field(None, ge=0, description="Leave empty for brand-new customers")

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 2,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": 190.0}}}


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int = Field(..., description="1 = likely to churn, 0 = likely to stay")


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    """Return the churn probability and binary prediction for one customer."""
    row = pd.DataFrame([customer.model_dump()])[RAW_COLUMNS]
    probability = float(artifacts["model"].predict_proba(row)[0, 1])
    return Prediction(churn_probability=round(probability, 4),
                      churn_prediction=int(probability >= artifacts["threshold"]))
