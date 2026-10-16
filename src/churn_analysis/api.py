"""FastAPI scoring service.

Run locally with ``make serve`` (or ``churn serve``); the bundle path comes from the
``CHURN_MODEL_PATH`` environment variable (default ``models/churn_model.joblib``).

Endpoints
---------
* ``GET /health``  - liveness and whether a model is loaded
* ``GET /model``   - model metadata (name, parameters, calibration, metrics)
* ``POST /predict`` - churn probability, risk band, offer recommendation and reason codes
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .serving import score

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer, using the column names and values of the Telco CSV."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "0001-DEMO",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 171.0,
            }
        }
    )

    customerID: str | None = None
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=600)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(gt=0, le=10_000)
    TotalCharges: float = Field(ge=0)


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=5000)


class Reason(BaseModel):
    feature: str
    value: str
    contribution: float


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float
    risk_band: Literal["low", "medium", "high"]
    expected_offer_value: float
    recommend_offer: bool
    top_reasons: list[Reason]


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


def create_app(bundle: dict | None = None, bundle_path: str | None = None) -> FastAPI:
    state: dict = {"bundle": bundle}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["bundle"] is None:
            path = bundle_path or os.environ.get("CHURN_MODEL_PATH", "models/churn_model.joblib")
            if os.path.exists(path):
                state["bundle"] = joblib.load(path)
        yield

    app = FastAPI(title="Customer churn scoring", version=__version__, lifespan=lifespan)

    def _bundle() -> dict:
        if state["bundle"] is None:
            raise HTTPException(status_code=503, detail="model not loaded; run `make export` first")
        return state["bundle"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["bundle"] is not None, "version": __version__}

    @app.get("/model")
    def model_info() -> dict:
        return _bundle()["metadata"]

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        b = _bundle()
        preds = score(b, [c.model_dump() for c in req.customers])
        return PredictResponse(model=b["metadata"]["model"], predictions=preds)

    return app


app = create_app()
