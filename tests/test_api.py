import pytest
from fastapi.testclient import TestClient

from churn_analysis.api import create_app
from churn_analysis.serving import build_bundle, risk_band, score

BIZ = {"horizon_months": 12, "incentive_months": 1.0, "contact_cost": 5.0, "acceptance_rate": 0.3}
CUSTOMER = {
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


@pytest.fixture(scope="module", params=["xgboost", "logreg"])
def bundle(request, xy):
    X, y = xy
    params = {"n_estimators": 30, "max_depth": 3} if request.param == "xgboost" else {"C": 1.0}
    return build_bundle(request.param, params, X, y, BIZ, calibration="sigmoid", n_jobs=1)


def test_health_without_model():
    with TestClient(create_app(bundle_path="does/not/exist.joblib")) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/predict", json={"customers": [CUSTOMER]}).status_code == 503


def test_predict_returns_probability_and_reasons(bundle):
    with TestClient(create_app(bundle=bundle)) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/model").json()["model"] == bundle["metadata"]["model"]
        resp = client.post(
            "/predict",
            json={
                "customers": [
                    CUSTOMER,
                    {**CUSTOMER, "Contract": "Two year", "tenure": 60, "customerID": "2"},
                ]
            },
        )
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert len(preds) == 2
        assert 0 <= preds[0]["churn_probability"] <= 1
        assert preds[0]["churn_probability"] > preds[1]["churn_probability"]
        assert preds[0]["top_reasons"] and preds[0]["customerID"] == "0001-DEMO"


def test_invalid_payload_is_rejected(bundle):
    with TestClient(create_app(bundle=bundle)) as client:
        assert (
            client.post(
                "/predict", json={"customers": [{**CUSTOMER, "Contract": "Forever"}]}
            ).status_code
            == 422
        )
        assert client.post("/predict", json={"customers": []}).status_code == 422


def test_score_matches_model_and_risk_bands(bundle):
    out = score(bundle, [CUSTOMER])
    assert out[0]["recommend_offer"] == (out[0]["expected_offer_value"] > 0)
    assert risk_band(0.7) == "high" and risk_band(0.3) == "medium" and risk_band(0.05) == "low"
