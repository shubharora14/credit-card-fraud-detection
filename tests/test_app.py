
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

# Ensure project root is available during test collection.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app import app, FEATURES, feature_data

client = TestClient(app)


def create_valid_input():

    # Use a real row from the training dataset.
    row = feature_data.iloc[0]

    features = {}

    for column in FEATURES:

        value = row[column]

        if pd.isna(value):
            raise ValueError(
                f"First training row has missing value in {column}. "
                "Choose a complete row for the test."
            )

        if pd.api.types.is_numeric_dtype(feature_data[column]):
            features[column] = float(value)
        else:
            features[column] = str(value)

    return features


def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_homepage_loads():

    response = client.get("/")

    assert response.status_code == 200

    assert "Credit Card Fraud Detection" in response.text


def test_prediction_response():

    sample_data = create_valid_input()

    response = client.post(
        "/predict",
        json={"features": sample_data}
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "result" in result
    assert result["prediction"] in [0, 1]


def test_invalid_prediction_returns_422():

    response = client.post(
        "/predict",
        json={"features": {}}
    )

    assert response.status_code == 422


def test_missing_feature_returns_422():

    sample_data = create_valid_input()

    first_feature = FEATURES[0]

    del sample_data[first_feature]

    response = client.post(
        "/predict",
        json={"features": sample_data}
    )

    assert response.status_code == 422
