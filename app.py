
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "train.csv"
STATIC_DIR = BASE_DIR / "static"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    filename=LOG_DIR / "predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# LOAD DATA CONTRACT INFORMATION
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {DATA_PATH}"
    )

training_data = pd.read_csv(DATA_PATH)

if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "Model must expose feature_names_in_. "
        "Retrain it using a pandas DataFrame."
    )

FEATURES = list(model.feature_names_in_)

missing_columns = [
    column for column in FEATURES
    if column not in training_data.columns
]

if missing_columns:
    raise ValueError(
        f"Features missing from training dataset: {missing_columns}"
    )

feature_data = training_data[FEATURES]

# Build feature constraints from training data.
# Numeric columns get min/max validation.
# Categorical columns get allowed-value validation.

FEATURE_RULES = {}

for column in FEATURES:

    series = feature_data[column]

    if pd.api.types.is_numeric_dtype(series):

        FEATURE_RULES[column] = {
            "type": "numeric",
            "min": float(series.min()),
            "max": float(series.max())
        }

    else:

        allowed_values = (
            series.dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        FEATURE_RULES[column] = {
            "type": "categorical",
            "allowed": allowed_values
        }


# --------------------------------------------------
# PYDANTIC REQUEST MODEL
# --------------------------------------------------

class PredictionInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraudulent credit card transactions",
    version="1.0.0"
)


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# --------------------------------------------------
# HOMEPAGE
# --------------------------------------------------

@app.get("/")
def homepage():

    html_file = STATIC_DIR / "index.html"

    if not html_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Homepage file not found"
        )

    return FileResponse(html_file)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(FEATURES)
    }

@app.get("/schema")
def get_schema():

    return {
        "features": FEATURE_RULES
    }


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def validate_features(input_features):

    expected_features = set(FEATURES)
    received_features = set(input_features.keys())

    missing = expected_features - received_features
    extra = received_features - expected_features

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {sorted(missing)}"
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected features: {sorted(extra)}"
        )

    for column in FEATURES:

        value = input_features[column]
        rule = FEATURE_RULES[column]

        if value is None:
            raise HTTPException(
                status_code=422,
                detail=f"{column} cannot be null"
            )

        if rule["type"] == "numeric":

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f"{column} must be numeric"
                )

            if not np.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"{column} must be finite"
                )

            if value < rule["min"] or value > rule["max"]:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{column} must be between "
                        f"{rule['min']} and {rule['max']}"
                    )
                )

        else:

            if not isinstance(value, str):
                raise HTTPException(
                    status_code=422,
                    detail=f"{column} must be a string"
                )

            if value not in rule["allowed"]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid value for {column}"
                )

    return True


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):

    input_features = data.features

    validate_features(input_features)

    try:

        input_df = pd.DataFrame(
            [input_features],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            probability = float(np.max(probabilities))

        result = {
            "prediction": int(prediction),
            "result": (
                "Fraudulent Transaction"
                if prediction == 1
                else "Legitimate Transaction"
            ),
            "confidence": (
                round(probability * 100, 2)
                if probability is not None
                else None
            )
        }

        logger.info(
            json.dumps({
                "input": input_features,
                "output": result
            })
        )

        return result

    except HTTPException:
        raise

    except Exception as error:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


