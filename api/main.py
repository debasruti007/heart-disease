"""
=============================================================================
 api/main.py

 FastAPI backend serving the trained heart disease screening model.

 Endpoint: POST /predict
   Input  : raw patient details (the 13 clinical fields, unencoded)
   Output : risk probability, risk band, top contributing features,
            and a "screening aid, not a diagnosis" disclaimer.

 Every prediction (input + output) is appended to api/logs/predictions.log.

 This file is the ONLY thing that talks to the model directly. The web
 app (Streamlit/Gradio) must call this API rather than loading the model
 itself - required by the brief: "the model must be served behind your
 own API endpoint."
=============================================================================
"""

import os
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = "models"
PROCESSED_DIR = "data/processed"
LOG_PATH = "api/logs/predictions.log"

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# -----------------------------------------------------------------------
# Load everything ONCE at startup, not on every request.
# -----------------------------------------------------------------------
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "final_model.keras"))
threshold = joblib.load(os.path.join(MODEL_DIR, "threshold.joblib"))
scaler = joblib.load(os.path.join(PROCESSED_DIR, "scaler.joblib"))
feature_columns = joblib.load(os.path.join(PROCESSED_DIR, "feature_columns.joblib"))
impute_values = joblib.load(os.path.join(PROCESSED_DIR, "impute_values.joblib"))

# Training data, used only to get a "typical patient" value per feature for
# the explanation step below (see explain_prediction).
X_train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
TRAIN_MEANS = X_train_df.mean()

# Build a map: clinical feature name -> the list of encoded columns it
# produced. Built from feature_columns, so it automatically matches
# whatever dummy-column naming data_prep.py actually produced.
FEATURE_GROUPS = {}
for col in NUMERIC_COLS:
    FEATURE_GROUPS[col] = [col]
for col in CATEGORICAL_COLS:
    FEATURE_GROUPS[col] = [c for c in feature_columns if c.startswith(col + "_")]

app = FastAPI(title="Heart Disease Risk Screener API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# -----------------------------------------------------------------------
# Request / response schemas.
# Field(...) constraints reject obviously invalid input automatically
# (e.g. negative age, sex=5) - this is the "handles bad input" requirement.
# ca and thal are Optional: if a clinic doesn't have that test result, the
# same imputation learned in data_prep.py is applied here too.
# -----------------------------------------------------------------------
class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: Optional[float] = Field(None, ge=0, le=3, description="Major vessels colored (0-3)")
    thal: Optional[float] = Field(None, description="Thalassemia code (3, 6 or 7)")


class FeatureContribution(BaseModel):
    feature: str
    direction: str      # "increased risk" or "decreased risk"
    impact: float        # how much the probability changed, 0-1 scale


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_band: str
    flagged_for_review: bool
    top_contributing_features: list[FeatureContribution]
    disclaimer: str = (
        "This tool is a screening aid, not a medical diagnosis. "
        "Please consult a qualified healthcare professional."
    )


# -----------------------------------------------------------------------
# Preprocessing: turn one raw patient into the exact encoded, scaled
# feature vector the model expects - same logic as data_prep.py, applied
# to a single row instead of a whole dataset.
# -----------------------------------------------------------------------
def encode_patient(patient: dict) -> np.ndarray:
    patient = dict(patient)   # don't mutate the caller's dict

    # Missing ca/thal: fill with the value learned from the TRAINING set,
    # same as data_prep.py does for the offline data.
    for col in ["ca", "thal"]:
        if patient.get(col) is None:
            patient[col] = impute_values[col]

    vector = pd.Series(0.0, index=feature_columns)

    for col in NUMERIC_COLS:
        vector[col] = patient[col]

    for col in CATEGORICAL_COLS:
        value = patient[col]
        # Try the value as given, then as int, then as float - this matches
        # whichever way pandas happened to name the dummy column during
        # data_prep.py, without needing to hardcode that format here.
        for candidate in (f"{col}_{value}", f"{col}_{int(value)}", f"{col}_{float(value)}"):
            if candidate in feature_columns:
                vector[candidate] = 1.0
                break
        # If no candidate matched, this value IS the reference category
        # that one-hot encoding dropped (drop_first=True) - leaving all
        # its dummy columns at 0 is the correct encoding, not an error.

    # Scale numeric columns with the SAME scaler fitted in data_prep.py
    numeric_values = vector[NUMERIC_COLS].values.reshape(1, -1)
    vector[NUMERIC_COLS] = scaler.transform(numeric_values)[0]

    return vector.values.reshape(1, -1).astype("float32"), patient


# -----------------------------------------------------------------------
# Explanation: which clinical features pushed this patient's score up or
# down, relative to a "typical" training-set patient.
#
# Method: for each clinical feature (age, sex, cp, ...), replace just
# that feature's encoded value(s) with the TRAINING SET AVERAGE for those
# columns, re-run the model, and measure how much the probability moves.
# A feature whose real value differs a lot from "typical" - in a
# direction the model cares about - shows the biggest impact here.
#
# Note for the viva: this is a per-patient, perturbation-based explanation,
# not the same thing as the whole-model permutation importance you would
# get from scikit-learn's permutation_importance() (which shuffles a
# feature across many patients and measures the drop in overall accuracy).
# Both are called "permutation importance" in the loose sense of
# "importance measured by perturbing a feature," but this one explains
# ONE prediction, while sklearn's explains the model as a whole.
# -----------------------------------------------------------------------
def explain_prediction(vector: np.ndarray, baseline_prob: float, top_n: int = 5):
    contributions = []
    for feature_name, cols in FEATURE_GROUPS.items():
        perturbed = vector.copy()
        for col in cols:
            idx = feature_columns.index(col)
            perturbed[0, idx] = TRAIN_MEANS[col]

        perturbed_prob = float(model.predict(perturbed, verbose=0).ravel()[0])
        impact = baseline_prob - perturbed_prob   # positive = this feature was raising risk
        contributions.append((feature_name, impact))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    result = []
    for feature_name, impact in contributions[:top_n]:
        direction = "increased risk" if impact > 0 else "decreased risk"
        result.append(FeatureContribution(feature=feature_name, direction=direction,
                                           impact=round(abs(impact), 4)))
    return result


def risk_band(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    return "High"


def log_prediction(raw_input: dict, response: PredictionResponse):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": raw_input,
        "risk_probability": response.risk_probability,
        "risk_band": response.risk_band,
        "flagged_for_review": response.flagged_for_review,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    try:
        vector, filled_patient = encode_patient(patient.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process input: {e}")

    prob = float(model.predict(vector, verbose=0).ravel()[0])
    top_features = explain_prediction(vector, prob)

    response = PredictionResponse(
        risk_probability=round(prob, 4),
        risk_band=risk_band(prob),
        flagged_for_review=bool(prob >= threshold),
        top_contributing_features=top_features,
    )
    log_prediction(filled_patient, response)
    return response