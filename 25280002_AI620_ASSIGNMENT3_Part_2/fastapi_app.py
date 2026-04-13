# 25280002_AI620_ASSIGNMENT3_Part_2/fastapi_app.py

"""
Run:
    uvicorn fastapi_app:app --reload --port 8000
UI: http://localhost:8000/docs
"""

import json
import os

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="PakWheels Car Price Category API",
    description="Classifies a used car as **High Price** or **Low Price** using a trained SVM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD MODEL

MODEL_FILE    = "models/pakwheels_svm_model.pkl"
METADATA_FILE = "models/model_metadata.json"

model    = None
metadata = {}

NUMERIC_FEATURES     = ["year", "car_age", "engine", "mileage"]
CATEGORICAL_FEATURES = ["transmission", "fuel", "body", "city"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@app.on_event("startup")
def load_model():
    global model, metadata, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(
            f"Model not found at '{MODEL_FILE}'. "
            "Run task1_train_model.ipynb first to train and save the model."
        )

    model = joblib.load(MODEL_FILE)

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        NUMERIC_FEATURES     = metadata.get("numeric_features",     NUMERIC_FEATURES)
        CATEGORICAL_FEATURES = metadata.get("categorical_features", CATEGORICAL_FEATURES)
        ALL_FEATURES         = metadata.get("all_features",         ALL_FEATURES)

    print(f"[startup] Model loaded — test accuracy: {metadata.get('accuracy', 'N/A')}")
    print(f"[startup] Features: {ALL_FEATURES}")


# SCHEMAS
class CarFeatures(BaseModel):
    """Input features — field names match the PakWheels CSV and the assignment example."""
    year:         int   = Field(..., ge=1990, le=2025,    example=2018)
    engine:       float = Field(..., ge=600,  le=10000,   example=1300,
                                description="Engine capacity in cc")
    mileage:      float = Field(..., ge=0,    le=1_000_000, example=45000,
                                description="Odometer reading in km")
    transmission: str   = Field(..., example="Manual")
    fuel:         str   = Field(..., example="Petrol")
    body:         str   = Field(..., example="Hatchback")
    city:         str   = Field(..., example="Lahore")

    class Config:
        schema_extra = {
            "example": {
                "year": 2018, "engine": 1300, "mileage": 45000,
                "transmission": "Manual", "fuel": "Petrol",
                "body": "Hatchback", "city": "Lahore",
            }
        }


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence:      float
    probabilities:   dict



# ROUTES
@app.get("/", tags=["health"])
def root():
    return {
        "status":   "ok",
        "message":  "PakWheels Price Category API is running.",
        "accuracy": metadata.get("accuracy"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: CarFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # We must map the Pydantic names to the exact names used during training
    row = {
        "year":               features.year,
        "car_age":            2026 - features.year,
        "engine_capacity_cc": features.engine,  
        "mileage":            features.mileage,
        "transmission":       features.transmission,
        "fuel":               features.fuel,
        "body_type":          features.body,     
        "city":               features.city,
        "make":               "Toyota"          
    }

    NUMERIC_FEATURES     = ["year", "car_age", "engine_capacity_cc", "mileage"]
    CATEGORICAL_FEATURES = ["make", "transmission", "fuel", "body_type", "city"]
    ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    input_df = pd.DataFrame([row])[ALL_FEATURES]

    try:
        pred_class = int(model.predict(input_df)[0])
        pred_proba = model.predict_proba(input_df)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(
        predicted_class=pred_class,
        predicted_label="High Price" if pred_class == 1 else "Low Price",
        confidence=round(pred_proba[pred_class], 4),
        probabilities={
            "Low Price":  round(pred_proba[0], 4),
            "High Price": round(pred_proba[1], 4),
        },
    )


@app.get("/metadata", tags=["info"])
def get_metadata():
    """Return model metadata: accuracy, price threshold, feature names."""
    return metadata



if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
