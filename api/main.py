"""
api/main.py — FastAPI server for real-time shopper purchase prediction.

Endpoints:
  GET  /              → health check
  GET  /model-info    → metadata about the loaded model
  POST /predict       → single session prediction + intervention flag
  POST /predict-batch → batch prediction from JSON list

Start with:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.features import session_dict_to_dataframe, ALL_FEATURES

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
META_PATH = MODELS_DIR / "best_model_meta.json"

pipeline = None
model_meta = {}

def load_model():
    global pipeline, model_meta

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Resolve alias to get version metadata
    client = mlflow.MlflowClient()
    target = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, MODEL_ALIAS)

    # Load the actual sklearn pipeline
    model_uri = f"models:/{MODEL_REGISTRY_NAME}@{MODEL_ALIAS}"
    pipeline = mlflow.sklearn.load_model(model_uri)

    model_meta = {
        "model_name": target.tags.get("model_name", "unknown"),
        "run_id": target.run_id,
        "version": target.version,
        "stage": MODEL_ALIAS,
        "roc_auc": float(target.tags.get("roc_auc", 0)),
        "intervention_threshold": INTERVENTION_THRESHOLD,
    }

    print(f"✅ Loaded model alias='{MODEL_ALIAS}' from {model_uri}")
    
def load_model_best():
    global pipeline, model_meta
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python scripts/train.py' first."
        )
    pipeline = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        model_meta = json.loads(META_PATH.read_text())
    print(f"✅ Model loaded: {model_meta.get('model_name', 'unknown')}  "
          f"(ROC-AUC = {model_meta.get('roc_auc', '?'):.4f})")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Shopper Purchase Prediction API",
    description=(
        "Predicts the likelihood that a web session results in a purchase. "
        "Sessions below the intervention threshold are flagged for a promotional offer."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_model()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SessionFeatures(BaseModel):
    """All raw features for one browser session."""
    Administrative: int = Field(0, ge=0, description="# administrative pages visited")
    Administrative_Duration: float = Field(0.0, ge=0)
    Informational: int = Field(0, ge=0)
    Informational_Duration: float = Field(0.0, ge=0)
    ProductRelated: int = Field(0, ge=0)
    ProductRelated_Duration: float = Field(0.0, ge=0)
    BounceRates: float = Field(0.0, ge=0, le=1)
    ExitRates: float = Field(0.0, ge=0, le=1)
    PageValues: float = Field(0.0, ge=0)
    SpecialDay: float = Field(0.0, ge=0, le=1, description="Proximity to special day (0–1)")
    Month: str = Field("Nov", description="Abbreviated month, e.g. 'Nov'")
    OperatingSystems: int = Field(2, ge=1)
    Browser: int = Field(2, ge=1)
    Region: int = Field(1, ge=1)
    TrafficType: int = Field(2, ge=1)
    VisitorType: str = Field("Returning_Visitor", description="Returning_Visitor | New_Visitor | Other")
    Weekend: bool = Field(False)

    class Config:
        json_schema_extra = {
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 40.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 12,
                "ProductRelated_Duration": 450.0,
                "BounceRates": 0.02,
                "ExitRates": 0.04,
                "PageValues": 15.0,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }


class PredictionResult(BaseModel):
    purchase_probability: float = Field(..., description="P(session ends in purchase)")
    no_purchase_probability: float = Field(..., description="P(session does NOT purchase)")
    prediction: int = Field(..., description="1 = likely purchase, 0 = likely no purchase")
    intervene: bool = Field(..., description="True if a promotional offer should be triggered")
    intervention_threshold: float
    model_name: str
    confidence: str = Field(..., description="High / Medium / Low confidence bucket")


class BatchRequest(BaseModel):
    sessions: list[SessionFeatures]


class BatchResult(BaseModel):
    results: list[PredictionResult]
    total_sessions: int
    intervention_count: int
    intervention_rate: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MLFLOW_TRACKING_TOKEN = os.getenv("MLFLOW_TRACKING_TOKEN", "")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_REGISTRY_NAME = "shopper_best_model"
INTERVENTION_THRESHOLD = 0.30  # fallback if meta not loaded

def _predict_session(session: SessionFeatures) -> PredictionResult:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    threshold = model_meta.get("intervention_threshold", INTERVENTION_THRESHOLD)
    model_name = model_meta.get("model_name", "unknown")

    # Build DataFrame row
    df = session_dict_to_dataframe(session.model_dump())

    # Predict
    prob_purchase = float(pipeline.predict_proba(df)[0, 1])
    prob_no_purchase = 1.0 - prob_purchase
    prediction = int(pipeline.predict(df)[0])

    # Intervention flag: trigger when purchase probability is low
    intervene = prob_purchase < threshold

    # Confidence bucket
    gap = abs(prob_purchase - 0.5)
    if gap > 0.35:
        confidence = "High"
    elif gap > 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResult(
        purchase_probability=round(prob_purchase, 4),
        no_purchase_probability=round(prob_no_purchase, 4),
        prediction=prediction,
        intervene=intervene,
        intervention_threshold=threshold,
        model_name=model_name,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "model": model_meta.get("model_name", "not loaded"),
        "roc_auc": model_meta.get("roc_auc"),
        "intervention_threshold": model_meta.get("intervention_threshold", INTERVENTION_THRESHOLD),
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if not model_meta:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return model_meta


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(session: SessionFeatures):
    """
    Score a single browser session.

    Returns purchase probability and whether to show a promotional offer.
    """
    return _predict_session(session)


@app.post("/predict-batch", response_model=BatchResult, tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    """
    Score multiple sessions at once.

    Returns per-session results plus aggregate intervention statistics.
    """
    if len(batch.sessions) == 0:
        raise HTTPException(status_code=400, detail="sessions list must not be empty")
    if len(batch.sessions) > 500:
        raise HTTPException(status_code=400, detail="Max 500 sessions per batch")

    results = [_predict_session(s) for s in batch.sessions]
    intervention_count = sum(1 for r in results if r.intervene)

    return BatchResult(
        results=results,
        total_sessions=len(results),
        intervention_count=intervention_count,
        intervention_rate=round(intervention_count / len(results), 4),
    )
