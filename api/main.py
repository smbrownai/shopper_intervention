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

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import asyncio
import subprocess

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

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
INTERVENTION_THRESHOLD = 0.30  # fallback if meta not loaded

pipeline = None
model_meta = {}

def load_model():
    global pipeline, model_meta, threshold_config

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if not META_PATH.exists():
        raise FileNotFoundError(f"No metadata found at {META_PATH}")

    full_meta = json.loads(META_PATH.read_text())
    champion = full_meta["champion"]
    run_id = champion["run_id"]

    model_uri = f"runs:/{run_id}/model"
    pipeline = mlflow.sklearn.load_model(model_uri)

    model_meta = {
        "model_name": champion["model_name"],
        "run_id": run_id,
        "roc_auc": champion["roc_auc"],
        "intervention_threshold": champion["intervention_threshold"],
        "challenger": full_meta.get("challenger"),
    }

    # Load threshold config from metadata if present
    if "threshold_config" in full_meta:
        threshold_config.update(full_meta["threshold_config"])

    print(f"✅ Loaded champion model run_id='{run_id}' from {model_uri}")
    
    
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
    try:
        load_model()
    except Exception as e:
        import traceback
        print(f"❌ Model load failed: {e}")
        traceback.print_exc()


async def run_training(overrides=None):
    import tempfile
    env = os.environ.copy()

    if overrides:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(overrides, tmp)
        tmp.close()
        env["TRAIN_OVERRIDES_PATH"] = tmp.name

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            ["python", "scripts/train.py"],
            capture_output=True, text=True, env=env
        )
    )
    return result  # ← just return, let the caller decide what to do
    

# Training state — lets Streamlit poll status
training_status = {"running": False, "last_result": None}

class RetrainRequest(BaseModel):
    overrides: dict = {}

@app.post("/retrain", tags=["Model"])
async def retrain(request: RetrainRequest = RetrainRequest()):
    if training_status["running"]:
        return {"status": "already_running"}

    training_status["running"] = True
    training_status["last_result"] = None

    async def _run():
        try:
            result = await run_training(request.overrides)
            if result.returncode == 0:
                training_status["last_result"] = "success"
                load_model()
            else:
                training_status["last_result"] = f"error: {result.stderr[-500:]}"
        except Exception as e:
            training_status["last_result"] = f"error: {e}"
        finally:
            training_status["running"] = False

    asyncio.create_task(_run())
    return {"status": "training_started"}


@app.get("/retrain-status", tags=["Model"])
async def retrain_status():
    """Poll this to check if training is still running."""
    return {
        "running": training_status["running"],
        "last_result": training_status["last_result"],
        "model": model_meta.get("model_name"),
        "roc_auc": model_meta.get("roc_auc"),
        "version": model_meta.get("run_id"),
    }

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


class ThresholdConfig(BaseModel):
    mode: str = "lower"
    lower: float = 0.30
    upper: float = 0.70


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

def _predict_session(session: SessionFeatures) -> PredictionResult:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_name = model_meta.get("model_name", "unknown")

    df = session_dict_to_dataframe(session.model_dump())

    prob_purchase = float(pipeline.predict_proba(df)[0, 1])
    prob_no_purchase = 1.0 - prob_purchase
    prediction = int(pipeline.predict(df)[0])

    # Intervention logic — single threshold or range
    if threshold_config["mode"] == "range":
        intervene = threshold_config["lower"] <= prob_purchase <= threshold_config["upper"]
    else:
        intervene = prob_purchase < threshold_config["lower"]

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
        intervention_threshold=threshold_config["lower"],
        model_name=model_name,
        confidence=confidence,
    )

def save_threshold_config():
    """Persist threshold config to metadata JSON."""
    if not META_PATH.exists():
        return
    full_meta = json.loads(META_PATH.read_text())
    full_meta["threshold_config"] = threshold_config
    META_PATH.write_text(json.dumps(full_meta, indent=2))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.api_route("/", methods=["GET", "HEAD"], tags=["Health"])
async def root():
    return {
        "status": "ok",
        "model": model_meta.get("model_name", "not loaded"),
        "roc_auc": model_meta.get("roc_auc"),
        "intervention_threshold": model_meta.get("intervention_threshold", INTERVENTION_THRESHOLD),
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    if not model_meta:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return {**model_meta, "threshold_config": threshold_config}


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

@app.post("/threshold", tags=["Config"])
async def set_threshold(config: ThresholdConfig):
    threshold_config["mode"] = config.mode
    threshold_config["lower"] = config.lower
    threshold_config["upper"] = config.upper
    save_threshold_config()
    return {"status": "updated", "config": threshold_config}
