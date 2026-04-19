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
import time
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
MODEL_REGISTRY_NAME = "shopper_best_model"
INTERVENTION_THRESHOLD = 0.30  # fallback if meta not loaded

pipeline = None
pipeline_challenger = None
model_meta = {}
challenger_meta = {}

# Cache for threshold optimizer — invalidated when load_model() is called
_optimizer_cache: dict = {}  # keys: "y", "y_prob"

# Runtime threshold config
threshold_config = {
    "mode": "lower",
    "lower": 0.30,
    "upper": 0.70,
}

def _fetch_run_metrics(client, run_id: str) -> dict:
    """Fetch metrics and WDR params from an MLflow run; return empty dict on failure."""
    try:
        run = client.get_run(run_id)
        m = run.data.metrics
        p = run.data.params
        return {
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "wasted_discount_rate": m.get("wasted_discount_rate"),
            "intervention_count": m.get("intervention_count"),
            "wdr_mode": p.get("wdr_mode"),
            "wdr_lower": float(p["wdr_lower"]) if p.get("wdr_lower") is not None else None,
            "wdr_upper": float(p["wdr_upper"]) if p.get("wdr_upper") is not None else None,
        }
    except Exception:
        return {}


def load_model():
    global pipeline, pipeline_challenger, model_meta, challenger_meta, threshold_config
    _optimizer_cache.clear()  # invalidate scored probabilities after model reload

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    full_meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    meta_champion = full_meta.get("champion", {})

    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    # Always resolve via the registry alias so we load whatever is currently
    # tagged "champion" — not a potentially stale run_id from the JSON file.
    champion_uri = f"models:/{MODEL_REGISTRY_NAME}@champion"
    pipeline = mlflow.sklearn.load_model(champion_uri)

    # Resolve the actual run_id and version from the registry alias
    champion_mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "champion")
    run_id = champion_mv.run_id
    version = champion_mv.version

    champion_metrics = _fetch_run_metrics(client, run_id)
    run_data = client.get_run(run_id).data if run_id else None

    challenger_meta = {}
    try:
        challenger_uri = f"models:/{MODEL_REGISTRY_NAME}@challenger"
        pipeline_challenger = mlflow.sklearn.load_model(challenger_uri)
        challenger_mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "challenger")
        ch_run_id = challenger_mv.run_id
        challenger_metrics = _fetch_run_metrics(client, ch_run_id)
        ch_run_data = client.get_run(ch_run_id).data
        challenger_meta = {
            "model_name": ch_run_data.params.get("model_type", full_meta.get("challenger", {}).get("model_name", "—")),
            "run_id": ch_run_id,
            "version": challenger_mv.version,
            "roc_auc": ch_run_data.metrics.get("roc_auc", full_meta.get("challenger", {}).get("roc_auc", 0.0)),
            **challenger_metrics,
        }
        print(f"✅ Loaded challenger model v{challenger_mv.version} run_id='{ch_run_id}'")
    except Exception as e:
        print(f"⚠️ Could not load challenger model: {e}")

    model_meta = {
        "model_name": run_data.params.get("model_type", meta_champion.get("model_name", "—")) if run_data else meta_champion.get("model_name", "—"),
        "run_id": run_id,
        "version": version,
        "roc_auc": (run_data.metrics.get("roc_auc", meta_champion.get("roc_auc", 0.0)) if run_data else meta_champion.get("roc_auc", 0.0)),
        "intervention_threshold": meta_champion.get("intervention_threshold", INTERVENTION_THRESHOLD),
        **champion_metrics,
    }

    # Load threshold config from metadata if present
    if "threshold_config" in full_meta:
        threshold_config.update(full_meta["threshold_config"])

    print(f"✅ Loaded champion model run_id='{run_id}' from {champion_uri}")
    import threading
    threading.Thread(target=_warm_optimizer_cache, daemon=True).start()
    
    
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


def _warm_optimizer_cache():
    """Score the full dataset with the current pipeline and cache results.
    Runs in a background thread so it doesn't block startup or requests."""
    if pipeline is None or "y_prob" in _optimizer_cache:
        return
    try:
        import pandas as pd
        print("🔥 Pre-warming optimizer cache...")
        data_path = ROOT / "data" / "online_shoppers_intention.csv"
        if not data_path.exists():
            data_path = "https://dagshub.com/smbrownai/shopper_intervention/raw/main/data/online_shoppers_intention.csv"
        df = pd.read_csv(data_path)
        _optimizer_cache["y"] = df["Revenue"].astype(int).values
        _optimizer_cache["y_prob"] = pipeline.predict_proba(df.drop(columns=["Revenue"]))[:, 1]
        print(f"✅ Optimizer cache warm — {len(_optimizer_cache['y']):,} sessions scored.")
    except Exception as e:
        print(f"⚠️ Optimizer cache warm failed: {e}")


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
                output = (result.stderr or "") + (result.stdout or "")
                training_status["last_result"] = f"error: {output[-1000:] or '(no output captured)'}"
        except Exception as e:
            training_status["last_result"] = f"error: {e}"
        finally:
            if training_status["last_result"] is None:
                training_status["last_result"] = "error: training task ended without a result"
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
    use_challenger: bool = Field(False, description="If True, use challenger model instead of champion")

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
    inference_ms: float = Field(..., description="Model inference time in milliseconds")


class ThresholdConfig(BaseModel):
    mode: str = "lower"
    lower: float = 0.30
    upper: float = 0.70


class BatchRequest(BaseModel):
    sessions: list[SessionFeatures]
    use_challenger: bool = False


class BatchResult(BaseModel):
    results: list[PredictionResult]
    total_sessions: int
    intervention_count: int
    intervention_rate: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict_session(session: SessionFeatures, use_challenger: bool = False) -> PredictionResult:
    use_challenger = use_challenger and pipeline_challenger is not None
    active_pipeline = pipeline_challenger if use_challenger else pipeline
    active_name = challenger_meta.get("model_name", "unknown") if use_challenger else model_meta.get("model_name", "unknown")

    if active_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = session_dict_to_dataframe(session.model_dump())
    start = time.perf_counter()
    prob_purchase = float(active_pipeline.predict_proba(df)[0, 1])
    elapsed_ms = (time.perf_counter() - start) * 1000
    prob_no_purchase = 1.0 - prob_purchase
    prediction = int(active_pipeline.predict(df)[0])

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
        model_name=active_name,
        confidence=confidence,
        inference_ms=elapsed_ms
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
@app.post("/admin/reload-model", tags=["Admin"])
def reload_model():
    """
    Reload the champion model from MLflow without restarting the server.
    """
    global pipeline
    champion_uri = f"models:/{MODEL_REGISTRY_NAME}@champion"
    pipeline = mlflow.sklearn.load_model(champion_uri)
    return {"status": "champion model reloaded"}


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
    return {
        "champion": model_meta,
        "challenger": challenger_meta or None,
        "threshold_config": threshold_config,
    }


@app.get("/model-history", tags=["Model"])
async def model_history():
    """Summary of every MLflow experiment run grouped by model type."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Search all experiment runs — covers every trained model, not just
        # registry-registered ones. Exclude the data_validation parent run.
        experiment = client.get_experiment_by_name("shopper_purchase_prediction")
        if experiment is None:
            return {"models": []}

        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="params.model_type != ''",
            max_results=5000,
        )

        by_type: dict = {}
        for run in all_runs:
            auc = run.data.metrics.get("roc_auc")
            model_type = run.data.params.get("model_type")
            if not model_type or auc is None:
                continue
            entry = by_type.setdefault(model_type, {
                "model_type": model_type,
                "run_count": 0,
                "best_roc_auc": None,
                "best_run_id": None,
            })
            entry["run_count"] += 1
            if entry["best_roc_auc"] is None or auc > entry["best_roc_auc"]:
                entry["best_roc_auc"] = auc
                entry["best_run_id"] = run.info.run_id

        rows = sorted(by_type.values(), key=lambda r: -(r["best_roc_auc"] or 0))
        return {"models": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(session: SessionFeatures):
    """
    Score a single browser session.

    Returns purchase probability and whether to show a promotional offer.
    """
    return _predict_session(session, use_challenger=session.use_challenger)


@app.post("/predict-batch", response_model=BatchResult, tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    if len(batch.sessions) == 0:
        raise HTTPException(status_code=400, detail="sessions list must not be empty")
    if len(batch.sessions) > 25000:
        raise HTTPException(status_code=400, detail="Max 25,000 sessions per batch")
    results = [_predict_session(s, use_challenger=batch.use_challenger) for s in batch.sessions]
    intervention_count = sum(1 for r in results if r.intervene)
    return BatchResult(
        results=results,
        total_sessions=len(results),
        intervention_count=intervention_count,
        intervention_rate=round(intervention_count / len(results), 4),
    )

@app.post("/recommend-threshold", tags=["Config"])
async def recommend_threshold(payload: dict):
    """
    Given a target WDR, sweep both single and range modes and return
    whichever achieves the target WDR with the most intervention coverage.

    Body: { "target_wdr": 0.20 }
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    target_wdr = float(payload.get("target_wdr", 0.30))

    try:
        import numpy as np
        import pandas as pd

        if "y_prob" not in _optimizer_cache:
            raise HTTPException(
                status_code=503,
                detail="Optimizer cache is still warming up — please wait 30 seconds and try again.",
            )

        y = _optimizer_cache["y"]
        y_prob = _optimizer_cache["y_prob"]

        thresholds = np.linspace(0.05, 0.95, 181)  # 0.5% steps

        def _best_single():
            """Highest-coverage single threshold that meets target WDR."""
            candidates = []
            for t in thresholds:
                mask = y_prob < t
                n = int(mask.sum())
                if n == 0:
                    continue
                wdr = float((mask & (y == 1)).sum() / n)
                candidates.append((t, wdr, n))
            # prefer those that meet target (wdr ≤ target), maximise coverage
            meeting = [(t, w, n) for t, w, n in candidates if w <= target_wdr]
            if meeting:
                t, w, n = max(meeting, key=lambda x: x[2])
            else:
                # none meets target — pick closest
                t, w, n = min(candidates, key=lambda x: abs(x[1] - target_wdr))
            return {"mode": "single", "recommended_lower": round(float(t), 3),
                    "recommended_upper": None, "achieved_wdr": round(w, 4), "intervention_count": n}

        def _best_range():
            """Fix lower=0.30, sweep upper; highest-coverage range that meets target WDR."""
            lower = 0.30
            candidates = []
            for upper in thresholds[thresholds > lower]:
                mask = (y_prob >= lower) & (y_prob <= upper)
                n = int(mask.sum())
                if n == 0:
                    continue
                wdr = float((mask & (y == 1)).sum() / n)
                candidates.append((upper, wdr, n))
            meeting = [(u, w, n) for u, w, n in candidates if w <= target_wdr]
            if meeting:
                u, w, n = max(meeting, key=lambda x: x[2])
            else:
                u, w, n = min(candidates, key=lambda x: abs(x[1] - target_wdr))
            return {"mode": "range", "recommended_lower": lower,
                    "recommended_upper": round(float(u), 3), "achieved_wdr": round(w, 4), "intervention_count": n}

        single = _best_single()
        rng = _best_range()

        # Pick the mode that meets the target; if both do, pick higher coverage.
        # If neither does, pick the one closest to target.
        def _score(r):
            meets = r["achieved_wdr"] <= target_wdr
            return (meets, r["intervention_count"] if meets else -abs(r["achieved_wdr"] - target_wdr))

        best = single if _score(single) >= _score(rng) else rng
        best["target_wdr"] = target_wdr
        best["single"] = single
        best["range"] = rng
        return best

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threshold", tags=["Config"])
async def get_threshold():
    return threshold_config

@app.post("/threshold", tags=["Config"])
async def set_threshold(config: ThresholdConfig):
    threshold_config["mode"] = config.mode
    threshold_config["lower"] = config.lower
    threshold_config["upper"] = config.upper
    save_threshold_config()
    return {"status": "updated", "config": threshold_config}