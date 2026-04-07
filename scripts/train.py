"""
train.py — Train and evaluate three classifiers on the online shoppers intention dataset.
Logs every run to MLflow, saves the best model for the FastAPI server.

Models:
  1. Logistic Regression (x3)
  2. Decision Tree (x3)
  3. Random Forest (x3)
  4. Gradient Boosting and XGBoost

Usage:
    python scripts/train.py [--data data/online_shoppers_intention.csv]
"""

import argparse
import os
import sys
import time
import warnings
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

import dagshub
dagshub.init(
    repo_owner='smbrownai',
    repo_name='shopper_intervention',
    mlflow=True,
    token=os.getenv("DAGSHUB_TOKEN")
)

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.features import load_data, build_preprocessor, ALL_FEATURES, TARGET

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "shopper_purchase_prediction"
MODEL_REGISTRY_NAME = "shopper_best_model"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Intervention threshold: sessions with P(purchase) < this get flagged
INTERVENTION_THRESHOLD = 0.30


def build_model_configs(overrides=None):
    overrides = overrides or {}

    def get(model_name, param, default):
        return overrides.get(model_name, {}).get(param, default)

    return [
        (
            "LR_baseline",
            LogisticRegression(
                C=get("LR_baseline", "C", 1.0),
                solver=get("LR_baseline", "solver", "lbfgs"),
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
            {"C": get("LR_baseline", "C", 1.0), "solver": get("LR_baseline", "solver", "lbfgs")},
        ),
        (
            "LR_high_regularization",
            LogisticRegression(
                C=get("LR_high_regularization", "C", 0.01),
                solver=get("LR_high_regularization", "solver", "lbfgs"),
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
            {"C": get("LR_high_regularization", "C", 0.01), "solver": get("LR_high_regularization", "solver", "lbfgs")},
        ),
        (
            "LR_L1",
            LogisticRegression(
                C=get("LR_L1", "C", 1.0),
                penalty=get("LR_L1", "penalty", "l1"),
                solver=get("LR_L1", "solver", "saga"),
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
            {"C": get("LR_L1", "C", 1.0), "penalty": get("LR_L1", "penalty", "l1"), "solver": get("LR_L1", "solver", "saga")},
        ),
        (
            "DT_shallow",
            DecisionTreeClassifier(
                max_depth=get("DT_shallow", "max_depth", 4),
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"max_depth": get("DT_shallow", "max_depth", 4)},
        ),
        (
            "DT_medium",
            DecisionTreeClassifier(
                max_depth=get("DT_medium", "max_depth", 8),
                min_samples_leaf=get("DT_medium", "min_samples_leaf", 10),
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"max_depth": get("DT_medium", "max_depth", 8), "min_samples_leaf": get("DT_medium", "min_samples_leaf", 10)},
        ),
        (
            "DT_entropy",
            DecisionTreeClassifier(
                max_depth=get("DT_entropy", "max_depth", 8),
                criterion=get("DT_entropy", "criterion", "entropy"),
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"max_depth": get("DT_entropy", "max_depth", 8), "criterion": get("DT_entropy", "criterion", "entropy")},
        ),
        (
            "RF_baseline",
            RandomForestClassifier(
                n_estimators=get("RF_baseline", "n_estimators", 200),
                max_depth=get("RF_baseline", "max_depth", 12),
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            {"n_estimators": get("RF_baseline", "n_estimators", 200), "max_depth": get("RF_baseline", "max_depth", 12), "max_features": "sqrt"},
        ),
        (
            "RF_log2_features",
            RandomForestClassifier(
                n_estimators=get("RF_log2_features", "n_estimators", 200),
                max_depth=get("RF_log2_features", "max_depth", 12),
                max_features=get("RF_log2_features", "max_features", "log2"),
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            {"n_estimators": get("RF_log2_features", "n_estimators", 200), "max_depth": get("RF_log2_features", "max_depth", 12), "max_features": get("RF_log2_features", "max_features", "log2")},
        ),
        (
            "RF_deep",
            RandomForestClassifier(
                n_estimators=get("RF_deep", "n_estimators", 300),
                max_depth=get("RF_deep", "max_depth", None),
                max_features=get("RF_deep", "max_features", "sqrt"),
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            {"n_estimators": get("RF_deep", "n_estimators", 300), "max_depth": get("RF_deep", "max_depth", "None"), "max_features": get("RF_deep", "max_features", "sqrt")},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=get("GradientBoosting", "n_estimators", 200),
                learning_rate=get("GradientBoosting", "learning_rate", 0.05),
                max_depth=get("GradientBoosting", "max_depth", 4),
                min_samples_leaf=get("GradientBoosting", "min_samples_leaf", 10),
                subsample=get("GradientBoosting", "subsample", 0.8),
                random_state=RANDOM_STATE
            ),
            {"n_estimators": get("GradientBoosting", "n_estimators", 200), "learning_rate": get("GradientBoosting", "learning_rate", 0.05), "max_depth": get("GradientBoosting", "max_depth", 4), "subsample": get("GradientBoosting", "subsample", 0.8)},
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=get("XGBoost", "n_estimators", 200),
                learning_rate=get("XGBoost", "learning_rate", 0.05),
                max_depth=get("XGBoost", "max_depth", 4),
                subsample=get("XGBoost", "subsample", 0.8),
                colsample_bytree=get("XGBoost", "colsample_bytree", 0.8),
                scale_pos_weight=get("XGBoost", "scale_pos_weight", 5),
                eval_metric="auc", random_state=RANDOM_STATE, verbosity=0
            ),
            {"n_estimators": get("XGBoost", "n_estimators", 200), "learning_rate": get("XGBoost", "learning_rate", 0.05), "max_depth": get("XGBoost", "max_depth", 4), "subsample": get("XGBoost", "subsample", 0.8), "colsample_bytree": get("XGBoost", "colsample_bytree", 0.8), "scale_pos_weight": get("XGBoost", "scale_pos_weight", 5)},
        ),
    ]


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_and_log(model_name, estimator, params, X_train, X_test, y_train, y_test, preprocessor):
    """Train one model, log to MLflow, return (run_id, roc_auc, pipeline)."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=model_name) as run:
        # --- Build full pipeline ---
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        # --- Cross-validation on training set ---
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # --- Fit on full training set ---
        pipeline.fit(X_train, y_train)

        # --- Evaluate on test set ---
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        print(f"  Test Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Test ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  Test F1       : {metrics['f1']:.4f}")
        print(f"  Test Recall   : {metrics['recall']:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]))

        # --- Log to MLflow ---
        mlflow.log_params(params)
        mlflow.log_params({
            "model_type": model_name,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "intervention_threshold": INTERVENTION_THRESHOLD,
            "class_weight": "balanced",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        # Confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        }
        mlflow.log_dict(cm_dict, "confusion_matrix.json")

        # Feature importance (RF / DT)
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            ohe_cols = (
                pipeline.named_steps["preprocessor"]
                .transformers_[1][1]
                .named_steps["ohe"]
                .get_feature_names_out()
            )
            from scripts.features import NUMERIC_FEATURES
            feat_names = list(NUMERIC_FEATURES) + list(ohe_cols)
            importances = dict(zip(feat_names, classifier.feature_importances_.tolist()))
            mlflow.log_dict(importances, "feature_importances.json")

        # Log model and register immediately while run is still active
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=None,
        )
        
        # Register while run is still open
        run_id = run.info.run_id
        print(f"  MLflow run_id: {run_id}")
        return run_id, metrics["roc_auc"], pipeline
        

def save_best_model_metadata(model_name: str, run_id: str, roc_auc: float, models_dir: Path):
    """Write metadata JSON so the API server can load the right model."""
    meta = {
        "model_name": model_name,
        "run_id": run_id,
        "roc_auc": roc_auc,
        "intervention_threshold": INTERVENTION_THRESHOLD,
    }
    meta_path = models_dir / "best_model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n✅ Best model metadata saved → {meta_path}")
    return meta_path


def main():
    parser = argparse.ArgumentParser(description="Train shopper purchase prediction models")
    parser.add_argument("--data", default="data/online_shoppers_intention.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    is_cloud = os.getenv("RENDER") == "true"

    # Load overrides if passed from API
    overrides_path = os.getenv("TRAIN_OVERRIDES_PATH")
    overrides = json.loads(Path(overrides_path).read_text()) if overrides_path else {}

    if is_cloud:
        model_configs = [
            ("LR_baseline", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE), {"C": 1.0}),
            ("DT_medium", DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE), {"max_depth": 8}),
            ("XGBoost", XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8, scale_pos_weight=5, eval_metric="auc", random_state=RANDOM_STATE, verbosity=0), {"n_estimators": 100}),
        ]
    else:
        model_configs = build_model_configs(overrides)

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)

    print(f"\n📂 Loading data from {data_path} ...")
    X, y = load_data(str(data_path))
    print(f"   Rows: {len(X):,}  |  Purchase rate: {y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n🔬 MLflow experiment: '{EXPERIMENT_NAME}'")

    preprocessor = build_preprocessor()

    results = []
    for model_name, estimator, params in model_configs:
        run_id, roc_auc, pipeline = train_and_log(
            model_name, estimator, params, X_train, X_test, y_train, y_test, preprocessor
        )
        results.append((model_name, run_id, roc_auc, pipeline))

    best = max(results, key=lambda r: r[2])
    best_name, best_run_id, best_auc, best_pipeline = best

    sorted_results = sorted(results, key=lambda r: -r[2])
    meta = {
        "champion": {
            "model_name": best_name,
            "run_id": best_run_id,
            "roc_auc": best_auc,
            "intervention_threshold": INTERVENTION_THRESHOLD,
        }
    }
    if len(sorted_results) > 1:
        second_best_name, second_best_run_id, second_best_auc, _ = sorted_results[1]
        meta["challenger"] = {
            "model_name": second_best_name,
            "run_id": second_best_run_id,
            "roc_auc": second_best_auc,
            "intervention_threshold": INTERVENTION_THRESHOLD,
        }
    
    meta_path = Path("models/best_model_meta.json")
    meta_path.parent.mkdir(exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Model metadata saved → {meta_path}")
    
    print("\n📊 Model Leaderboard (Test ROC-AUC):")
    print(f"  {'Model':<22} {'ROC-AUC':>10}  {'Run ID'}")
    print(f"  {'-'*22} {'-'*10}  {'-'*36}")
    for name, run_id, auc, _ in sorted(results, key=lambda r: -r[2]):
        marker = " ← best" if name == best_name else ""
        print(f"  {name:<22} {auc:>10.4f}  {run_id}{marker}")

    print(f"\n✅ Training complete.")


if __name__ == "__main__":
    main()
