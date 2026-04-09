"""
train.py — Train and evaluate three classifiers on the online shoppers intention dataset.
Logs every run to MLflow, saves the best model for the FastAPI server.

Models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost

Usage:
    python scripts/train.py [--data data/online_shoppers_intention.csv]
"""

import argparse
import os
import sys
import warnings
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

import dagshub.auth
dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN", ""))
import dagshub
dagshub.init(repo_owner='smbrownai', repo_name='shopper_intervention', mlflow=True)

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.features import load_data, build_preprocessor, validate_data

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
        #(
        #    "LR_high_regularization",
        #    LogisticRegression(
        #        C=get("LR_high_regularization", "C", 0.01),
        #        solver=get("LR_high_regularization", "solver", "lbfgs"),
        #        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        #    ),
        #    {"C": get("LR_high_regularization", "C", 0.01), "solver": get("LR_high_regularization", "solver", "lbfgs")},
        #),
        #(
        #    "LR_L1",
        #    LogisticRegression(
        #        C=get("LR_L1", "C", 1.0),
        #        penalty=get("LR_L1", "penalty", "l1"),
        #        solver=get("LR_L1", "solver", "saga"),
        #        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        #    ),
        #    {"C": get("LR_L1", "C", 1.0), "penalty": get("LR_L1", "penalty", "l1"), "solver": get("LR_L1", "solver", "saga")},
        #),
        #(
        #    "DT_shallow",
        #    DecisionTreeClassifier(
        #        max_depth=get("DT_shallow", "max_depth", 4),
        #        class_weight="balanced", random_state=RANDOM_STATE
        #    ),
        #    {"max_depth": get("DT_shallow", "max_depth", 4)},
        #),
        (
            "DT_medium",
            DecisionTreeClassifier(
                max_depth=get("DT_medium", "max_depth", 8),
                min_samples_leaf=get("DT_medium", "min_samples_leaf", 10),
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"max_depth": get("DT_medium", "max_depth", 8), "min_samples_leaf": get("DT_medium", "min_samples_leaf", 10)},
        ),
        #(
        #    "DT_entropy",
        #    DecisionTreeClassifier(
        #        max_depth=get("DT_entropy", "max_depth", 8),
        #        criterion=get("DT_entropy", "criterion", "entropy"),
        #        class_weight="balanced", random_state=RANDOM_STATE
        #    ),
        #    {"max_depth": get("DT_entropy", "max_depth", 8), "criterion": get("DT_entropy", "criterion", "entropy")},
        #),
        (
            "RF_baseline",
            RandomForestClassifier(
                n_estimators=get("RF_baseline", "n_estimators", 200),
                max_depth=get("RF_baseline", "max_depth", 12),
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            {"n_estimators": get("RF_baseline", "n_estimators", 200), "max_depth": get("RF_baseline", "max_depth", 12), "max_features": "sqrt"},
        ),
        #(
        #    "RF_log2_features",
        #    RandomForestClassifier(
        #        n_estimators=get("RF_log2_features", "n_estimators", 200),
        #        max_depth=get("RF_log2_features", "max_depth", 12),
        #        max_features=get("RF_log2_features", "max_features", "log2"),
        #        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        #    ),
        #    {"n_estimators": get("RF_log2_features", "n_estimators", 200), "max_depth": get("RF_log2_features", "max_depth", 12), "max_features": get("RF_log2_features", "max_features", "log2")},
        #),
        #(
        #    "RF_deep",
        #    RandomForestClassifier(
        #        n_estimators=get("RF_deep", "n_estimators", 300),
        #        max_depth=get("RF_deep", "max_depth", None),
        #        max_features=get("RF_deep", "max_features", "sqrt"),
        #        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        #    ),
        #    {"n_estimators": get("RF_deep", "n_estimators", 300), "max_depth": get("RF_deep", "max_depth", "None"), "max_features": get("RF_deep", "max_features", "sqrt")},
        #),
        #(
        #    "GradientBoosting",
        #    GradientBoostingClassifier(
        #        n_estimators=get("GradientBoosting", "n_estimators", 200),
        #        learning_rate=get("GradientBoosting", "learning_rate", 0.05),
        #        max_depth=get("GradientBoosting", "max_depth", 4),
        #        min_samples_leaf=get("GradientBoosting", "min_samples_leaf", 10),
        #        subsample=get("GradientBoosting", "subsample", 0.8),
        #        random_state=RANDOM_STATE
        #    ),
        #    {"n_estimators": get("GradientBoosting", "n_estimators", 200), "learning_rate": get("GradientBoosting", "learning_rate", 0.05), "max_depth": get("GradientBoosting", "max_depth", 4), "subsample": get("GradientBoosting", "subsample", 0.8)},
        #),
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


def train_and_log(model_name, estimator, params, X_train, X_test, y_train, y_test, preprocessor, numeric_imputer_strategy="median", excluded_features=None):
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
            "numeric_imputer_strategy": numeric_imputer_strategy,
            "excluded_features": str(excluded_features) or "none",
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
            num_feature_names = preprocessor.transformers_[0][2]  # active numeric features
            feat_names = list(num_feature_names) + list(ohe_cols)
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
        

def main():
    parser = argparse.ArgumentParser(description="Train shopper purchase prediction models")
    parser.add_argument("--data", default="data/online_shoppers_intention.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    is_cloud = os.getenv("RENDER") == "true"

    # Load overrides if passed from API
    overrides_path = os.getenv("TRAIN_OVERRIDES_PATH")
    overrides = json.loads(Path(overrides_path).read_text()) if overrides_path else {}

    CLOUD_MODELS = {"LR_baseline", "DT_medium", "RF_baseline", "XGBoost"}

    if is_cloud:
        model_configs = [
            c for c in build_model_configs(overrides)
            if c[0] in CLOUD_MODELS
        ]
    else:
        model_configs = build_model_configs(overrides)

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"⬇️  Data file not found locally, fetching from DagHub ...")
        import pandas as pd
        import urllib.request
        data_url = "https://dagshub.com/smbrownai/shopper_intervention/raw/main/data/online_shoppers_intention.csv"
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(data_url, str(data_path))
            print(f"   ✅ Downloaded to {data_path}")
        except Exception as e:
            print(f"ERROR: could not fetch data: {e}")
            sys.exit(1)

    print(f"\n📂 Loading data from {data_path} ...")
    import pandas as pd
    raw_df = pd.read_csv(str(data_path))

    print("\n🔍 Validating data ...")
    validation = validate_data(raw_df)

    for msg in validation["errors"]:
        print(f"   ❌ ERROR: {msg}")
    for msg in validation["warnings"]:
        print(f"   ⚠️  WARNING: {msg}")
    if validation["passed"] and not validation["warnings"]:
        print("   ✅ All checks passed")

    if not validation["passed"]:
        print("\n🚫 Training aborted due to data validation errors.")
        sys.exit(1)


    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n🔬 MLflow experiment: '{EXPERIMENT_NAME}'")

    # Read DVC hash for this data file (links MLflow run to exact data version)
    dvc_file = data_path.parent / (data_path.name + ".dvc")
    data_version_hash = None
    if dvc_file.exists():
        import yaml
        dvc_meta = yaml.safe_load(dvc_file.read_text())
        data_version_hash = dvc_meta["outs"][0].get("md5", "unknown")

    # Log validation stats to a shared parent run so every training session has a data quality record
    with mlflow.start_run(run_name="data_validation") as val_run:
        mlflow.log_params({
            "data_path": str(data_path),
            "warnings": len(validation["warnings"]),
            "data_version_hash": data_version_hash or "untracked",
        })
        mlflow.log_metrics({k: v for k, v in validation["stats"].items() if isinstance(v, (int, float))})
        mlflow.log_dict({"warnings": validation["warnings"], "stats": validation["stats"]}, "validation_report.json")

    preprocessor_overrides = overrides.get("_preprocessor", {})
    numeric_imputer_strategy = preprocessor_overrides.get("numeric_imputer_strategy", "median")
    excluded_features = preprocessor_overrides.get("excluded_features", [])
    drop_duplicates = preprocessor_overrides.get("drop_duplicates", False)
    preprocessor = build_preprocessor(
        numeric_imputer_strategy=numeric_imputer_strategy,
        excluded_features=excluded_features
    )

    X, y = load_data(str(data_path), drop_duplicates=drop_duplicates)
    if drop_duplicates:
        n_after = len(X)
        print(f"   Duplicates removed: {len(raw_df) - n_after:,}  |  Rows after dedup: {n_after:,}  |  Purchase rate: {y.mean()*100:.1f}%")
    else:
        print(f"   Rows: {len(X):,}  |  Purchase rate: {y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    results = []
    for model_name, estimator, params in model_configs:
        run_id, roc_auc, pipeline = train_and_log(
            model_name, estimator, params, X_train, X_test, y_train, y_test, preprocessor, numeric_imputer_strategy=numeric_imputer_strategy, excluded_features=excluded_features,
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

    client = MlflowClient()
    
    # Register champion
    champion_uri = f"runs:/{best_run_id}/model"
    champion_mv = mlflow.register_model(champion_uri, MODEL_REGISTRY_NAME)
    client.set_registered_model_alias(
        MODEL_REGISTRY_NAME, "champion", champion_mv.version
    )
    meta["champion"]["version"] = champion_mv.version
    
    # Rewrite meta with version included
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Champion registered: v{champion_mv.version} ({best_name})")
    
    # Register challenger (if exists)
    if "challenger" in meta:
        challenger_run_id = meta["challenger"]["run_id"]
        challenger_uri = f"runs:/{challenger_run_id}/model"
        challenger_mv = mlflow.register_model(challenger_uri, MODEL_REGISTRY_NAME)
        client.set_registered_model_alias(
            MODEL_REGISTRY_NAME, "challenger", challenger_mv.version
        )
        print(f"✅ Challenger registered: v{challenger_mv.version} ({meta['challenger']['model_name']})")
    
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
