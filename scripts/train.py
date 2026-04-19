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
_dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
if _dagshub_token:
    dagshub.auth.add_app_token(token=_dagshub_token)
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


def train_and_log(model_name, estimator, params, X_train, X_test, y_train, y_test, preprocessor, numeric_imputer_strategy="median", excluded_features=None, threshold_config=None):
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

        tc = threshold_config or {"mode": "lower", "lower": INTERVENTION_THRESHOLD, "upper": 1.0}
        wdr_mode = tc.get("mode", "lower")
        wdr_lower = tc.get("lower", INTERVENTION_THRESHOLD)
        wdr_upper = tc.get("upper", 1.0)
        if wdr_mode == "range":
            interventions = (y_prob >= wdr_lower) & (y_prob <= wdr_upper)
        else:
            interventions = y_prob < wdr_lower
        wasted_discounts = interventions & (y_test.values == 1)
        n_interventions = interventions.sum()
        wasted_discount_rate = wasted_discounts.sum() / n_interventions if n_interventions > 0 else 0
        mlflow.log_metric("wasted_discount_rate", wasted_discount_rate)
        mlflow.log_metric("intervention_count", int(n_interventions))
        mlflow.log_params({"wdr_mode": wdr_mode, "wdr_lower": wdr_lower, "wdr_upper": wdr_upper})

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

    threshold_overrides = overrides.get("_threshold", {})
    threshold_config = {
        "mode": threshold_overrides.get("mode", "lower"),
        "lower": threshold_overrides.get("lower", INTERVENTION_THRESHOLD),
        "upper": threshold_overrides.get("upper", 1.0),
    }
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
            model_name, estimator, params, X_train, X_test, y_train, y_test, preprocessor, numeric_imputer_strategy=numeric_imputer_strategy, excluded_features=excluded_features, threshold_config=threshold_config,
        )
        results.append((model_name, run_id, roc_auc, pipeline))

    client = MlflowClient()

    import datetime
    trained_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    REGISTERED_MODEL_DESCRIPTION = (
        "Shopper purchase-intent classifier for e-commerce session intervention.\n\n"
        "Predicts P(purchase) for a browsing session using 17 behavioral and temporal features "
        "(page views, bounce rates, session duration, month, visitor type, etc.).\n\n"
        f"Sessions with P(purchase) < intervention_threshold (default {INTERVENTION_THRESHOLD}) "
        "are flagged for promotional intervention.\n\n"
        "Champion/challenger aliases always point to the top-2 ROC-AUC versions across ALL runs. "
        "Primary selection metric: ROC-AUC on a held-out 20% test split.\n\n"
        "Training data: UCI Online Shoppers Intention dataset (12,330 sessions, ~16% purchase rate)."
    )

    # --- Register ALL models from this run so they are searchable globally ---
    print("\n📦 Registering all models from this run...")
    current_versions = []
    for name, run_id_i, auc_i, _ in results:
        uri = f"runs:/{run_id_i}/model"
        mv = mlflow.register_model(uri, MODEL_REGISTRY_NAME)
        try:
            client.update_model_version(
                name=MODEL_REGISTRY_NAME,
                version=str(mv.version),
                description=(
                    f"{name}\n\n"
                    f"Trained: {trained_at}\n"
                    f"Test ROC-AUC : {auc_i:.4f}\n"
                    f"Intervention threshold: P(purchase) < {INTERVENTION_THRESHOLD}\n"
                    f"Data version hash: {data_version_hash or 'untracked'}\n"
                    f"MLflow run: {run_id_i}"
                ),
            )
        except Exception as e:
            print(f"  ⚠️  Could not set description for v{mv.version}: {e}")
        current_versions.append((name, run_id_i, auc_i, mv.version))
        print(f"  Registered v{mv.version}: {name} (ROC-AUC={auc_i:.4f})")

    client.update_registered_model(name=MODEL_REGISTRY_NAME, description=REGISTERED_MODEL_DESCRIPTION)

    # --- Find global champion/challenger across ALL registered versions ---
    print("\n🔍 Scanning all registered versions for global champion/challenger...")
    all_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    version_aucs = []
    for v in all_versions:
        try:
            run = client.get_run(v.run_id)
            auc = run.data.metrics.get("roc_auc")
            model_type = run.data.params.get("model_type", "unknown")
            if auc is not None:
                version_aucs.append({
                    "version": int(v.version),
                    "run_id": v.run_id,
                    "roc_auc": auc,
                    "model_name": model_type,
                })
        except Exception:
            pass

    version_aucs.sort(key=lambda x: -x["roc_auc"])

    meta_path = Path("models/best_model_meta.json")
    meta_path.parent.mkdir(exist_ok=True)
    meta = {}

    if version_aucs:
        champ = version_aucs[0]
        client.set_registered_model_alias(MODEL_REGISTRY_NAME, "champion", str(champ["version"]))
        try:
            client.update_model_version(
                name=MODEL_REGISTRY_NAME,
                version=str(champ["version"]),
                description=(
                    f"CHAMPION — {champ['model_name']}\n\n"
                    f"Aliased: {trained_at}\n"
                    f"Test ROC-AUC : {champ['roc_auc']:.4f}\n"
                    f"MLflow run: {champ['run_id']}\n\n"
                    "Global champion: highest ROC-AUC across all registered versions."
                ),
            )
        except Exception as e:
            print(f"  ⚠️  Could not set champion description: {e}")
        meta["champion"] = {
            "model_name": champ["model_name"],
            "run_id": champ["run_id"],
            "roc_auc": champ["roc_auc"],
            "version": champ["version"],
            "intervention_threshold": INTERVENTION_THRESHOLD,
        }
        print(f"✅ Global champion: v{champ['version']} ({champ['model_name']}, ROC-AUC={champ['roc_auc']:.4f})")

    if len(version_aucs) > 1:
        chal = version_aucs[1]
        client.set_registered_model_alias(MODEL_REGISTRY_NAME, "challenger", str(chal["version"]))
        try:
            client.update_model_version(
                name=MODEL_REGISTRY_NAME,
                version=str(chal["version"]),
                description=(
                    f"CHALLENGER — {chal['model_name']}\n\n"
                    f"Aliased: {trained_at}\n"
                    f"Test ROC-AUC : {chal['roc_auc']:.4f}\n"
                    f"MLflow run: {chal['run_id']}\n\n"
                    f"Global challenger: second-highest ROC-AUC across all registered versions. "
                    f"Champion is v{version_aucs[0]['version']} ({version_aucs[0]['model_name']}, AUC={version_aucs[0]['roc_auc']:.4f})."
                ),
            )
        except Exception as e:
            print(f"  ⚠️  Could not set challenger description: {e}")
        meta["challenger"] = {
            "model_name": chal["model_name"],
            "run_id": chal["run_id"],
            "roc_auc": chal["roc_auc"],
            "version": chal["version"],
            "intervention_threshold": INTERVENTION_THRESHOLD,
        }
        print(f"✅ Global challenger: v{chal['version']} ({chal['model_name']}, ROC-AUC={chal['roc_auc']:.4f})")
    
    print(f"✅ Model metadata saved → {meta_path}")

    print("\n📊 This Run — Leaderboard (Test ROC-AUC):")
    print(f"  {'Model':<22} {'ROC-AUC':>10}  {'Version':>8}  {'Run ID'}")
    print(f"  {'-'*22} {'-'*10}  {'-'*8}  {'-'*36}")
    for name, run_id_i, auc_i, version_i in sorted(current_versions, key=lambda r: -r[2]):
        champ_marker = " ← global champion" if meta.get("champion", {}).get("version") == version_i else ""
        chal_marker  = " ← global challenger" if meta.get("challenger", {}).get("version") == version_i else ""
        print(f"  {name:<22} {auc_i:>10.4f}  v{version_i:>7}  {run_id_i}{champ_marker}{chal_marker}")

    print(f"\n✅ Training complete.")


if __name__ == "__main__":
    main()
