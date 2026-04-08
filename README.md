# 🛒 Online Shopper Intervention System

Predicts e-commerce sessions **unlikely to convert** so a promotional incentive can be
offered in real time. Built as a full ML portfolio project with a training pipeline,
REST API, and interactive dashboard.

## Live Demo
- **Dashboard:** [shopintervene.streamlit.app](https://shopintervene.streamlit.app)
- **API Docs:** [shopper-intervention.onrender.com/docs](https://shopper-intervention.onrender.com/docs)
- **MLflow Experiments:** [dagshub.com/smbrownai/shopper_intervention](https://dagshub.com/smbrownai/shopper_intervention.mlflow)

---

## Architecture

```
data/
  online_shoppers_intention.csv   — UCI dataset (12,330 sessions)

scripts/
  features.py                     — Shared preprocessing pipeline (train + inference)
  train.py                        — MLflow training: LR, DT, RF, XGBoost

api/
  main.py                         — FastAPI inference server

ui/
  app.py                          — Streamlit dashboard

models/
  best_model_meta.json            — Champion/challenger metadata (synced with MLflow Registry)
```

---

## Models

Four classifiers are trained and compared on every run:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, balanced class weights |
| Decision Tree | Configurable depth and criterion |
| Random Forest | Ensemble, balanced class weights |
| XGBoost | Gradient boosting with scale_pos_weight |

The best model by **ROC-AUC** is promoted to **champion** and registered in the
MLflow Model Registry. The second-best becomes the **challenger**. Both are loaded
at API startup and selectable from the UI.

---

## Preprocessing Pipeline

Defined in `scripts/features.py` and shared between training and inference:

- **Numeric features** — median or mean imputed (configurable), then StandardScaler
- **Categorical features** — mode imputed, then OneHotEncoded (`handle_unknown="ignore"`)
- **Feature groups** — Technical and Engagement Rate features can be excluded at
  training time to evaluate their impact on model performance
- **No data leakage** — preprocessor is fit on training data only, applied consistently
  at inference time

---

## Intervention Logic

The model outputs **P(purchase)** for each session. Sessions below a configurable
threshold (default 30%) are flagged for intervention — a promotional incentive such
as a coupon or free shipping offer.

The threshold is a **business decision** applied to model output, not a model parameter.
It can be set as a single cutoff or a probability range, and is persisted across
API restarts.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export DAGSHUB_TOKEN=your_token_here
export MLFLOW_TRACKING_URI=https://dagshub.com/smbrownai/shopper_intervention.mlflow
```

### 3. Train models
```bash
python scripts/train.py
# Logs all runs to MLflow, registers champion and challenger in MLflow Registry,
# writes models/best_model_meta.json
```

### 4. Start FastAPI server
```bash
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### 5. Launch Streamlit dashboard
```bash
streamlit run ui/app.py
```

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| 📊 Dataset Explorer | EDA charts — purchase rates, visitor types, page values, bounce/exit rates |
| 🎯 Score a Session | Manual feature entry → live prediction → intervention decision + gauge chart |
| 📋 Batch Scoring | Upload CSV → batch predictions → 5 analysis charts + downloadable results |
| 📈 Model Performance | Champion and challenger metrics, intervention logic explanation |
| 🔁 Retrain Model | Hyperparameter overrides, preprocessing options, kick off training from UI |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check, current model info |
| GET | `/model-info` | Champion/challenger metadata |
| POST | `/predict` | Score a single session |
| POST | `/predict-batch` | Score up to 25,000 sessions |
| GET/POST | `/threshold` | Get or set intervention threshold |
| POST | `/retrain` | Trigger a training run |
| GET | `/retrain-status` | Poll training progress |

---

## Target Variable

`Revenue = True` → session resulted in a purchase  
`Revenue = False` → session did NOT purchase → candidate for promotional intervention

The dataset is imbalanced (~84% no purchase, ~16% purchase). All models use
`class_weight='balanced'` and are evaluated on **ROC-AUC** rather than accuracy.

---

## Dataset

UCI ML Repository — [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  
12,330 sessions | 18 features | Binary classification
