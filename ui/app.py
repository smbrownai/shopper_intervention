"""
ui/app.py — Streamlit dashboard for Online Shopper Intervention System.

Tabs:
  1. 📊 Dataset Explorer  — EDA charts on the training data
  2. 🎯 Score a Session   — Manual feature entry → live API call → intervention decision
  3. 📋 Batch Scoring     — Upload CSV → batch API call → downloadable results
  4. 📈 Model Performance — Cached metrics / model metadata

Run with:
    streamlit run ui/app.py
"""

import sys
import json
import io
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
META_PATH = ROOT / "models" / "best_model_meta.json"

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_BASE = "https://shopper-intervention.onrender.com"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Shopper Intervention System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.caption(f"API: {API_URL}")
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df["Weekend"] = df["Weekend"].astype(str)
    df["Purchased"] = df["Revenue"].map({True: "Purchase", False: "No Purchase"})
    return df


def api_health():
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


def call_predict(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot reach API. Is `uvicorn api.main:app --port 8000` running?"
    except Exception as e:
        return None, str(e)


def call_predict_batch(sessions: list[dict]):
    try:
        r = requests.post(f"{API_BASE}/predict-batch", json={"sessions": sessions}, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot reach API. Is `uvicorn api.main:app --port 8000` running?"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛒 Shopper Intervention")
    st.caption("Predict & intervene before customers leave")
    st.divider()

    healthy, health_data = api_health()
    if healthy:
        st.success(f"API online ✅")
        st.caption(f"Model: **{health_data.get('model', '?')}**")
        st.caption(f"ROC-AUC: **{health_data.get('roc_auc', '?')}**")
        threshold = health_data.get("intervention_threshold", 0.30)
        st.caption(f"Threshold: **{threshold}**")
    else:
        st.error("API offline ❌")
        st.caption("Run: `uvicorn api.main:app --reload --port 8000`")

    st.divider()
    st.markdown("**Navigation**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Explorer",
    "🎯 Score a Session",
    "📋 Batch Scoring",
    "📈 Model Performance",
    "🔁 Retrain Model"
])

# ===========================================================================
# TAB 1 — Dataset Explorer
# ===========================================================================

with tab1:
    st.header("Dataset Explorer")
    df = load_data()

    if df is None:
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", f"{len(df):,}")
    col2.metric("Purchases", f"{df['Revenue'].sum():,}", f"{df['Revenue'].mean()*100:.1f}%")
    col3.metric("No Purchase", f"{(~df['Revenue']).sum():,}", f"{(~df['Revenue']).mean()*100:.1f}%")
    col4.metric("Intervention Candidates", f"{(~df['Revenue']).sum():,}", "sessions to target")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        # Purchase rate by month
        month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_df = (
            df.groupby("Month")["Revenue"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "Purchases", "count": "Sessions"})
            .reset_index()
        )
        month_df["Purchase Rate"] = month_df["Purchases"] / month_df["Sessions"]
        month_df["Month"] = pd.Categorical(month_df["Month"], categories=month_order, ordered=True)
        month_df = month_df.sort_values("Month")

        fig = px.bar(
            month_df, x="Month", y="Purchase Rate",
            color="Purchase Rate",
            color_continuous_scale="RdYlGn",
            title="Purchase Rate by Month",
            labels={"Purchase Rate": "Purchase Rate"},
            text_auto=".1%",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Revenue by visitor type
        vt_df = (
            df.groupby(["VisitorType", "Purchased"])
            .size()
            .reset_index(name="Count")
        )
        fig2 = px.bar(
            vt_df, x="VisitorType", y="Count", color="Purchased",
            barmode="group",
            color_discrete_map={"Purchase": "#2ecc71", "No Purchase": "#e74c3c"},
            title="Sessions by Visitor Type",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Page values distribution
        fig3 = px.histogram(
            df, x="PageValues", color="Purchased",
            nbins=50,
            log_y=True,
            color_discrete_map={"Purchase": "#2ecc71", "No Purchase": "#e74c3c"},
            title="Page Values Distribution (log scale)",
            barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Exit vs Bounce rate scatter
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig4 = px.scatter(
            sample, x="BounceRates", y="ExitRates",
            color="Purchased",
            color_discrete_map={"Purchase": "#2ecc71", "No Purchase": "#e74c3c"},
            title="Bounce Rate vs Exit Rate",
            opacity=0.5,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Weekend effect
    wk_df = df.groupby(["Weekend", "Purchased"]).size().reset_index(name="Count")
    wk_df["Weekend"] = wk_df["Weekend"].map({"True": "Weekend", "False": "Weekday"})
    fig5 = px.bar(
        wk_df, x="Weekend", y="Count", color="Purchased",
        barmode="group",
        color_discrete_map={"Purchase": "#2ecc71", "No Purchase": "#e74c3c"},
        title="Weekend vs Weekday Sessions",
    )
    st.plotly_chart(fig5, use_container_width=True)


# ===========================================================================
# TAB 2 — Score a Session
# ===========================================================================

with tab2:
    st.header("Score a Single Session")
    st.caption("Enter session features and get a real-time purchase prediction + intervention decision.")

    with st.form("session_form"):
        st.subheader("Page Interactions")
        c1, c2, c3 = st.columns(3)
        admin = c1.number_input("Administrative Pages", min_value=0, value=0)
        admin_dur = c2.number_input("Admin Duration (s)", min_value=0.0, value=0.0)

        info = c1.number_input("Informational Pages", min_value=0, value=0)
        info_dur = c2.number_input("Info Duration (s)", min_value=0.0, value=0.0)

        prod = c1.number_input("Product-Related Pages", min_value=0, value=10)
        prod_dur = c2.number_input("Product Duration (s)", min_value=0.0, value=300.0)

        st.subheader("Engagement Rates")
        r1, r2, r3 = st.columns(3)
        bounce = r1.slider("Bounce Rate", 0.0, 1.0, 0.02, 0.01)
        exit_r = r2.slider("Exit Rate", 0.0, 1.0, 0.04, 0.01)
        page_val = r3.number_input("Page Value", min_value=0.0, value=0.0)

        st.subheader("Session Context")
        s1, s2, s3, s4 = st.columns(4)
        month = s1.selectbox("Month", ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"], index=8)
        visitor = s2.selectbox("Visitor Type", ["Returning_Visitor","New_Visitor","Other"])
        weekend = s3.checkbox("Weekend", value=False)
        special = s4.slider("Special Day", 0.0, 1.0, 0.0, 0.2)

        st.subheader("Technical")
        t1, t2, t3, t4 = st.columns(4)
        os_ = t1.number_input("Operating System", min_value=1, max_value=8, value=2)
        browser = t2.number_input("Browser", min_value=1, max_value=13, value=2)
        region = t3.number_input("Region", min_value=1, max_value=9, value=1)
        traffic = t4.number_input("Traffic Type", min_value=1, max_value=20, value=2)

        submitted = st.form_submit_button("🔮 Predict & Check Intervention", type="primary")

    if submitted:
        payload = {
            "Administrative": int(admin),
            "Administrative_Duration": float(admin_dur),
            "Informational": int(info),
            "Informational_Duration": float(info_dur),
            "ProductRelated": int(prod),
            "ProductRelated_Duration": float(prod_dur),
            "BounceRates": float(bounce),
            "ExitRates": float(exit_r),
            "PageValues": float(page_val),
            "SpecialDay": float(special),
            "Month": month,
            "OperatingSystems": int(os_),
            "Browser": int(browser),
            "Region": int(region),
            "TrafficType": int(traffic),
            "VisitorType": visitor,
            "Weekend": weekend,
        }

        result, err = call_predict(payload)

        if err:
            st.error(err)
        else:
            st.divider()
            c_left, c_right = st.columns([1, 2])

            with c_left:
                prob_purchase = result["purchase_probability"]
                prob_no = result["no_purchase_probability"]
                intervene = result["intervene"]

                if intervene:
                    st.error("## 🚨 INTERVENE")
                    st.markdown("**Offer a promotional incentive!**")
                    st.markdown(f"Purchase probability is only **{prob_purchase*100:.1f}%** — well below the {result['intervention_threshold']*100:.0f}% threshold.")
                else:
                    st.success("## ✅ No Intervention Needed")
                    st.markdown(f"Purchase probability: **{prob_purchase*100:.1f}%**")

                st.metric("Purchase Probability", f"{prob_purchase*100:.1f}%")
                st.metric("No-Purchase Probability", f"{prob_no*100:.1f}%")
                st.metric("Confidence", result["confidence"])
                st.metric("Model", result["model_name"])

            with c_right:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_purchase * 100,
                    title={"text": "Purchase Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ecc71" if not intervene else "#e74c3c"},
                        "steps": [
                            {"range": [0, result["intervention_threshold"] * 100], "color": "#fadbd8"},
                            {"range": [result["intervention_threshold"] * 100, 100], "color": "#d5f5e3"},
                        ],
                        "threshold": {
                            "line": {"color": "orange", "width": 4},
                            "thickness": 0.75,
                            "value": result["intervention_threshold"] * 100,
                        },
                    },
                    number={"suffix": "%"},
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# TAB 3 — Batch Scoring
# ===========================================================================

with tab3:
    st.header("Batch Session Scoring")
    st.caption("Upload a CSV of sessions to score them all at once and identify intervention candidates.")

    REQUIRED_COLS = [
        "Administrative","Administrative_Duration","Informational","Informational_Duration",
        "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues",
        "SpecialDay","Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend",
    ]

    st.info(f"CSV must contain columns: `{', '.join(REQUIRED_COLS)}`")

    use_sample = st.checkbox("Use a sample from the training dataset (first 50 rows)", value=True)

    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload session CSV", type=["csv"])

    if st.button("▶️ Run Batch Scoring", type="primary"):
        if use_sample:
            df = load_data()
            batch_df = df[REQUIRED_COLS].head(50).copy()
        elif uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLS if c not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()
            batch_df = batch_df[REQUIRED_COLS]
        else:
            st.warning("Please upload a CSV or check 'Use sample'.")
            st.stop()

        # Convert to JSON list
        sessions = []
        for _, row in batch_df.iterrows():
            d = row.to_dict()
            d["Weekend"] = str(d["Weekend"]) in ("True", "true", "1")
            for col in ["Administrative","Informational","ProductRelated","OperatingSystems","Browser","Region","TrafficType"]:
                d[col] = int(d[col])
            sessions.append(d)

        with st.spinner(f"Scoring {len(sessions)} sessions..."):
            batch_result, err = call_predict_batch(sessions)

        if err:
            st.error(err)
        else:
            results_list = batch_result["results"]
            st.divider()

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Sessions Scored", batch_result["total_sessions"])
            k2.metric("Intervention Candidates", batch_result["intervention_count"])
            k3.metric("Intervention Rate", f"{batch_result['intervention_rate']*100:.1f}%")
            avg_prob = np.mean([r["purchase_probability"] for r in results_list])
            k4.metric("Avg Purchase Prob", f"{avg_prob*100:.1f}%")

            # Results table
            results_df = pd.DataFrame([{
                "Session": i + 1,
                "Purchase Prob": f"{r['purchase_probability']*100:.1f}%",
                "No-Purchase Prob": f"{r['no_purchase_probability']*100:.1f}%",
                "Prediction": "Purchase" if r["prediction"] == 1 else "No Purchase",
                "Intervene": "🚨 YES" if r["intervene"] else "✅ No",
                "Confidence": r["confidence"],
            } for i, r in enumerate(results_list)])

            st.dataframe(
                results_df.style.apply(
                    lambda col: ["background-color: #fadbd8" if v == "🚨 YES" else "" for v in col],
                    subset=["Intervene"],
                ),
                use_container_width=True,
            )

            # Download
            prob_col = [r["purchase_probability"] for r in results_list]
            batch_df = batch_df.copy().reset_index(drop=True)
            batch_df["purchase_probability"] = prob_col
            batch_df["intervene"] = [r["intervene"] for r in results_list]
            batch_df["prediction"] = ["Purchase" if r["prediction"] == 1 else "No Purchase" for r in results_list]
            csv_out = batch_df.to_csv(index=False)

            st.download_button(
                "⬇️ Download Results CSV",
                csv_out,
                file_name="scored_sessions.csv",
                mime="text/csv",
            )


# ===========================================================================
# TAB 4 — Model Performance
# ===========================================================================

with tab4:
    st.header("Model Performance")

    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        st.success(f"Best model: **{meta['model_name']}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("ROC-AUC (Test)", f"{meta['roc_auc']:.4f}")
        col2.metric("Model Type", meta["model_name"])
        col3.metric("Intervention Threshold", f"{meta['intervention_threshold']*100:.0f}%")
        st.caption(f"MLflow run ID: `{meta['run_id']}`")
    else:
        st.warning("No model metadata found. Run `python scripts/train.py` first.")

    st.divider()
    st.subheader("How the Intervention Works")
    st.markdown("""
    1. **Every browser session** is scored by the model in real time (via the FastAPI endpoint).
    2. The model outputs **P(purchase)** — a probability between 0 and 1.
    3. If **P(purchase) < threshold** (default 30%), the session is flagged for **intervention**.
    4. The intervention system shows the user a **promotional incentive** (coupon, free shipping, etc.)
       to convert the session into a purchase.
    5. The threshold can be tuned: a **lower threshold** is more conservative (fewer interventions),
       a **higher threshold** catches more at-risk sessions (but may waste promotions on buyers who would have purchased anyway).
    """)

    st.subheader("Class Imbalance Note")
    st.markdown("""
    The dataset is **imbalanced** (~84% No Purchase, ~16% Purchase). All models use
    `class_weight='balanced'` to compensate. We evaluate on **ROC-AUC** rather than
    raw accuracy, which would be misleading on imbalanced data.
    """)

    st.subheader("Run MLflow UI")
    st.code("mlflow ui --port 5050", language="bash")
    st.caption("Then open http://localhost:5050 to compare all experiment runs side by side.")


# ===========================================================================
# TAB 5 — Retrain Model
# ===========================================================================

with tab5:
    st.header("🔁 Retrain Model")
    st.caption("Adjust hyperparameters and kick off a new training run.")

    # Current model info
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            info = requests.get(f"{API_URL}/model-info").json()
            st.metric("Current Model", info.get("model_name", "—"))
            st.metric("ROC-AUC", f"{info.get('roc_auc', 0):.4f}")
            st.metric("Version", info.get("version", "—"))
        except Exception:
            st.caption("⚠️ Could not reach API")

    with col2:
        st.warning("⚠️ Retraining replaces the champion model if the new run scores higher.")

    st.divider()
    st.subheader("Hyperparameter Overrides")
    st.caption("Adjust key parameters per model. Leave as-is to use defaults.")

    overrides = {}

    with st.expander("Logistic Regression", expanded=False):
        lr_c = st.slider("C (regularization)", 0.001, 10.0, 1.0, step=0.01, key="lr_c")
        lr_solver = st.selectbox("Solver", ["lbfgs", "saga"], key="lr_solver")
        overrides["LR_baseline"] = {"C": lr_c, "solver": lr_solver}
        overrides["LR_high_regularization"] = {"C": lr_c * 0.01, "solver": lr_solver}

    with st.expander("Decision Tree", expanded=False):
        dt_depth = st.slider("Max Depth", 2, 20, 8, key="dt_depth")
        dt_criterion = st.selectbox("Criterion", ["gini", "entropy"], key="dt_criterion")
        dt_min_samples = st.slider("Min Samples Leaf", 1, 50, 10, key="dt_min")
        overrides["DT_shallow"] = {"max_depth": max(2, dt_depth - 4)}
        overrides["DT_medium"] = {"max_depth": dt_depth, "min_samples_leaf": dt_min_samples}
        overrides["DT_entropy"] = {"max_depth": dt_depth, "criterion": dt_criterion}

    with st.expander("Random Forest", expanded=False):
        rf_estimators = st.slider("N Estimators", 50, 500, 200, step=50, key="rf_n")
        rf_depth = st.slider("Max Depth", 4, 30, 12, key="rf_depth")
        rf_features = st.selectbox("Max Features", ["sqrt", "log2"], key="rf_features")
        overrides["RF_baseline"] = {"n_estimators": rf_estimators, "max_depth": rf_depth}
        overrides["RF_log2_features"] = {"n_estimators": rf_estimators, "max_depth": rf_depth, "max_features": "log2"}
        overrides["RF_deep"] = {"n_estimators": rf_estimators + 100, "max_depth": None}

    with st.expander("Gradient Boosting & XGBoost", expanded=False):
        gb_estimators = st.slider("N Estimators", 50, 500, 200, step=50, key="gb_n")
        gb_lr = st.slider("Learning Rate", 0.01, 0.3, 0.05, step=0.01, key="gb_lr")
        gb_depth = st.slider("Max Depth", 2, 10, 4, key="gb_depth")
        gb_subsample = st.slider("Subsample", 0.5, 1.0, 0.8, step=0.05, key="gb_sub")
        xgb_scale = st.slider("XGB Scale Pos Weight", 1, 20, 5, key="xgb_scale")
        overrides["GradientBoosting"] = {"n_estimators": gb_estimators, "learning_rate": gb_lr, "max_depth": gb_depth, "subsample": gb_subsample}
        overrides["XGBoost"] = {"n_estimators": gb_estimators, "learning_rate": gb_lr, "max_depth": gb_depth, "subsample": gb_subsample, "scale_pos_weight": xgb_scale}

    st.divider()
    if st.button("🚀 Start Retraining", type="primary"):
        response = requests.post(f"{API_URL}/retrain", json={"overrides": overrides})
        result = response.json()

        if result["status"] == "already_running":
            st.info("⏳ A training run is already in progress.")
        else:
            st.info("Training started — polling every 5 seconds...")
            with st.spinner("Training in progress..."):
                while True:
                    time.sleep(5)
                    status = requests.get(f"{API_URL}/retrain-status").json()
                    if not status["running"]:
                        break

            if status["last_result"] == "success":
                st.success(
                    f"✅ Retraining complete!\n\n"
                    f"**Model:** {status['model']}  \n"
                    f"**ROC-AUC:** {status['roc_auc']:.4f}  \n"
                    f"**Version:** {status['version']}"
                )
                st.balloons()
            else:
                st.error(f"❌ Training failed: {status['last_result']}")
