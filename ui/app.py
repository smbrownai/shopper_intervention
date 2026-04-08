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

API_URL = os.getenv("API_URL", "https://shopper-intervention.onrender.com")
# Safe default — overwritten by API if available
threshold_data = {"mode": "lower", "lower": 0.30, "upper": 0.70}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Shopper Intervention System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200, r.json(), None
    except Exception as e:
        return False, {}, str(e)


def call_predict(payload: dict):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot reach API. Is `uvicorn api.main:app --port 8000` running?"
    except Exception as e:
        return None, str(e)


def call_predict_batch(sessions: list[dict], use_challenger: bool = False):
    try:
        r = requests.post(
            f"{API_URL}/predict-batch",
            json={"sessions": sessions, "use_challenger": use_challenger},
            timeout=180
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot reach API."
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛒 Shopping Intervention")
    st.caption("Predict sessions for special promo")
    st.caption("BANA 7075 Group 2 Winter 2026")
    st.divider()

    st.markdown("**Intervention Threshold**")
    st.caption("The probability cutoff used to flag sessions for outreach. This is a business decision applied to model output, not a model parameter.")
    
    # Load current config from API
    try:
        current = requests.get(f"{API_URL}/threshold", timeout=3).json()
        current_mode = current.get("mode", "lower")
        current_lower = current.get("lower", 0.30)
        current_upper = current.get("upper", 0.70)
    except Exception:
        current_mode = "lower"
        current_lower = 0.30
        current_upper = 0.70
    
    threshold_mode = st.radio(
        "Mode",
        ["Single threshold", "Range"],
        index=0 if current_mode == "lower" else 1,
        key="threshold_mode"
    )
    
    if threshold_mode == "Single threshold":
        lower = st.slider(
            "Intervene if P(purchase) below",
            0.0, 1.0, current_lower, step=0.01,
            key="threshold_lower"
        )
        upper = lower
        mode = "lower"
    else:
        lower, upper = st.slider(
            "Intervene if P(purchase) within range",
            0.0, 1.0, (current_lower, current_upper), step=0.01,
            key="threshold_range"
        )
        mode = "range"
    
    if st.button("Apply Threshold", key="apply_threshold"):
        try:
            r = requests.post(f"{API_URL}/threshold", json={
                "mode": mode,
                "lower": float(lower),
                "upper": float(upper),
            })
            st.success("✅ Threshold updated")
        except Exception:
            st.error("Could not reach API")        
    st.divider()

    st.markdown("**Debug**")
    st.sidebar.caption(f"API: {API_URL}")

    healthy, health_data, health_error = api_health()
    if healthy:
        try:
            threshold_data = requests.get(f"{API_URL}/threshold", timeout=3).json()
            threshold_data.setdefault("mode", "lower")
            threshold_data.setdefault("lower", 0.30)
            threshold_data.setdefault("upper", 0.70)
        except Exception:
            threshold_data = {"mode": "lower", "lower": 0.30, "upper": 0.70}

        st.success(f"API online ✅")
        #st.caption(f"Model: **{health_data.get('model', '?')}**")
        #st.caption(f"ROC-AUC: **{health_data.get('roc_auc', '?')}**")
        #if threshold_data.get("mode", "lower") == "range":
        #    st.caption(f"Threshold: **{threshold_data.get('lower', 0.30):.0%} – {threshold_data.get('upper', 0.70):.0%}** (range)")
        #else:
        #    st.caption(f"Threshold: **{threshold_data.get('lower', 0.30):.0%}** (below)")

        with st.expander("📦 Model Metadata", expanded=False):
            try:
                meta = requests.get(f"{API_URL}/model-info", timeout=5).json()
                st.markdown("**Champion**")
                st.json(meta.get("champion", meta))
                if meta.get("challenger"):
                    st.markdown("**Challenger**")
                    st.json(meta["challenger"])
            except Exception:
                st.warning("Could not fetch model metadata.")

        with st.expander("🔁 Retrain Status", expanded=False):
            try:
                st.json(requests.get(f"{API_URL}/retrain-status", timeout=3).json())
            except Exception:
                st.warning("Could not fetch retrain status.")

        with st.expander("⚙️ Threshold Config", expanded=False):
            st.json(threshold_data)

        with st.expander("🔗 Links", expanded=False):
            st.markdown("[📂 Training Data (GitHub)](https://github.com/smbrownai/shopper_intervention/blob/main/data/online_shoppers_intention.csv)")
            st.markdown("[📊 MLflow Experiments (DagHub)](https://dagshub.com/smbrownai/shopper_intervention.mlflow)")
            st.markdown("[🐙 GitHub Repository](https://github.com/smbrownai/shopper_intervention)")
            st.markdown("[🗄️ DagHub Repository](https://dagshub.com/smbrownai/shopper_intervention)")

    else:
        threshold_data = {"mode": "lower", "lower": 0.30, "upper": 0.70}
        st.error("API offline ❌")
        st.caption(f"Error: `{health_error}`")
        with st.expander("🔗 Links", expanded=False):
            st.markdown("[📂 Training Data (GitHub)](https://github.com/smbrownai/shopper_intervention/blob/main/data/online_shoppers_intention.csv)")
            st.markdown("[📊 MLflow Experiments (DagHub)](https://dagshub.com/smbrownai/shopper_intervention.mlflow)")
            st.markdown("[🐙 GitHub Repository](https://github.com/smbrownai/shopper_intervention)")
            st.markdown("[🗄️ DagHub Repository](https://dagshub.com/smbrownai/shopper_intervention)")
        st.caption("Run: `uvicorn api.main:app --reload --port 8000`")

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

    # Check if challenger is available
    has_challenger = bool(model_meta.get("challenger")) if "model_meta" in dir() else False
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        has_challenger = bool(info.get("challenger"))
        champion_name = info.get("model_name", "Champion")
        challenger_name = info.get("challenger", {}).get("model_name", "Challenger")
    except Exception:
        has_challenger = False
        champion_name = "Champion"
        challenger_name = "Challenger"
    
    model_choice = st.radio(
        "Model",
        [f"Champion ({champion_name})", f"Challenger ({challenger_name})"] if has_challenger else [f"Champion ({champion_name})"],
        horizontal=True,
        key="model_choice_single",
        help="Select which model to score with. Challenger is only available after a training run produces one."
    )
    use_challenger = "Challenger" in model_choice

    with st.form("session_form"):
        st.subheader("Page Interactions")
        c1, c2, c3 = st.columns(3)
        admin = c1.number_input("Administrative Pages", min_value=0, value=0)
        admin_dur = c2.number_input("Admin Duration (s)", min_value=0.0, value=0.0)

        informational = c1.number_input("Informational Pages", min_value=0, value=0)
        informational_dur = c2.number_input("Info Duration (s)", min_value=0.0, value=0.0)

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
            "Informational": int(informational),
            "Informational_Duration": float(informational_dur),
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
            "use_challenger": use_challenger,
        }

        start = time.time()
        result, err = call_predict(payload)
        elapsed = time.time() - start

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

                st.metric("Inference Time", f"{elapsed*1000:.1f} ms")
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
                            {"range": [0, threshold_data.get("lower", 0.30) * 100], "color": "#fadbd8"},
                            {"range": [threshold_data.get("lower", 0.30) * 100, 100], "color": "#d5f5e3"},
                        ],
                        "threshold": {
                            "line": {"color": "orange", "width": 4},
                            "thickness": 0.75,
                            "value": threshold_data.get("lower", 0.30) * 100,
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
    st.caption("Upload a CSV of sessions to score them all at once and identify intervention candidates. Recommended file size is about 1,500 rows.")

    REQUIRED_COLS = [
        "Administrative","Administrative_Duration","Informational","Informational_Duration",
        "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues",
        "SpecialDay","Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend",
    ]

    st.info(f"CSV must contain columns: `{', '.join(REQUIRED_COLS)}`")

    # Model selector
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        has_challenger = bool(info.get("challenger"))
        champion_name = info.get("model_name", "Champion")
        challenger_name = info.get("challenger", {}).get("model_name", "Challenger")
    except Exception:
        has_challenger = False
        champion_name = "Champion"
        challenger_name = "Challenger"

    model_choice_batch = st.radio(
        "Model",
        [f"Champion ({champion_name})", f"Challenger ({challenger_name})"] if has_challenger else [f"Champion ({champion_name})"],
        horizontal=True,
        key="model_choice_batch",
    )
    use_challenger_batch = "Challenger" in model_choice_batch

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
            start = time.time()
            batch_result, err = call_predict_batch(sessions, use_challenger=use_challenger_batch)
            elapsed = time.time() - start

        if err:
            st.error(err)
        else:
            results_list = batch_result["results"]
            st.divider()

            # KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("Sessions Scored", batch_result["total_sessions"])
            k2.metric("Intervention Candidates", batch_result["intervention_count"])
            k3.metric("Intervention Rate", f"{batch_result['intervention_rate']*100:.1f}%")

            k4, k5, k6 = st.columns(3)
            avg_prob = np.mean([r["purchase_probability"] for r in results_list])
            k4.metric("Avg Purchase Prob", f"{avg_prob*100:.1f}%")
            k5.metric("Avg Inference Time", f"{elapsed/len(sessions)*1000:.2f} ms/session")
            k6.metric("Total Batch Time", f"{elapsed:.2f} s")

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
    st.header("Champion Model Performance")

    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.success(f"Best model: **{info.get('model_name', '—')}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("ROC-AUC (Test)", f"{info.get('roc_auc', 0):.4f}")
        col2.metric("Model Type", info.get("model_name", "—"))
        if threshold_data["mode"] == "range":
            col3.metric("Intervention Threshold", f"{threshold_data['lower']:.0%} – {threshold_data['upper']:.0%}")
        else:
            col3.metric("Intervention Threshold", f"{threshold_data['lower']:.0%}")
        st.caption(f"MLflow run ID: `{info.get('run_id', '—')}`")
        if info.get("challenger"):
            st.divider()
            st.subheader("Challenger Model")
            ch = info["challenger"]
            c1, c2 = st.columns(2)
            c1.metric("Challenger Model", ch.get("model_name", "—"))
            c2.metric("Challenger ROC-AUC", f"{ch.get('roc_auc', 0):.4f}")
    except Exception:
        st.warning("Could not load model info from API.")

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

    #st.subheader("Run MLflow UI")
    #st.code("mlflow ui --port 5050", language="bash")
    #st.caption("Then open http://localhost:5050 to compare all experiment runs side by side.")


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
        st.warning("⚠️ Retraining replaces the champion model if the new run scores higher. Note: Each training run updates two sources — a local metadata file (best_model_meta.json) that powers this dashboard, and the MLflow Model Registry on DagHub where champion and challenger versions are tagged with aliases for lineage tracking. These are kept in sync automatically by the training pipeline, but manual changes in the MLflow Registry will not be reflected here until the next training run.")

    overrides = {}
    
    st.divider()
    st.subheader("Preprocessing Options")
    st.caption("Raw session data is preprocessed through a shared pipeline used by both training and inference. Numeric features — such as page counts, durations, bounce and exit rates, and page values — are either median-imputed or mean-imputed to handle missing values and then standardized with zero mean and unit variance. Categorical features — including month, visitor type, operating system, browser, region, traffic type, and weekend flag — are mode-imputed and one-hot encoded. Technical and Engagement Rate features may be excluded from model training. Any unknown categories seen at inference time are ignored rather than causing an error. The same preprocessor is fit during training and applied consistently at prediction time, ensuring no data leakage.")
    
    imputer_strategy = st.radio(
        "Numeric Imputation Strategy",
        ["median", "mean"],
        horizontal=True,
        help="Median is more robust to outliers. Mean is sensitive to skewed features like PageValues."
    )
    
    st.markdown("**Exclude Feature Groups**")
    st.caption("Remove entire feature groups from training to evaluate their impact on model performance.")
    
    col_a, col_b = st.columns(2)
    exclude_technical = col_a.checkbox(
        "Exclude Technical Features",
        help="OperatingSystems, Browser, Region, TrafficType"
    )
    exclude_engagement = col_b.checkbox(
        "Exclude Engagement Rates",
        help="BounceRates, ExitRates, PageValues"
    )
    
    excluded_features = []
    if exclude_technical:
        excluded_features += ["OperatingSystems", "Browser", "Region", "TrafficType"]
    if exclude_engagement:
        excluded_features += ["BounceRates", "ExitRates", "PageValues"]
    
    overrides["_preprocessor"] = {
        "numeric_imputer_strategy": imputer_strategy,
        "excluded_features": excluded_features,
    }

    st.divider()
    st.subheader("Hyperparameter Overrides")
    st.caption("Adjust key parameters per model. Leave as-is to use defaults.")

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

    with st.expander("XGBoost", expanded=False):
        gb_estimators = st.slider("N Estimators", 50, 500, 200, step=50, key="gb_n")
        gb_lr = st.slider("Learning Rate", 0.01, 0.3, 0.05, step=0.01, key="gb_lr")
        gb_depth = st.slider("Max Depth", 2, 10, 4, key="gb_depth")
        gb_subsample = st.slider("Subsample", 0.5, 1.0, 0.8, step=0.05, key="gb_sub")
        xgb_scale = st.slider("XGB Scale Pos Weight", 1, 20, 5, key="xgb_scale")
        overrides["XGBoost"] = {"n_estimators": gb_estimators, "learning_rate": gb_lr, "max_depth": gb_depth, "subsample": gb_subsample, "scale_pos_weight": xgb_scale}

    st.divider()
    if st.button("🚀 Start Retraining", type="primary"):
        response = requests.post(f"{API_URL}/retrain", json={"overrides": overrides})
        result = response.json()

        if result["status"] == "already_running":
            st.info("⏳ A training run is already in progress.")
        else:
            status_placeholder = st.empty()
            status_placeholder.info("Training started — polling every 5 seconds...")
            with st.spinner("Training in progress..."):
                while True:
                    time.sleep(5)
                    status = requests.get(f"{API_URL}/retrain-status").json()
                    if not status["running"]:
                        break
            status_placeholder.empty()

            if status["last_result"] == "success":
                st.success(
                    f"✅ Retraining complete!\n\n"
                    f"**Model:** {status['model']}  \n"
                    f"**ROC-AUC:** {status['roc_auc']:.4f}  \n"
                    f"**Version:** {status['version']}"
                )
                #st.balloons()
            else:
                st.error(f"❌ Training failed: {status['last_result']}")
