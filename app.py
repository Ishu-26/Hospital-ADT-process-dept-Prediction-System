import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from ML_part import *

st.set_page_config(page_title="NEURO-FLOW OS", layout="wide")

# ================================
# LOAD SYSTEM
# ================================
@st.cache_resource
def load_system():
    df_raw = generate_hospital_data()
    df, df_ml = preprocess_data(df_raw)
    model, feature_cols, mae, r2 = train_model(df_ml)
    return df, model, feature_cols, mae, r2

df, model, feature_cols, mae, r2 = load_system()

# ================================
# HEADER
# ================================
st.title("🏥 NEURO-FLOW: AI Readiness Command Center")
st.markdown("### Is Your Hospital System AI-Ready?")

st.divider()

# ================================
# AI READINESS LOGIC
# ================================
def check_ai_readiness(r2, mae, df):
    variance = df['process_debt_mins'].std()

    if r2 > 0.75 and mae < 20 and variance < 40:
        return "READY", "🟢", "System is stable and predictable"
    else:
        return "NOT READY", "🔴", "High randomness or low model reliability"

status, emoji, reason = check_ai_readiness(r2, mae, df)

# ================================
# KPI PANEL
# ================================
st.subheader("📊 System KPIs")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Accuracy (R²)", f"{r2:.2f}")
c2.metric("Prediction Error (MAE)", f"{mae:.2f}")
c3.metric("Avg Delay", f"{df['process_debt_mins'].mean():.1f}")
c4.metric("Variance", f"{df['process_debt_mins'].std():.1f}")

st.divider()

# ================================
# AI READINESS STATUS
# ================================
st.subheader("🤖 AI Readiness Status")

st.metric("System Status", f"{emoji} {status}")
st.info(f"📌 Reason: {reason}")

# ================================
# SHAP EXPLAINABILITY
# ================================
st.subheader("🧬 Explainable AI (WHY this decision?)")

sample = df.sample(100)

X_sample = pd.get_dummies(sample, columns=['department', 'activity'], prefix=['dept', 'act'])

# Align columns
for col in feature_cols:
    if col not in X_sample.columns:
        X_sample[col] = 0

X_sample = X_sample[feature_cols]

explainer = shap.Explainer(model)
shap_values = explainer(X_sample)

# Plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_sample, show=False)
st.pyplot(fig)

# ================================
# INTERPRETATION TEXT
# ================================
top_features = np.abs(shap_values.values).mean(axis=0)
top_idx = np.argsort(top_features)[::-1][:3]

important_features = [feature_cols[i] for i in top_idx]

if status == "READY":
    st.success(f"🟢 AI is reliable. Key driving factors: {important_features}")
else:
    st.error(f"🔴 AI NOT reliable. System affected by unstable factors: {important_features}")

st.divider()

# ================================
# PREDICTION PANEL
# ================================
st.subheader("🧠 Predict Patient Risk")

col1, col2 = st.columns(2)

with col1:
    priority = st.selectbox("Priority Level", [1,2,3])
    queue = st.slider("Queue Length", 1, 50, 10)
    beds = st.slider("Bed Availability", 0, 5, 2)

with col2:
    shift = st.selectbox("Shift", ["Morning", "Evening", "Night"])
    dept = st.selectbox("Department", ["ER", "OPD", "Ward"])

shift_map = {"Morning":1, "Evening":2, "Night":3}

patient = {
    "priority_level": priority,
    "queue_length": queue,
    "bed_availability": beds,
    "shift_encoded": shift_map[shift],
    f"dept_{dept}": 1
}

def get_risk(pred):
    if pred < 40:
        return "LOW", "🟢"
    elif pred < 70:
        return "MEDIUM", "🟡"
    else:
        return "HIGH", "🔴"

if st.button("🚀 Predict"):

    pred, _ = predict_patient(patient, model, feature_cols)
    risk, emoji = get_risk(pred)

    st.metric("Predicted Delay", f"{pred:.2f} mins")
    st.metric("Risk Level", f"{emoji} {risk}")

    if risk == "HIGH":
        st.error("🔴 CRITICAL ALERT")
    elif risk == "MEDIUM":
        st.warning("🟡 Warning")
    else:
        st.success("🟢 Normal")

st.divider()

# ================================
# BOTTLENECK ANALYSIS
# ================================
st.subheader("📊 Bottleneck Analysis")

dept_avg = df.groupby("department")["process_debt_mins"].mean()
st.bar_chart(dept_avg)

st.warning(f"Highest Bottleneck: {dept_avg.idxmax()}")

st.divider()

# ================================
# VISUALS
# ================================
st.subheader("📈 System Insights")

fig2 = generate_visuals(df)
st.pyplot(fig2)