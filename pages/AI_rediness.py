import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ML_part import *

st.title("🤖 AI Readiness Audit System")

# ================================
# LOAD DATA
# ================================
df_raw = generate_hospital_data()
df, df_ml = preprocess_data(df_raw)
model, cols, mae, r2 = train_model(df_ml)

# ================================
# AI READINESS CHECK
# ================================
def check_ai_readiness(r2, mae, df):
    variance = df['process_debt_mins'].std()

    reasons = []

    if r2 < 0.75:
        reasons.append("Low model accuracy (R² < 0.75) → AI cannot reliably predict delays")

    if mae > 20:
        reasons.append("High prediction error (MAE > 20 mins) → Predictions are not precise")

    if variance > 40:
        reasons.append("High system variability → Hospital process is unstable and inconsistent")

    if len(reasons) == 0:
        return "READY", "🟢", ["System is stable, predictable, and AI can generalize well"]

    return "NOT READY", "🔴", reasons


status, emoji, reasons = check_ai_readiness(r2, mae, df)

# ================================
# KPI PANEL
# ================================
st.subheader("📊 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{r2:.2f}")
c2.metric("MAE", f"{mae:.2f} mins")
c3.metric("Variance", f"{df['process_debt_mins'].std():.2f}")

st.divider()

# ================================
# STATUS
# ================================
st.subheader("🚦 AI Readiness Status")
st.metric("System Status", f"{emoji} {status}")

# ================================
# REASONS (DETAILED)
# ================================
st.subheader("❗ Why System is NOT AI Ready?")

for r in reasons:
    st.error(r)

# ================================
# SHAP ANALYSIS
# ================================
st.subheader("🧬 Explainable AI Insights")

sample = df.sample(100)

X = pd.get_dummies(sample, columns=['department', 'activity'])

# Align columns
for col in cols:
    if col not in X.columns:
        X[col] = 0

X = X[cols]

explainer = shap.Explainer(model)
shap_values = explainer(X)

# Plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)

# ================================
# SHAP INTERPRETATION
# ================================
st.subheader("🔍 Key Drivers of Delay")

# Get top features
importance = np.abs(shap_values.values).mean(axis=0)
top_idx = np.argsort(importance)[::-1][:5]
top_features = [cols[i] for i in top_idx]

st.write("Top factors affecting delay:")
for f in top_features:
    st.write(f"• {f}")

# ================================
# ACTIONABLE RECOMMENDATIONS
# ================================
st.subheader("🛠️ How to Make System AI Ready?")

actions = []

if r2 < 0.75:
    actions.append("Improve model performance by adding more training data or better features")

if mae > 20:
    actions.append("Reduce prediction error by tuning model hyperparameters")

if df['process_debt_mins'].std() > 40:
    actions.append("Stabilize hospital workflow (reduce randomness in queues and bed allocation)")

# SHAP-based recommendations
if any("queue_length" in f for f in top_features):
    actions.append("Optimize queue management (major delay contributor)")

if any("bed_availability" in f for f in top_features):
    actions.append("Improve bed allocation system")

if any("shift_encoded" in f for f in top_features):
    actions.append("Balance staffing across shifts (especially Night shift)")

# Show actions
for a in actions:
    st.info(f"➡️ {a}")

# ================================
# FINAL SUMMARY
# ================================
st.divider()

st.subheader("📌 Final AI Audit Conclusion")

if status == "READY":
    st.success("System is AI Ready. Model is reliable and stable for deployment.")
else:
    st.error("System is NOT AI Ready. Requires process stabilization and model improvement.")