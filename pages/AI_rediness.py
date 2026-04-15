import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ML_part import *

st.title("🤖 AI Readiness")

df_raw = generate_hospital_data()
df, df_ml = preprocess_data(df_raw)
model, cols, mae, r2 = train_model(df_ml)

def check_ai_readiness(r2, mae, df):
    var = df['process_debt_mins'].std()
    if r2 > 0.75 and mae < 20 and var < 40:
        return "READY", "🟢"
    return "NOT READY", "🔴"

status, emoji = check_ai_readiness(r2, mae, df)

st.metric("Status", f"{emoji} {status}")

# SHAP
sample = df.sample(100)
X = pd.get_dummies(sample, columns=['department', 'activity'])

for col in cols:
    if col not in X.columns:
        X[col] = 0

X = X[cols]

explainer = shap.Explainer(model)
shap_values = explainer(X)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)