import streamlit as st
from ML_part import *

st.title("📊 Dashboard")

df_raw = generate_hospital_data()
df, df_ml = preprocess_data(df_raw)
model, cols, mae, r2 = train_model(df_ml)

c1, c2, c3, c4 = st.columns(4)

c1.metric("R²", f"{r2:.2f}")
c2.metric("MAE", f"{mae:.2f}")
c3.metric("Avg Delay", f"{df['process_debt_mins'].mean():.1f}")
c4.metric("Variance", f"{df['process_debt_mins'].std():.1f}")

st.bar_chart(df.groupby("department")["process_debt_mins"].mean())