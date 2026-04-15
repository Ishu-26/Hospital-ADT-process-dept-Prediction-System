import streamlit as st
from ML_part import *

st.title("📈 Analytics")

df_raw = generate_hospital_data()
df, _ = preprocess_data(df_raw)

fig = generate_visuals(df)
st.pyplot(fig)