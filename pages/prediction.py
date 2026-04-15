import streamlit as st
from ML_part import *

st.title("🧠 Prediction Panel")

df_raw = generate_hospital_data()
df, df_ml = preprocess_data(df_raw)
model, cols, _, _ = train_model(df_ml)

priority = st.selectbox("Priority", [1,2,3])
queue = st.slider("Queue", 1, 50, 10)
beds = st.slider("Beds", 0, 5, 2)

shift = st.selectbox("Shift", ["Morning","Evening","Night"])
dept = st.selectbox("Dept", ["ER","OPD","Ward"])

shift_map = {"Morning":1,"Evening":2,"Night":3}

patient = {
    "priority_level": priority,
    "queue_length": queue,
    "bed_availability": beds,
    "shift_encoded": shift_map[shift],
    f"dept_{dept}": 1
}

if st.button("Predict"):
    pred, _ = predict_patient(patient, model, cols)
    st.metric("Delay", f"{pred:.2f} mins")