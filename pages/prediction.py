import streamlit as st
from ML_part import *

st.title("🧠 Prediction Panel")

df = st.session_state.df
model = st.session_state.model
cols = st.session_state.cols
mae = st.session_state.mae
r2 = st.session_state.r2

if "df" not in st.session_state:
    st.warning("⚠️ Please upload dataset from Home page first")
    st.stop()

#inputs
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

#explanation function
def explain_prediction(patient):
    reasons = []

    if patient['queue_length'] > 20:
        reasons.append("High queue length is increasing delay")

    if patient['bed_availability'] <= 1:
        reasons.append("Low bed availability is causing waiting")

    if patient['shift_encoded'] == 3:
        reasons.append("Night shift has lower staff availability")

    if patient['priority_level'] == 1:
        reasons.append("Low priority patients are processed slower")

    return reasons

if st.button("🚀 Predict"):

    pred, _ = predict_patient(patient, model, cols)

    # MAIN OUTPUT
    st.metric("⏱️ Predicted Delay", f"{pred:.2f} mins")

#risk level:
    if pred > 80:
        risk = "HIGH"
        emoji = "🔴"
    elif pred > 60:
        risk = "MEDIUM"
        emoji = "🟡"
    else:
        risk = "LOW"
        emoji = "🟢"

    st.metric("🚦 Risk Level", f"{emoji} {risk}")

  
    st.subheader("🔍 Why this delay?")

    reasons = explain_prediction(patient)

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("System operating under normal conditions")

#suggested action
    st.subheader("🛠️ Suggested Action")

    if pred > 80:
        st.error("🚨 Critical: Add staff or reduce queue immediately")
    elif pred > 60:
        st.warning("⚠️ Monitor queue and bed allocation")
    else:
        st.success("✅ System is under control")