import streamlit as st
from ML_part import *

st.title("📊 Dashboard")

df = st.session_state.df
model = st.session_state.model
cols = st.session_state.cols
mae = st.session_state.mae
r2 = st.session_state.r2

if "df" not in st.session_state:
    st.warning("⚠️ Please upload dataset from Home page first")
    st.stop()



# KPI metrices
c1, c2, c3, c4 = st.columns(4)

c1.metric("R²", f"{r2:.2f}")
c2.metric("MAE", f"{mae:.2f}")
c3.metric("Avg Delay", f"{df['process_debt_mins'].mean():.1f}")
c4.metric("Variance", f"{df['process_debt_mins'].std():.1f}")

# system health summary
st.subheader("🧾 System Health Summary")

def interpret_system(r2, mae, df):
    avg_delay = df['process_debt_mins'].mean()
    variance = df['process_debt_mins'].std()

    insights = []

    # Accuracy
    if r2 > 0.75:
        insights.append("✅ AI predictions are reliable")
    else:
        insights.append("⚠️ AI predictions are not fully reliable")

    # Error
    if mae < 20:
        insights.append("✅ Prediction error is low (system is predictable)")
    else:
        insights.append("⚠️ High prediction error (system is inconsistent)")

    # Delay
    if avg_delay < 60:
        insights.append("🟢 Patient wait time is under control")
    else:
        insights.append("🔴 Patient wait time is high")

    # Stability
    if variance < 40:
        insights.append("🟢 System is stable")
    else:
        insights.append("🔴 System behavior is unpredictable")

    return insights

summary = interpret_system(r2, mae, df)

for s in summary:
    st.write(s)

st.divider()

# department analysis
st.subheader("🏥 Department-wise Delay Analysis")

dept_avg = df.groupby("department")["process_debt_mins"].mean()

st.bar_chart(dept_avg)

# graph explanation
worst_dept = dept_avg.idxmax()
best_dept = dept_avg.idxmin()

st.info(f"""
📌 **What this means:**

- 🚨 Highest delay: **{worst_dept}** → Needs attention  
- ✅ Best performing: **{best_dept}**  
- This helps identify where hospital resources are under pressure  
""")

# quick recommandation
st.subheader("🛠️ Quick Recommendation")

if worst_dept == "ER":
    st.warning("Increase ER staff or add fast-track system")
elif worst_dept == "Ward":
    st.warning("Improve bed allocation system")
elif worst_dept == "Lab":
    st.warning("Optimize lab processing time")
else:
    st.success("No major bottlenecks detected")