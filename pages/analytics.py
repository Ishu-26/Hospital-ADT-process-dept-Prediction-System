import streamlit as st
import plotly.express as px
import pandas as pd
from ML_part import *
import math

st.title("📈 Hospital Analytics")

# load data

df = st.session_state.df
model = st.session_state.model
cols = st.session_state.cols
mae = st.session_state.mae
r2 = st.session_state.r2

if "df" not in st.session_state:
    st.warning("⚠️ Please upload dataset from Home page first")
    st.stop()

# filters
st.sidebar.header("🔍 Filter Data")

dept_filter = st.sidebar.multiselect(
    "Select Department",
    options=df["department"].unique(),
    default=df["department"].unique()
)

shift_filter = st.sidebar.multiselect(
    "Select Shift",
    options=df["shift_time"].unique(),
    default=df["shift_time"].unique()
)

priority_filter = st.sidebar.multiselect(
    "Select Priority",
    options=df["priority_level"].unique(),
    default=df["priority_level"].unique()
)

# Apply filters
filtered_df = df[
    (df["department"].isin(dept_filter)) &
    (df["shift_time"].isin(shift_filter)) &
    (df["priority_level"].isin(priority_filter))
]

st.success(f"Showing {len(filtered_df)} records after filtering")

# KPI comparison
st.subheader("📊 KPI Comparison")

c1, c2 = st.columns(2)

c1.metric("Filtered Avg Delay", f"{filtered_df['process_debt_mins'].mean():.1f}")
c2.metric("Overall Avg Delay", f"{df['process_debt_mins'].mean():.1f}")

st.divider()

# shift impact
st.subheader("🕒 Shift Impact on Delay")

st.bar_chart(filtered_df.groupby("shift_time")["process_debt_mins"].mean())

st.info("""
📌 **Explanation:**  
Night shift usually has higher delays due to reduced staff availability.  
If Night > Morning significantly → staffing issue.
""")

# queue effect
st.subheader("👥 Queue vs Delay")



fig = px.scatter(
    filtered_df,
    x="queue_length",
    y="process_debt_mins",
    color="priority_level",   
    title="Queue Length vs Delay"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
📌 **Explanation:**  
As queue length increases, delay increases linearly.  
This indicates system congestion.
""")

# priority analysis
st.subheader("🚑 Priority Handling Efficiency")

st.bar_chart(filtered_df.groupby("priority_level")["process_debt_mins"].mean())

st.info("""
📌 **Explanation:**  
Priority 3 (critical patients) should have lowest delay.  
If not → triage system is failing.
""")

# bed avaibility
st.subheader("🛏️ Bed Availability Impact")

bed_df = filtered_df[filtered_df["bed_availability"] >= 0]

st.line_chart(bed_df.groupby("bed_availability")["process_debt_mins"].mean())

st.info("""
📌 **Explanation:**  
Low bed availability (0–1) leads to very high delays.  
Indicates resource shortage.
""")

# bottleneck detection
st.subheader("🚨 Bottleneck Detection")

dept_delay = filtered_df.groupby("department")["process_debt_mins"].mean()

worst_dept = dept_delay.idxmax()
worst_value = dept_delay.max()

st.warning(f"⚠️ Major Bottleneck: {worst_dept} ({worst_value:.1f} mins avg delay)")

# actionable insights
st.subheader("🛠️ Recommended Actions")

actions = []

# Rule-based insights
if worst_dept == "ER":
    actions.append("Increase ER staffing or create fast-track lanes")

if filtered_df["queue_length"].mean() > 20:
    actions.append("Reduce queue by adding more service counters")

if filtered_df["bed_availability"].mean() < 2:
    actions.append("Improve bed management system")

night_delay = filtered_df[filtered_df["shift_time"]=="Night"]["process_debt_mins"].mean()
morning_delay = filtered_df[filtered_df["shift_time"]=="Morning"]["process_debt_mins"].mean()

if night_delay > morning_delay:
    actions.append("Increase Night shift staffing")

# Show actions
if actions:
    for act in actions:
        st.success(f"➡️ {act}")
else:
    st.info("System operating efficiently. No major issues detected.")

# summary
st.divider()

st.subheader("📌 Summary for Admin")

st.write(f"""
- Highest delay department: **{worst_dept}**
- Average processing delay: **{filtered_df['process_debt_mins'].mean():.1f} mins**
- Average queue load: **around {math.ceil(filtered_df['queue_length'].mean()):.1f} patients in waiting**
- Average resource availability: **around {math.ceil(filtered_df['bed_availability'].mean()):.1f} beds free**
""")