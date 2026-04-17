
import streamlit as st
import pandas as pd
from ML_part import *
from auth import *

st.set_page_config(page_title="NEURO-FLOW OS")

init_db()

# ================================
# SESSION STATE
# ================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ================================
# FUNCTIONS (KEEP OUTSIDE)
# ================================
def load_data():
    uploaded_file = st.file_uploader(
        "📂 Upload Hospital Dataset to predict system process dept",
        type=["csv"]
    )

    st.info("Dataset must contain: priority_level, queue_length, bed_availability, shift_time, department, start_time, end_time,Activity,process_debt_mins")

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

        required_cols = [
    "case_id",
    "priority_level",
    "queue_length",
    "bed_availability",
    "shift_time",
    "department",
    "process_debt_mins",
    "start_time",
    "end_time",
    "Activity",
    "previous_activity"
]

        if not all(col in df_raw.columns for col in required_cols):
            st.error("❌ Invalid dataset format")
            st.stop()

        st.session_state.df_raw = df_raw
        st.success("✅ Dataset uploaded")

    elif "df_raw" not in st.session_state:
        df_raw = pd.read_csv("adt_ml_ready_dataset.csv")
        st.session_state.df_raw = df_raw
        st.info("Using Default Dataset")

    return st.session_state.df_raw


@st.cache_resource
def build_model(df_raw):
    df, df_ml = preprocess_data(df_raw)
    model, cols, mae, r2 = train_model(df_ml)
    return df, model, cols, mae, r2


# ================================
# LOGIN / SIGNUP
# ================================
if not st.session_state.logged_in:

    st.title("🔐 NEURO-FLOW Authentication")
    menu = st.radio("Choose Option", ["Login", "Signup"])

    if menu == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")

        if st.button("Signup"):
            if new_pass != confirm_pass:
                st.warning("Passwords do not match")
            elif len(new_pass) < 5:
                st.warning("Password too short")
            else:
                if create_user(new_user, new_pass):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists")


# ================================
# MAIN APP
# ================================
else:
    st.sidebar.title("🏥 NEURO-FLOW")
    st.sidebar.success("Logged in")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("Welcome to NEURO-FLOW OS 🚀")
# ================================
# INFO CARDS (SYSTEM OVERVIEW)
# ================================

    st.markdown("""
    <style>
    .card {
    background-color: #111111;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #333;
    transition: 0.3s;
    }
    .card:hover {
    border: 1px solid #00FFAA;
    transform: scale(1.02);
    }
    .card-title {
    font-size: 20px;
    font-weight: bold;
    color: #00FFAA;
}
.card-text {
    font-size: 14px;
    color: #CCCCCC;
}
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown("""
    <div class="card">
        <div class="card-title">🚑 Problem</div>
        <div class="card-text">
        Hospitals face unpredictable delays due to queue overload, poor bed allocation, and inefficient workflows.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div class="card">
        <div class="card-title">🤖 Our Solution</div>
        <div class="card-text">
        NEURO-FLOW uses AI to predict process delays and identify bottlenecks in real-time hospital operations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
    <div class="card">
        <div class="card-title">📊 What It Analyzes</div>
        <div class="card-text">
        Queue length, bed availability, patient priority, department flow, and activity transitions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
    <div class="card">
        <div class="card-title">🚀 Outcome</div>
        <div class="card-text">
        Detect system inefficiencies, improve hospital flow, and ensure AI readiness for smart decision-making.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ✅ LOAD DATA ONCE
    df_raw = load_data()

    # ✅ BUILD MODEL
    df, model, cols, mae, r2 = build_model(df_raw)

    # ✅ STORE FOR OTHER PAGES
    st.session_state.df = df
    st.session_state.model = model
    st.session_state.cols = cols
    st.session_state.mae = mae
    st.session_state.r2 = r2