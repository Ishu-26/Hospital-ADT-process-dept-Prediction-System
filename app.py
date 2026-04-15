import streamlit as st
from auth import *

st.set_page_config(page_title="NEURO-FLOW OS")

init_db()

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ================================
# NAVIGATION (LOGIN / SIGNUP)
# ================================
if not st.session_state.logged_in:

    st.title("🔐 NEURO-FLOW Authentication")

    menu = st.radio("Choose Option", ["Login", "Signup"])

    # ---------------- LOGIN ----------------
    if menu == "Login":
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # ---------------- SIGNUP ----------------
    elif menu == "Signup":
        st.subheader("Create Account")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")

        if st.button("Signup"):
            if new_pass != confirm_pass:
                st.warning("Passwords do not match")
            elif len(new_pass) < 5:
                st.warning("Password must be at least 5 characters")
            else:
                success = create_user(new_user, new_pass)
                if success:
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists")

# ================================
# MAIN APP
# ================================
else:
    st.sidebar.title("🏥 NEURO-FLOW")
    st.sidebar.success("Logged in")

    st.title("Welcome to NEURO-FLOW OS 🚀")
    st.write("Use sidebar to navigate.")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()