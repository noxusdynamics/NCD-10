import streamlit as st
import joblib

st.title("Model Load Test")

try:
    bundle = joblib.load("crop_model.joblib")
    st.success("Model loaded successfully!")
    st.write("Features:", bundle["features"])
    st.write("Classes:", bundle["classes"])
except Exception as e:
    st.error(f"Error loading model: {e}")