import streamlit as st
import pandas as pd
import joblib

st.title("Crop Recommendation System")

bundle = joblib.load("crop_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

N = st.number_input("Nitrogen (N)", value=60.0)
P = st.number_input("Phosphorus (P)", value=50.0)
K = st.number_input("Potassium (K)", value=42.0)
temperature = st.number_input("Temperature (°C)", value=28.0)
humidity = st.number_input("Humidity (%)", value=87.0)
ph = st.number_input("Soil pH", value=6.2)
rainfall = st.number_input("Rainfall (mm)", value=160.0)

if st.button("Recommend Crop"):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Recommended Crop: {prediction.capitalize()}")