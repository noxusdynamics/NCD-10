import streamlit as st
import pandas as pd

st.set_page_config(page_title="Test", layout="centered")
st.title("Test 2")

with st.form("test_form"):
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("X value", value=10.0)
    with col2:
        y = st.number_input("Y value", value=20.0)
    submitted = st.form_submit_button("Submit")

if submitted:
    st.write("You entered:", x, y)