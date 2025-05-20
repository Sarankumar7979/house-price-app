# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 House Price Prediction App")

st.write("Enter the details below to predict the house price:")

# Select a few common features
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 500, 4000, 1500)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)

# Create input DataFrame
input_data = pd.DataFrame({
    'OverallQual': [overall_qual],
    'GrLivArea': [gr_liv_area],
    'GarageCars': [garage_cars],
    'TotalBsmtSF': [total_bsmt_sf]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")
