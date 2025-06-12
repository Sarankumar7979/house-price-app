import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 House Price Prediction App")

st.write("Enter the details below to predict the house price:")

# Inputs that match the training features
beds = st.number_input("Number of Bedrooms (Beds)", min_value=0, max_value=10, value=3)
baths = st.number_input("Number of Bathrooms (Baths)", min_value=0, max_value=10, value=2)
living_space = st.number_input("Living Space (sq ft)", min_value=100, max_value=10000, value=1500)
latitude = st.number_input("Latitude", format="%.6f", value=40.712776)
longitude = st.number_input("Longitude", format="%.6f", value=-74.005974)

# Create input DataFrame with correct feature names
input_data = pd.DataFrame([[beds, baths, living_space, latitude, longitude]],
                          columns=['Beds', 'Baths', 'Living Space', 'Latitude', 'Longitude'])

# Debug (optional)
# st.write("Input DataFrame:")
# st.write(input_data)

# Scale and predict
input_scaled = scaler.transform(input_data)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")
