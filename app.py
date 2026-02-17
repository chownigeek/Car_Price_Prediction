import streamlit as st
import joblib
import numpy as np
import pandas as pd

def preprocess_user_input(input_df, model_columns):
    """
    Preprocess user input the same way as training data:
    - One-hot encode categorical features
    - Reindex to match training columns
    """
    # One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Replace spaces in column names with underscores (same as training)
    input_encoded.columns = input_encoded.columns.str.replace(" ", "_")

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    return input_encoded

# Load trained model and column order
model = joblib.load("best_car_price_model.pkl")
columns = joblib.load("model_columns.pkl")   # Save this in your notebook when training

st.title("🚗 Car Price Prediction App")

# User inputs
brand = st.selectbox("Brand", ["Audi", "BMW", "Toyota", "Hyundai", "Honda"])  
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])  
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])  
condition = st.selectbox("Condition", ["New", "Used"])  

engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=8.0, step=0.1)
mileage = st.number_input("Mileage (in km)", min_value=0, max_value=300000, step=1000)
car_age = st.slider("Car Age (years)", 0, 30, 5)


# Convert inputs into DataFrame
# User input dictionary
input_dict = {
    "Brand": brand,
    "Fuel_Type": fuel_type,
    "Transmission": transmission,
    "Condition": condition,
    "Engine_Size": engine_size,
    "Mileage": mileage,
    "Car_Age": car_age
}

# Create DataFrame
input_df = pd.DataFrame([input_dict])


# Predict price
if st.button("Predict Price"):
    input_encoded = preprocess_user_input(input_df, columns)
    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
