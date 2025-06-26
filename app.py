import streamlit as st
import pickle
import pandas as pd

# Load the trained pipeline/model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("🚗 Car Selling Price Prediction")
st.write("Fill in the details below to get an estimated selling price for your car!")

# Input fields
brand = st.text_input("Brand")
model_name = st.text_input("Model")
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
condition = st.selectbox("Condition", ['New', 'Like New', 'Excellent', 'Good', 'Fair'])
year = st.number_input("Year of Manufacture", 2000, 2025, 2015)
engine_size = st.number_input("Engine Size (L)", 0.5, 10.0, 2.0, step=0.1)
mileage = st.number_input("Mileage (kilometers)", 0, 500000, 50000)

# Prepare input data
input_df = pd.DataFrame([{
    'Brand': brand,
    'Model': model_name,
    'Year': year,
    'Engine Size': engine_size,
    'Fuel Type': fuel,
    'Transmission': transmission,
    'Mileage': mileage,
    'Condition': condition
}])

# Predict
if st.button("Predict Selling Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"The predicted selling price is: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error in Prediction: {e}")

