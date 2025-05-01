# prompt: buatkan streamlit sesuai ini https://www.kaggle.com/datasets/nehalbirla/motorcycle-dataset/data

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('finalized_model.sav', 'rb') as f:
    model = pickle.load(f)

# Load the scaler (if used during training)
# scaler = StandardScaler() # Assuming you used StandardScaler
# # ... load scaler parameters if needed

# Function to preprocess user input
def preprocess_input(year, km_driven, owner, brand):
    input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'owner': [owner]
    })

    # One-hot encode the brand (assuming you did this during training)
    brands = ['Honda', 'Yamaha', 'Bajaj', 'KTM', 'Royal Enfield', 'Suzuki', 'TVS'] # add all possible brands here
    for b in brands:
        input_data[b] = 0
    input_data[brand] = 1

    # Scale the features
    # scaled_data = scaler.transform(input_data)  # Uncomment if you used scaling
    # return scaled_data
    return input_data


# Streamlit app
st.title("Motorcycle Price Prediction")

# Get user input
year = st.number_input("Year", min_value=2001, max_value=2023, value=2018)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=25000)
owner = st.selectbox("Owner", [0, 1, 2, 3])
brands = ['Honda', 'Yamaha', 'Bajaj', 'KTM', 'Royal Enfield', 'Suzuki', 'TVS', 'Other']
brand = st.selectbox("Brand", brands)

# Preprocess user input
input_data = preprocess_input(year, km_driven, owner, brand)

if st.button("Predict Price"):
    try:
        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction
        st.success(f"Predicted Price: {prediction[0]:.2f}")

    except ValueError as e:
        st.error(f"Error during prediction: {e}")

