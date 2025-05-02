import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore

# Load your pre-trained model and data (replace with your actual loading logic)
# ... (Your data loading and model training code from the previous example) ...

# Dummy data and model (replace with your actual data and model)
df = pd.DataFrame({
    'year': [2018, 2019, 2020, 2021],
    'km_driven': [10000, 20000, 30000, 40000],
    'selling_price': [50000, 60000, 70000, 80000],
    'owner_Second Owner': [0,1,0,0],
    'seller_type_Individual': [1,0,1,0],
    'brand_Royal Enfield': [0,0,0,1]

})

X = df.drop('selling_price', axis=1)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)

# Streamlit app
st.title("Used Motorcycle Price Prediction")

# Input features
year = st.number_input("Year", min_value=2000, max_value=2024, value=2020)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=10000)
owner = st.selectbox("Owner", ["First Owner", "Second Owner"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
brand = st.selectbox("Brand", ["Royal Enfield", "Other"]) # Add other brands


# Convert categorical inputs to numerical using your pre-trained model's encoding
# (adjust based on your actual model's expected input)
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'owner_Second Owner': [1 if owner == "Second Owner" else 0],
    'seller_type_Individual': [1 if seller_type == "Individual" else 0],
    'brand_Royal Enfield': [1 if brand == "Royal Enfield" else 0]
})

if st.button("Predict Price"):
    prediction = lr.predict(input_data)[0]
    st.write(f"Predicted Selling Price: {prediction:.2f}")
