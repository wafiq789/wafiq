import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Prediksi Rating Produk")

# Input fitur
price = st.number_input("Harga Produk", value=0.0)
total_sales = st.number_input("Total Penjualan", value=0)

# Prediksi
if st.button("Prediksi"):
    features = np.array([[price, total_sales]])
    features_scaled = scaler.transform(features)  
    prediction = model.predict(features_scaled)[0]
    st.success(f"Prediksi Rating: {prediction:.2f}")
