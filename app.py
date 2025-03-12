import streamlit as st
import joblib
import numpy as np

# Load trained model & scaler
model = joblib.load("water_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌊 Water Quality Prediction App")
st.write("Enter water quality parameters to check if water is **Safe or Unsafe**.")

# Input sliders for user input
pH = st.slider("pH Level", 0.0, 14.0, 7.0)
Hardness = st.slider("Hardness", 0.0, 500.0, 200.0)
Solids = st.slider("Solids", 0.0, 50000.0, 10000.0)
Chloramines = st.slider("Chloramines", 0.0, 15.0, 5.0)
Sulfate = st.slider("Sulfate", 0.0, 500.0, 250.0)
Conductivity = st.slider("Conductivity", 0.0, 1000.0, 500.0)
Organic_carbon = st.slider("Organic Carbon", 0.0, 30.0, 10.0)
Trihalomethanes = st.slider("Trihalomethanes", 0.0, 120.0, 50.0)
Turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0)

# Predict button
if st.button("Check Water Potability"):
    # Prepare input & normalize
    features = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    features_scaled = scaler.transform(features)
    
    # Predict & display
    prediction = model.predict(features_scaled)[0]
    result = "✅ Safe 🚰" if prediction == 1 else "❌ Unsafe ⚠️"
    st.success(f"**Prediction:** {result}")
