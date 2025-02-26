import streamlit as st
import pickle
import numpy as np

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

st.title("🔥 :green[Algerian Forest Fire Weather Index Prediction]")

st.write(":green[Enter the weather conditions to predict the Fire Weather Index (FWI).]")

temperature = st.number_input("🌡️ :red[Temperature (°C)]", min_value=-10.0, max_value=50.0, value=25.0,)
humidity = st.number_input("💧 :red[RH (Relative Humidity %)]", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("💨 :red[Wind Speed (km/h)]", min_value=0.0, max_value=100.0, value=10.0)
rain = st.number_input("🌧️ :red[Rainfall (mm)]", min_value=0.0, max_value=100.0, value=0.0)
ffmc = st.number_input("🔥 :red[FFMC (Fine Fuel Moisture Code)]", min_value=0.0, max_value=100.0, value=85.0)
dmc = st.number_input("🌿 :red[DMC (Duff Moisture Code)]", min_value=0.0, max_value=100.0, value=50.0)
isi = st.number_input("🔥 :red[ISI (Initial Spread Index)]", min_value=0.0, max_value=50.0, value=5.0)
region = st.selectbox("📍 :red[Region (0 : Bejaia , 1 : Sidi-Bel Abbes)]", [0, 1])

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1627162319041-ba11296b64c9?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
if st.button("Predict FWI"):
    try:
        
        input_features = np.array([[temperature, humidity, wind_speed, rain, ffmc, dmc, isi, region]])
        scaled_data = standard_scaler.transform(input_features)
        predicted_fwi = ridge_model.predict(scaled_data)[0]
        
        st.title(f"🔥 :green[Predicted Fire Weather Index (FWI)]: :green[({predicted_fwi:.2f})]")

    except Exception as e:
        st.error(f"Error: {str(e)}")
