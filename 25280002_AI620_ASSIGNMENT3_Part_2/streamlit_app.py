# 25280002_AI620_ASSIGNMENT3_Part_2/streamlit_app.py
import requests
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PakWheels Price Predictor",
    layout="wide", 
)

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem ; }
    [data-testid= "stMetricValue"]{ font-size: 1.8rem !important; }
    div.stButton > button {height: 3em;margin-top: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("System Info")
    try:
        health_resp = requests.get("http://localhost:8000/", timeout=2)
        if health_resp.status_code == 200:
            st.success("API: Online")
        else:
            st.warning("API: Error")

    except:
        st.error("API: Offline")
    st.divider()


st.title("PakWheels Price Predictor")
st.markdown("Enter details to classify car value category.")



with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        make = st.selectbox("Make", ["Toyota", "Honda", "Suzuki", "KIA", "Hyundai", "Daihatsu", "Mitsubishi", "Nissan"])
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2018, step=1)
        body = st.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "Compact SUV", "Crossover", 
                                          "Mini Van", "Van", "Pickup", "Coupe", "Wagon"])

    with col2:
        engine = st.number_input("Engine (cc)", min_value=600, max_value=10000, value=1300, step=100)
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=45000, step=1000)
        city = st.text_input("City", value="Lahore", help="Enter city name as listed on PakWheels")

    with col3:
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "CNG", "Electric", "LPG"])
        
    predict_btn = st.button("ANALYZE MARKET CATEGORY", use_container_width=True)


if predict_btn:
    payload = {
        "year": int(year),
        "engine": float(engine),
        "mileage": float(mileage),
        "transmission": transmission,
        "fuel": fuel,
        "body": body,
        "city": city.strip(),
        "make": make
    }

    try:
        resp = requests.post("http://localhost:8000/predict", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            st.divider()
            
            res_col1, res_col2 = st.columns([1, 1])
            
            
            with res_col1:
                label = data["predicted_label"]
                icon = "" if label == "High Price" else ""
                st.subheader(f"{icon} Result: {label}")
                st.metric("Confidence", f"{data['confidence']:.1%}")


        else:
            st.error(f"Prediction Failed: {resp.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")

        