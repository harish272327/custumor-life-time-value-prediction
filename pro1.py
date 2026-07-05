import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model5.joblib")

st.set_page_config(page_title="Insurance Claim Prediction", page_icon="🚗")

st.title("🚗 Insurance Claim Prediction")

st.write("Enter the customer details below.")

# -----------------------------
# User Inputs
# -----------------------------
policy_id = st.number_input("Policy ID", value=1)

exposure = st.number_input("Exposure", min_value=0.0, value=1.0)

car_age = st.number_input("Car Age", min_value=0, value=5)

driver_age = st.number_input("Driver Age", min_value=18, value=35)

density = st.number_input("Density", min_value=0.0, value=50.0)

claim = st.selectbox("Is Claim Made?", [0, 1])

power = st.selectbox(
    "Vehicle Power",
    ['e','f','g','h','i','j','k','l','m','n','o']
)

brand = st.selectbox(
    "Brand",
    [
        "Japanese (except Nissan) or Korean",
        "Mercedes, Chrysler or BMW",
        "Opel, General Motors or Ford",
        "Renault, Nissan or Citroen",
        "Volkswagen, Audi, Skoda or Seat",
        "other"
    ]
)

gas = st.selectbox("Fuel Type", ["Regular", "Diesel"])

region = st.selectbox(
    "Region",
    [
        "Basse-Normandie",
        "Bretagne",
        "Centre",
        "Haute-Normandie",
        "Ile-de-France",
        "Limousin",
        "Nord-Pas-de-Calais",
        "Pays-de-la-Loire",
        "Poitou-Charentes"
    ]
)

# -----------------------------
# Feature Engineering
# -----------------------------
log_exposure = exposure

features = {
    "PolicyID": policy_id,
    "Exposure": exposure,
    "CarAge": car_age,
    "DriverAge": driver_age,
    "Density": density,
    "Log_Exposure": log_exposure,
    "Is_Claim_Made": claim,

    "Power_e":0,
    "Power_f":0,
    "Power_g":0,
    "Power_h":0,
    "Power_i":0,
    "Power_j":0,
    "Power_k":0,
    "Power_l":0,
    "Power_m":0,
    "Power_n":0,
    "Power_o":0,

    "Brand_Japanese (except Nissan) or Korean":0,
    "Brand_Mercedes, Chrysler or BMW":0,
    "Brand_Opel, General Motors or Ford":0,
    "Brand_Renault, Nissan or Citroen":0,
    "Brand_Volkswagen, Audi, Skoda or Seat":0,
    "Brand_other":0,

    "Gas_Regular":0,

    "Region_Basse-Normandie":0,
    "Region_Bretagne":0,
    "Region_Centre":0,
    "Region_Haute-Normandie":0,
    "Region_Ile-de-France":0,
    "Region_Limousin":0,
    "Region_Nord-Pas-de-Calais":0,
    "Region_Pays-de-la-Loire":0,
    "Region_Poitou-Charentes":0,

    "CarAge_Group_0-5":0,
    "CarAge_Group_6-10":0,
    "CarAge_Group_11-15":0,
    "CarAge_Group_16+":0,

    "Density_Category_Medium":0,
    "Density_Category_High":0,
    "Density_Category_Very High":0,

    "DriverAge_Group_25-34":0,
    "DriverAge_Group_35-44":0,
    "DriverAge_Group_45-54":0,
    "DriverAge_Group_55-64":0,
    "DriverAge_Group_65+":0
}

# Power
features[f"Power_{power}"] = 1

# Brand
features[f"Brand_{brand}"] = 1

# Fuel
if gas == "Regular":
    features["Gas_Regular"] = 1

# Region
features[f"Region_{region}"] = 1

# Car Age Group
if car_age <= 5:
    features["CarAge_Group_0-5"] = 1
elif car_age <= 10:
    features["CarAge_Group_6-10"] = 1
elif car_age <= 15:
    features["CarAge_Group_11-15"] = 1
else:
    features["CarAge_Group_16+"] = 1

# Density Category
if density < 50:
    pass
elif density < 100:
    features["Density_Category_Medium"] = 1
elif density < 200:
    features["Density_Category_High"] = 1
else:
    features["Density_Category_Very High"] = 1

# Driver Age Group
if 25 <= driver_age <= 34:
    features["DriverAge_Group_25-34"] = 1
elif 35 <= driver_age <= 44:
    features["DriverAge_Group_35-44"] = 1
elif 45 <= driver_age <= 54:
    features["DriverAge_Group_45-54"] = 1
elif 55 <= driver_age <= 64:
    features["DriverAge_Group_55-64"] = 1
elif driver_age >= 65:
    features["DriverAge_Group_65+"] = 1

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([features])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Claim Amount: {prediction:.2f}")

    # Updated Streamlit UI (regression-safe)

    
    
