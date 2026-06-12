import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("artifacts/best_model.joblib")

st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California Housing Price Prediction")

st.markdown(
    "Enter housing details below to predict the median house value."
)

# Input Fields

med_inc = st.number_input(
    "Median Income",
    min_value=0.0,
    value=3.5
)

house_age = st.number_input(
    "House Age",
    min_value=1.0,
    value=25.0
)

ave_rooms = st.number_input(
    "Average Rooms",
    min_value=0.0,
    value=5.0
)

ave_bedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.0
)

population = st.number_input(
    "Population",
    min_value=1,
    value=1500
)

ave_occup = st.number_input(
    "Average Occupancy",
    min_value=0.0,
    value=3.0
)

latitude = st.number_input(
    "Latitude",
    value=34.05
)

longitude = st.number_input(
    "Longitude",
    value=-118.24
)

# Predict Button

if st.button("Predict House Price"):

    features = np.array([
        [
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude
        ]
    ])

    prediction = model.predict(features)[0]

    st.success(
        f"Predicted Median House Value: ${prediction * 100000:,.2f}"
    )