import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("artifacts/model/best_model.joblib")
preprocessor = joblib.load("artifacts/preprocessor.joblib")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether the customer will churn."
)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        value=12
    )

    usage_frequency = st.number_input(
        "Usage Frequency",
        min_value=0,
        value=10
    )

    support_calls = st.number_input(
        "Support Calls",
        min_value=0,
        value=2
    )

    payment_delay = st.number_input(
        "Payment Delay",
        min_value=0,
        value=0
    )

with col2:

    total_spend = st.number_input(
        "Total Spend",
        min_value=0.0,
        value=500.0
    )

    last_interaction = st.number_input(
        "Last Interaction",
        min_value=0,
        value=5
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    contract_length = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )

if st.button("Predict Churn"):

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    transformed_data = preprocessor.transform(input_df)

    prediction = model.predict(transformed_data)[0]

    probability = model.predict_proba(
        transformed_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"Customer will Churn\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"Customer will Stay\n\nProbability of Churn: {probability:.2%}"
        )