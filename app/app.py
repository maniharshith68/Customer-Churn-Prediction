import streamlit as st
import requests
import pandas as pd
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("Customer Churn Prediction")

# -----------------------
# Load model locally (for SHAP)
# -----------------------
with open("../models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

theta = model["theta"]
scaler = model["scaler"]
features = model["feature_names"]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------
# User input form
# -----------------------
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# -----------------------
# Call API
# -----------------------
if st.button("Predict"):

    # ⚠️ Replace with your Render API URL later
    API_URL = "https://churn-api-l13a.onrender.com/predict"

    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Prediction")
        st.write(f"Churn Probability: {result['churn_probability']:.3f}")
        st.write(f"Prediction: {result['prediction']}")

        # -----------------------
        # SHAP Explanation
        # -----------------------
        df_input = pd.DataFrame([input_data])

        def model_predict(X_input):
            X_scaled = scaler.transform(X_input)
            X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
            return sigmoid(X_scaled.dot(theta))

        explainer = shap.Explainer(model_predict, df_input)
        shap_values = explainer(df_input)

        st.subheader("SHAP Explanation")

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, show=False)
        st.pyplot(fig)

    else:
        try:
            response = requests.post(API_URL, json=input_data, timeout=60)

            st.write("Status Code:", response.status_code)
            st.write("Response:", response.text)

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction successful!")
                st.write(result)
            else:
                st.error("API error")

        except Exception as e:
            st.error(f"Error: {e}")
