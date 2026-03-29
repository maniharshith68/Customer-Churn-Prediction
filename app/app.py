import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import requests

st.set_page_config(layout="wide")
st.title("Customer Churn Prediction Dashboard")

# ------------------------
# Load Model & Metrics
# ------------------------
MODEL_PATH = "../models/churn_model.pkl"
METRICS_PATH = "../models/churn_metrics.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(METRICS_PATH, "rb") as f:
    metrics = pickle.load(f)

theta = model["theta"]
scaler = model["scaler"]
features = model["feature_names"]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ------------------------
# API URL (Optional)
# ------------------------
API_URL = os.getenv("API_URL")  # set this in your environment if you want to use API mode
use_api = API_URL is not None

# ------------------------
# Show Model Metrics
# ------------------------
st.subheader("Model Performance Metrics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    st.metric("Cross-validation Accuracy", f"{metrics['cv_accuracy']*100:.2f}%")
    st.metric("AUC Score", f"{metrics['auc']:.3f}")
    st.metric("Cross-validation AUC", f"{metrics['cv_auc']:.3f}")

with col2:
    # ROC Curve
    fig, ax = plt.subplots()
    ax.plot(metrics['fpr'], metrics['tpr'], label=f"AUC = {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Prediction Form
# ------------------------
st.subheader("Predict Customer Churn")
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

predict_btn = st.button("Predict")

if predict_btn:
    df_input = pd.DataFrame([input_data])

    if use_api:
        # ------------------------
        # Call API mode
        # ------------------------
        try:
            response = requests.post(API_URL, json=input_data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                churn_prob = result['churn_probability']
                pred_class = result['prediction']
            else:
                st.error(f"API error {response.status_code}: {response.text}")
                st.stop()
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.stop()
    else:
        # ------------------------
        # Local model mode
        # ------------------------
        X_scaled = scaler.transform(df_input)
        X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        churn_prob = float(sigmoid(X_scaled.dot(theta))[0])
        pred_class = int(churn_prob > 0.5)

    # ------------------------
    # Show Prediction
    # ------------------------
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {churn_prob:.3f}")
    st.write(f"Prediction Class (0 = No, 1 = Yes): {pred_class}")

    # ------------------------
    # SHAP Global Explanation (using sample data)
    # ------------------------
    st.subheader("SHAP Global Feature Importance")
    sample_df = pd.DataFrame(np.random.rand(50, len(features)), columns=features)  # sample for background
    def model_predict(X_input):
        X_scaled = scaler.transform(X_input)
        X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return sigmoid(X_scaled.dot(theta))
    explainer_global = shap.Explainer(model_predict, sample_df)
    shap_values_global = explainer_global(sample_df)

    fig_global, ax_global = plt.subplots()
    shap.summary_plot(shap_values_global, sample_df, show=False)
    st.pyplot(fig_global)

    # ------------------------
    # SHAP Waterfall for individual prediction
    # ------------------------
    st.subheader("SHAP Waterfall for This Customer")
    explainer_local = shap.Explainer(model_predict, df_input)
    shap_values_local = explainer_local(df_input)

    fig_local, ax_local = plt.subplots()
    shap.waterfall_plot(shap_values_local[0], show=False)
    st.pyplot(fig_local)
