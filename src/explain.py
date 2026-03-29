import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Load saved model
# ---------------------------
with open("../models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

theta = model["theta"]
scaler = model["scaler"]
features = model["feature_names"]

print(f"0: {theta}, 1: {scaler}, 2: {features}")

# ---------------------------
# Define sigmoid function
# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------
# Define prediction function for SHAP
# ---------------------------
def model_predict(X_input):
    """
    X_input: pandas DataFrame with same columns as training features
    Returns: predicted probabilities
    """
    # ensure correct column order
    X_input = X_input[features]
    # scale features
    X_scaled = scaler.transform(X_input)
    # add intercept
    X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    # predict probabilities
    return sigmoid(X_scaled.dot(theta))

# ---------------------------
# Load data for explainability
# ---------------------------
# You can use original training data, or a sample of it
df_train = pd.read_csv("../data/processed_customer_churn.csv")
X_train = df_train.drop("Churn", axis=1)

# ---------------------------
# Create SHAP explainer
# ---------------------------
explainer = shap.Explainer(model_predict, X_train)

# compute shap values
shap_values = explainer(X_train)

# ---------------------------
# Summary plot
# ---------------------------
shap.summary_plot(shap_values, X_train)
plt.show()
