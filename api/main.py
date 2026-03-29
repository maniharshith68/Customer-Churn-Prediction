from fastapi import FastAPI
import pickle
import os
import numpy as np
import pandas as pd

app = FastAPI()

# -----------------------
# Load model at startup
# -----------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

theta = model["theta"]
scaler = model["scaler"]
features = model["feature_names"]

# -----------------------
# Sigmoid
# -----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------
# Health check
# -----------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict(data: dict):

    # convert input to dataframe
    df = pd.DataFrame([data])

    # ensure correct column order
    df = df[features]

    # preprocess
    X = scaler.transform(df)
    X = np.c_[np.ones((X.shape[0], 1)), X]

    # predict
    prob = sigmoid(X.dot(theta))[0]
    pred = int(prob > 0.5)

    return {
        "churn_probability": float(prob),
        "prediction": pred
    }
