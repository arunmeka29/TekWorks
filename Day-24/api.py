from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

# Load scaler
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home():
    return {"message": "Hierarchical Clustering API Running"}


@app.post("/predict")
def predict(height: float, weight: float):

    # Input data
    data = np.array([[height, weight]])

    # Scaling
    scaled_data = scaler.transform(data)

    # Dummy cluster logic
    if scaled_data[0][0] > 0:
        cluster = 1
    else:
        cluster = 0

    return {"cluster": cluster}