"""FastAPI application for customer segmentation.

This service loads a pre‑trained clustering model and a scaler, then exposes a simple
JSON API to predict the customer cluster based on *annual_income* and *spending_score*.
The implementation follows best practices: explicit CORS handling, graceful error
reporting, and comprehensive OpenAPI documentation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts the customer cluster using a pre‑trained model and scaler.",
    version="1.0.0",
)

# Allow cross‑origin requests (use specific origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model & scaler loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
except Exception as exc:
    logger.exception("Failed to load model or scaler: %s", exc)
    raise RuntimeError("Model initialization failed") from exc

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class Customer(BaseModel):
    """Input data representing a single customer (height and weight)."""

    height: float = Field(..., description="Height of the customer (cm)")
    weight: float = Field(..., description="Weight of the customer (kg)")


class PredictionResponse(BaseModel):
    """Response model returned by the /predict endpoint."""

    cluster: int = Field(..., description="Predicted cluster identifier")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/",
    summary="Health check",
    response_description="Service status",
)
def health_check():
    """Simple health‑check endpoint used by monitoring tools."""
    return {"status": "ok", "message": "FastAPI is running"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict customer cluster",
    response_description="Cluster identifier for the provided data",
)
def predict_cluster(customer: Customer):
    """
    Predict the cluster for a given customer.

    The function:
    1. Converts the input to a DataFrame.
    2. Scales the data using the pre‑loaded scaler.
    3. Calls the appropriate model method (`predict` or `fit_predict`).

    Any unexpected errors are caught and turned into a 500 HTTP error with a
    helpful message.
    """
    try:
        # Build DataFrame with Height and Weight matching the trained model
        df = pd.DataFrame(
            [
                {
                    "Height": customer.height,
                    "Weight": customer.weight,
                }
            ]
        )

        # Scale the features
        scaled = scaler.transform(df)

        # ---------------------------------------------------------------
        # Some clustering models (e.g., AgglomerativeClustering) only expose
        # `fit_predict`. They cannot predict on new data directly.
        # ---------------------------------------------------------------
        if hasattr(model, "predict"):
            # Standard case: scikit‑learn estimators with a predict method
            cluster_id = int(model.predict(scaled)[0])
        else:
            # Fallback for AgglomerativeClustering (Hierarchical Clustering)
            # which does not support predict() on new data. We assign the
            # new data point to the nearest cluster centroid.
            import numpy as np
            centroid_0 = np.array([0.986928, 0.990507])  # Scaled centroid for Cluster 0
            centroid_1 = np.array([-0.986928, -0.990507]) # Scaled centroid for Cluster 1
            
            point = scaled[0]
            dist_0 = np.linalg.norm(point - centroid_0)
            dist_1 = np.linalg.norm(point - centroid_1)
            
            cluster_id = 0 if dist_0 < dist_1 else 1

        return PredictionResponse(cluster=cluster_id)

    except HTTPException:
        # Re‑raise HTTP errors unchanged
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")
