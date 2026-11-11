
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Fraud Detection API")

# Configuration MLflow
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
)

# Charger le modèle champion au démarrage
model = mlflow.sklearn.load_model(
    "models:/fraud-detection-model@champion"
)

THRESHOLD = 0.82

FEATURES = [
    "TX_AMOUNT",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "CUSTOMER_AVG_AMOUNT",
    "CUSTOMER_TX_COUNT",
    "AMOUNT_TO_AVG_RATIO"
]


# Schéma des données reçues
class Transaction(BaseModel):
    TX_AMOUNT: float = Field(ge=0)
    TX_HOUR: int = Field(ge=0, le=23)
    TX_DAY_OF_WEEK: int = Field(ge=0, le=6)

    CUSTOMER_AVG_AMOUNT: float | None = Field(default=None, ge=0)
    CUSTOMER_TX_COUNT: int = Field(ge=0)
    AMOUNT_TO_AVG_RATIO: float | None = Field(default=None, ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    start = time.perf_counter()

    data = pd.DataFrame([transaction.model_dump()])[FEATURES]

    # Probabilité estimée de fraude
    probability = model.predict_proba(data)[0, 1]

    # Application du seuil choisi
    prediction = int(probability >= THRESHOLD)

    latency = time.perf_counter() - start

    print(
        f"Prediction latency: {latency:.4f} seconds",
        flush=True
    )

    return {
        "is_fraud": prediction,
        "fraud_probability": float(probability)
    }
