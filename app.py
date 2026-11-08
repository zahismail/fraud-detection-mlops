from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import mlflow
import mlflow.sklearn

import os
import time

app = FastAPI(title="Fraud Detection API")

# Charger le modèle une seule fois au démarrage
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
model = mlflow.sklearn.load_model("models:/fraud-detection-model@champion")


# Schéma des données reçues
class Transaction(BaseModel):
    TX_AMOUNT: float = Field(ge=0)
    TX_HOUR: int = Field(ge=0, le=23)
    TX_DAY_OF_WEEK: int = Field(ge=0, le=6)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    start = time.perf_counter()

    features = ["TX_AMOUNT", "TX_HOUR", "TX_DAY_OF_WEEK"]
    data = pd.DataFrame([transaction.model_dump()])[features]

    prediction = model.predict(data)[0]

    latency = time.perf_counter() - start

    print(f"Prediction latency: {latency:.4f} seconds", flush=True)

    return {"is_fraud": int(prediction)}