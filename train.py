from pathlib import Path
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.impute import SimpleImputer

# 1. Load prepared datasets
base_dir = Path(__file__).resolve().parent

train_df = pd.read_pickle(base_dir / "processed/train.pkl")
val_df = pd.read_pickle(base_dir / "processed/val.pkl")

features = ["TX_AMOUNT","TX_HOUR","TX_DAY_OF_WEEK","CUSTOMER_AVG_AMOUNT","CUSTOMER_TX_COUNT","AMOUNT_TO_AVG_RATIO"]

X_train = train_df[features]
y_train = train_df["TX_FRAUD"]

X_val = val_df[features]
y_val = val_df["TX_FRAUD"]

# 2. Configure MLflow
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
)
mlflow.set_experiment("fraud-detection")

# 3. Train and evaluate
with mlflow.start_run(run_name="logistic-regression-reproducible"):

    pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
  ])

    pipeline.fit(X_train, y_train)
#----------------------
   

#-------------------   
    threshold=0.82     
    probabilities = pipeline.predict_proba(X_val)[:,1]
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    # 4. Log experiment
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "class_weight": "balanced",
        "max_iter": 1000
    })

    mlflow.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # 5. Save model locally
    joblib.dump(pipeline, base_dir / "fraud_model.joblib")

    # 6. Save model in MLflow
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="fraud_model",
        skops_trusted_types=["numpy.dtype"]
    )

print("Training completed successfully!")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")