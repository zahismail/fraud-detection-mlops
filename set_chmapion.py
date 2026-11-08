import mlflow
import sys

mlflow.set_tracking_uri("http://127.0.0.1:5001")

version = sys.argv[1] if len(sys.argv) > 1 else "1"
client = mlflow.MlflowClient()

client.set_registered_model_alias(
    name="fraud-detection-model",
    alias="champion",
    version="1"
)

print(f"Champion alias assigned to model version {version}")