from unittest.mock import patch
from fastapi.testclient import TestClient

# Avoid contacting MLflow during module import.
with patch("mlflow.sklearn.load_model") as mock_load:
    mock_load.return_value.predict.return_value = [0]
    from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}