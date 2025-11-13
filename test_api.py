import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_prediction():
    image_path = "ml/images/test1.jpeg"
    with open(image_path, "rb") as img:
        files = {"file": ("test1.jpeg", img, "image/jpeg")}

        response = client.post("/predict_emotion", files=files)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["emotion"], str)
    assert isinstance(data["score"], float)
