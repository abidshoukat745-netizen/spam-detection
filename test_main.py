from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Spam Detection API is running!"}

def test_predict_spam():
    response = client.post("/predict", json={"text": "Win free money now click here"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert data["label"] in ["SPAM", "HAM"]

def test_predict_ham():
    response = client.post("/predict", json={"text": "See you at the meeting tomorrow"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in ["SPAM", "HAM"]

def test_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200