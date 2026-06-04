from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocessing import preprocess_text

app = FastAPI(title="Spam Detection API")

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class MessageRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: str

@app.get("/")
def root():
    return {"message": "Spam Detection API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    cleaned = preprocess_text(request.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][1]
    
    return {
        "label": "SPAM" if prediction == 1 else "HAM",
        "confidence": f"{round(float(probability) * 100, 2)}% spam probability"
    }