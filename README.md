📧 Spam Detection API
A machine learning API that classifies SMS messages as SPAM or HAM using Logistic
 Regression with TF-IDF vectorization.


🚀 Features

-REST API built with FastAPI
-Text preprocessing with NLTK (stopword removal, cleaning)
-TF-IDF vectorization for feature extraction
-Logistic Regression + SMOTE for handling class imbalance
-96% accuracy with only 10 false negatives on test set
-Interactive API docs at /docs

📁 Project Structure
    sentiment analysis/
    │
    ├── preprocessing.py      # Text cleaning and preprocessing
    ├── train.py              # Model training and evaluation
    ├── main.py               # FastAPI app
    ├── spam_model.pkl        # Saved trained model
    ├── vectorizer.pkl        # Saved TF-IDF vectorizer
    ├── cleaned_data.csv      # Preprocessed dataset
    └── README.md

⚙️ Setup & Installation
Copy and paste this entire block in your terminal to set up the project:

# 1. Clone the repo
git clone https://github.com/abidshoukat745-netizen/spam-detection-api.git
cd spam-detection-api

# 2. Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate

# 3. Install all dependencies
pip install fastapi uvicorn scikit-learn nltk pandas numpy imbalanced-learn joblib

# 4. Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# 5. Train the model
python train.py

# 6. Run the API
uvicorn main:app --reload

📡 API Endpoints
GET /
Health check — returns API status.
Response:
   {
  "message": "Spam Detection API is running!"
  }

--POST /predict
Classifies a message as SPAM or HAM.
Request Body:
    {
  "text": "Congratulations! You won a free iPhone, click here now!"
}

Response:
    {
  "label": "SPAM",
  "confidence": "73.82% spam probability"
}

📊 Model Performance
MetricScoreAccuracy : 96%
Precision : 0.97
Recall : 0.96
F1-Score : 0.96

Confusion Matrix:
[[952   7]
 [ 10 145]]

📬 Test the API
Visit http://127.0.0.1:8000/docs for the interactive Swagger UI after running the server.

or use curl:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"You won a free iPhone!\"}"

👤 Author
Abid Ali
AI/ML Engineer
