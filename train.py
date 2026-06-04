import pandas as pd
from preprocessing import preprocess_text

# Load the cleaned data
df = pd.read_csv(r'C:\Users\Mehar\Desktop\sentiment analysis\cleaned_data.csv')


# Drop rows with missing values
df = df.dropna(subset=['text'])
X = df['text']
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# text vectorization using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Handle class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_tfidf, y_train)

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test_tfidf)

# model evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# save the model
import joblib
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


# prediction function for API
def predict_spam(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][1]
    return {
        "label": "SPAM" if prediction == 1 else "HAM",
        "confidence": round(probability * 100, 2)
    }

# Test it
print(predict_spam("Congratulations! You won a free iPhone, click here now!"))
print(predict_spam("Hey, are you coming to dinner tonight?"))