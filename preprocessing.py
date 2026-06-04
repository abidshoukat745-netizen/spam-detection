import numpy as np
import pandas as pd

# read the data
df = pd.read_csv('cleaned_data.csv', encoding='latin-1')

print(df.head(5))

print(df.info())
print(df.columns)

df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
print(df.columns)

df = df.rename(columns={'v1': 'label', 'v2': 'text'})
print(df.columns)

print(df['label'].value_counts())

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# importing libraries for text preprocessing
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess_text)

# Save cleaned data
df.to_csv(r'C:\Users\Mehar\Desktop\sentiment analysis\cleaned_data.csv', index=False)

