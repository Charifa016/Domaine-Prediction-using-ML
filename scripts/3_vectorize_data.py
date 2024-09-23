import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os


def vectorize_texts(input_csv, output_vectorizer, output_train_data, output_test_data):
    df = pd.read_csv(input_csv)
    df["Texte"] = df["Texte"].fillna('')
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Texte"])
    y = df["Domaine"]
    with open(output_vectorizer, 'wb') as f:
        pickle.dump(vectorizer, f)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with open(output_train_data, 'wb') as f:
        pickle.dump((X_train, y_train), f)
    
    with open(output_test_data, 'wb') as f:
        pickle.dump((X_test, y_test), f)

input_csv = "data/processed/cleaned_cv_data.csv"
output_vectorizer = "models/vectorizer.pkl"
output_train_data = "data/processed/train_data.pkl"
output_test_data = "data/processed/test_data.pkl"

vectorize_texts(input_csv, output_vectorizer, output_train_data, output_test_data)


