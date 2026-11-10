
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load dataset
df = pd.read_csv("dataset/news.csv")

print("\nSample records:")
print(df[["title", "label"]].head(10))

# Display dataset information
print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())

# Remove rows with missing labels
df = df.dropna(subset=["label"]).copy()

# Convert text columns to strings and handle missing values
df["text"] = df["text"].fillna("").astype(str)

# Combine title and text if title exists
if "title" in df.columns:
    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["title"] + " " + df["text"]
else:
    df["content"] = df["text"]

# Remove empty news articles
df = df[df["content"].str.strip() != ""].copy()

# Prepare features and labels
X = df["content"]
y = df["label"]

print("Cleaned dataset shape:", df.shape)
print("Label distribution:")
print(y.value_counts())


# Combine title and news text if title exists
if "title" in df.columns:
    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
else:
    df["content"] = df["text"].astype(str)

X = df["content"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")

print("Model trained and saved successfully!")
