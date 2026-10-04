# train.py
# Trains a spam detector (TF-IDF + Naive Bayes) on the SMS spam dataset
# and saves the model and the vectorizer to disk.

import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Paths: the dataset we already have and the two files we save
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "dataset", "spam.csv")
MODEL_PATH = os.path.join(HERE, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(HERE, "tfidf_vectorizer.pkl")

# Column names inside the CSV file
TEXT_COLUMN = "v2"
LABEL_COLUMN = "v1"


def clean_text(text):
    """Make the text simple: lowercase and keep letters/numbers only."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # drop punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()  # squeeze extra spaces
    return text


def load_data():
    """Read the CSV, clean it up and return texts + labels (1 = spam, 0 = ham)."""
    # The file is latin-1 encoded, so we must tell pandas that.
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Keep only the two columns we need and give them easy names.
    df = df[[LABEL_COLUMN, TEXT_COLUMN]]
    df.columns = ["label", "text"]

    # Drop empty rows (missing values or blank text).
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Drop duplicate messages.
    df = df.drop_duplicates(subset=["text"])

    # Convert labels to numbers: spam = 1, everything else (ham) = 0.
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(
        lambda value: 1 if value == "spam" else 0
    )

    # Clean the text itself.
    df["text"] = df["text"].apply(clean_text)

    print("Rows after cleaning:", len(df))
    print("Label counts:", df["label"].value_counts().to_dict())
    return df["text"], df["label"]


def main():
    texts, labels = load_data()

    # 80% of the data to train, 20% to test.
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Turn text into numbers with TF-IDF.
    vectorizer = TfidfVectorizer(stop_words="english")
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # Train the Naive Bayes classifier.
    # alpha is the smoothing value; 0.1 works a bit better than the default 1.0.
    model = MultinomialNB(alpha=0.1)
    model.fit(x_train_vec, y_train)

    # Check how well it does on the test data.
    predictions = model.predict(x_test_vec)
    print("\nAccuracy: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["ham (0)", "spam (1)"]))

    # Save the model and the vectorizer so the app can use them later.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Saved:", os.path.basename(MODEL_PATH), "and", os.path.basename(VECTORIZER_PATH))


if __name__ == "__main__":
    main()
