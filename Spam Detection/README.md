# Spam Detection

A small spam detector: a message is classified as **Spam** or **Not Spam** using
TF-IDF + Multinomial Naive Bayes, with a simple Streamlit interface.

Dataset: `dataset/spam.csv` (latin-1 encoded, columns `v1` = label, `v2` = text).

## Train the model

```bash
python train.py
```

This prints the accuracy and classification report and saves `spam_model.pkl`
and `tfidf_vectorizer.pkl` next to the scripts.

## Run the app

```bash
streamlit run app.py
```

Type a message (or click one of the two examples), then click **Check Message**
to see the result and the confidence percentage.

Note: run `train.py` first, otherwise the app shows an error asking you to do so.
