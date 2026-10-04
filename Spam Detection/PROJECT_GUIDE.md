# Project Guide: Spam Detection

This guide explains every file in this project in plain, beginner friendly words.
Start with the first section, then jump to any file you want to understand.

## How everything fits together

Think of the project as a small factory with a few rooms.

1. The `dataset` folder holds the raw material: a file full of real text messages, and every message is already marked as spam or ham.
2. `train.py` is the worker. It reads the raw material, cleans it, teaches a Naive Bayes model, measures how good the model is, and saves two files: the trained model, and the tool that turns text into numbers.
3. `spam_model.pkl` and `tfidf_vectorizer.pkl` are the finished product. They are the memory of the training, so we never have to train again just to check one message.
4. `app.py` is the shop window. It is a small Streamlit web page where you type a message, click a button, and read Spam or Not Spam with a confidence score. It works by loading the two pkl files.
5. `spam_detection.ipynb` is the classroom. It walks through the very same journey as `train.py`, but one small step at a time, so a beginner can see and understand every part.
6. `requirements.txt` is the shopping list of libraries the project needs.
7. `README.md` is the short instruction card for the project.

The whole flow in one sentence: the dataset is read by `train.py`, which writes the two pkl files, which are then loaded by `app.py` to answer your questions.

---

## The dataset file: spam.csv inside the dataset folder

**What it is for.** This file holds the messages the project learns from. It is the only source of examples, and the project only ever reads it.

**What is inside it.** The file is a CSV file, which stands for comma separated values. In plain words, it is a plain text table where commas separate the columns.

* It contains 5572 rows, so 5572 real text messages.
* The column `v1` holds the label. Its values are `ham` for a normal message and `spam` for junk, and there are 4825 ham messages and 747 spam messages.
* The column `v2` holds the message text, and this is the column the model learns from.
* Three more columns, named `Unnamed: 2`, `Unnamed: 3`, and `Unnamed: 4`, are almost empty. The project ignores them.
* The file is stored in an old text format called latin-1. If pandas tries to read it as the modern utf-8 format, it stops with a UnicodeDecodeError. That is why both `train.py` and the notebook tell pandas to use latin-1.
* 403 messages appear more than once in the file. After the empty rows and the copies are removed, 5158 rows remain, made of 4516 ham messages and 642 spam messages.

**How it is used.** `train.py` reads it at the very start, and the notebook reads it in Step 2. Nothing writes to it, so the dataset stays exactly as it was.

---

## train.py

**What it is for.** This script does the learning. It reads the dataset, cleans the text, trains a multinomial Naive Bayes model, prints how well the model did, and saves the model plus the text tool to the disk.

Here is the walkthrough, part by part.

### Part 1: the imports

```python
import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
```

* `os` comes with Python and is used to build file paths that work on any computer.
* `re` also comes with Python and handles regular expressions, which are patterns used to remove symbols from text.
* `joblib` is the tool that saves a trained model into a file and loads it back.
* `pandas` reads the CSV file and works with the table of messages.
* `TfidfVectorizer` turns text into numbers.
* `train_test_split` cuts the data into a training part and a testing part.
* `MultinomialNB` is the Naive Bayes model itself.
* `accuracy_score` and `classification_report` measure the quality of the result.

### Part 2: the paths and the column names

```python
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "dataset", "spam.csv")
MODEL_PATH = os.path.join(HERE, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(HERE, "tfidf_vectorizer.pkl")

TEXT_COLUMN = "v2"
LABEL_COLUMN = "v1"
```

* `HERE` is the folder where the script itself lives, found automatically. Because of this, you can start the script from any folder and it still finds the right files.
* `DATA_PATH` points at the dataset, and `MODEL_PATH` and `VECTORIZER_PATH` point at the two files the script will save.
* The last two lines simply name the columns we need, so the rest of the code never has to repeat the odd names `v1` and `v2`.

### Part 3: the clean_text function

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

* `str(text).lower()` makes everything lowercase, so Free and free become the same word instead of two different words.
* The first `re.sub` replaces every character that is not a letter, a number, or a space with a space. In short, it removes all the symbols.
* The second `re.sub` squeezes several spaces into one space, and `strip` removes the spaces at the start and at the end.
* The result is a tidy message. Keeping this function simple matters, because the app must clean new messages in exactly the same way.

### Part 4: the load_data function

This function turns the raw file into clean text and clean labels. It works in small steps.

```python
df = pd.read_csv(DATA_PATH, encoding="latin-1")
```

Reads the file, and the `encoding="latin-1"` is what avoids the UnicodeDecodeError mentioned earlier.

```python
df = df[[LABEL_COLUMN, TEXT_COLUMN]]
df.columns = ["label", "text"]
```

Keeps only the two useful columns and renames them to `label` and `text`, which is easier to read and to type.

```python
df = df.dropna(subset=["label", "text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset=["text"])
```

Removes rows with a missing value, trims the spaces at the edges of each message, removes rows with an empty message, and finally removes rows whose message is an exact copy of an earlier row. Copies are dangerous, because they would make the model believe a repeated message is more common than it really is.

```python
df["label"] = df["label"].astype(str).str.strip().str.lower().map(
    lambda value: 1 if value == "spam" else 0
)
```

Turns the label words into numbers, because a model can only work with numbers. Spam becomes 1 and ham becomes 0.

```python
df["text"] = df["text"].apply(clean_text)
print("Rows after cleaning:", len(df))
print("Label counts:", df["label"].value_counts().to_dict())
```

Finally every message is passed through the cleaning function from Part 3, and two short lines are printed so you can see the effect: 5158 rows, with 4516 of label 0 and 642 of label 1. The function then gives back the text column and the label column.

### Part 5: the main function, where the learning happens

**1. The split into training data and testing data**

```python
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
```

The messages are cut into two piles: 80 percent for training and 20 percent for testing. Why not use everything for training? Because a model tested on the very messages it memorised would look perfect while learning nothing useful. The setting `random_state=42` is only a fixed starting point, so the split is always the same and the printed scores can be reproduced. `stratify=labels` keeps the same mix of spam and ham in both piles.

**2. Turning text into numbers with TF-IDF**

```python
vectorizer = TfidfVectorizer(stop_words="english")
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)
```

The vectorizer learns the words from the training messages and turns them into numbers, giving a high value to a word that is important in one message but rare in the others. Very common English words are ignored, because they carry almost no meaning. Notice the difference between the two calls: `fit_transform` on the training messages, which learns the words, and `transform` on the testing messages, which only applies what was learned. This keeps the test honest, because the model never sees the testing text while it is still learning.

**3. Training the model**

```python
model = MultinomialNB(alpha=0.1)
model.fit(x_train_vec, y_train)
```

The model reads every training message with its correct label and writes down how often each word appears in spam messages and in ham messages. The setting `alpha=0.1` is called smoothing, and it stops the model from giving a probability of zero to a word it has never seen before. The usual starting value is 1.0, and 0.1 simply scored better on this dataset, both in accuracy and in how much spam it caught.

**4. Checking the quality**

```python
predictions = model.predict(x_test_vec)
print("\nAccuracy: {:.4f}".format(accuracy_score(y_test, predictions)))
print(classification_report(y_test, predictions, target_names=["ham (0)", "spam (1)"]))
```

The model guesses the labels of the testing messages, and we compare its guesses with the real answers. The accuracy printed on this dataset is 0.9855, which means roughly 99 percent of the testing messages were labelled correctly. The report then shows precision, recall, and f1 for ham and for spam, so you can see that normal messages are almost never disturbed, while around 9 spam messages out of every 100 still slip through.

**5. Saving the result**

```python
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print("Saved:", os.path.basename(MODEL_PATH), "and", os.path.basename(VECTORIZER_PATH))
```

The trained model and the fitted vectorizer are written to the disk as two pkl files. Both are saved, because a new message has to be turned into numbers by the very same vectorizer the model was trained with.

**6. The starting point of the script**

```python
if __name__ == "__main__":
    main()
```

This line means: run the training only when this file is started directly, and not when it is imported by another file.

---

## spam_model.pkl

**What it is for.** This file is the trained model, saved to the disk so that the app does not have to train again every time it starts.

**What is inside it.** It is not readable text. `joblib` wrote a binary file of about 234 kilobytes holding everything the Naive Bayes model learned: for each of the words it saw, how often that word appeared in ham messages and how often in spam messages, how many messages of each kind were counted, and the smoothing value `alpha` of 0.1.

**How it is used.** `app.py` loads it with `joblib.load` and then calls two methods on it: `predict`, which gives the answer 0 or 1, and `predict_proba`, which gives the probability used for the confidence percentage. If you delete this file, the app shows an error asking you to run `train.py` first. You can always create it again simply by running the training script.

## tfidf_vectorizer.pkl

**What it is for.** This file remembers how the text was turned into numbers.

**What is inside it.** A binary file of about 147 kilobytes, again written by `joblib`, holding the fitted `TfidfVectorizer`. In practice it stores the words the vectorizer learned from the training messages, together with its rules: ignore the very common English words, work in lowercase, and give each word the right weight. There is no readable text inside, and it is not a model, so it makes no predictions on its own.

**How it is used.** `app.py` loads it and uses it to turn the message you typed into the same kind of numbers the model was trained with. This step is not optional. A new message turned into numbers with a different word list would be a language the model never learned. That is exactly why this file is saved next to the model and always used as a pair.

---

## app.py

**What it is for.** This is the small Streamlit web page. It shows a title, one text box, and one button, and it answers with Spam or Not Spam plus a confidence percentage. It never trains anything, it only uses the two saved pkl files.

Here is the walkthrough, block by block.

### Part 1: the imports and the file paths

```python
import os
import re

import joblib
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(HERE, "tfidf_vectorizer.pkl")
```

* `os` finds out which folder the app lives in, so the pkl files are always found no matter where you start the app from.
* `re` is used again for the same symbol removal as in training.
* `joblib` loads the two saved pkl files.
* `streamlit` is the library that draws the web page, and it is the reason the whole app fits in one small file.

### Part 2: two ready made example messages

```python
EXAMPLE_SPAM = (
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
    "Text FA to 87121 to receive entry question"
)
EXAMPLE_NOT_SPAM = (
    "Ok lar... Joking wif u oni... Are you coming home for dinner tonight?"
)
```

Two real messages from the dataset are stored so the user can click a button and try the app instantly, without having to invent a message. One is clearly spam and the other is clearly a normal message, and the brackets are only there to keep the long lines readable.

### Part 3: the same cleaning function

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

This is a copy of the cleaning step from `train.py`, and the copy is on purpose. The model was trained on text in this exact shape, so a message typed in the app must be cleaned in exactly the same way before it is checked. If the app skipped this step, the scores would be worse for no obvious reason.

### Part 4: loading the model once

```python
@st.cache_resource
def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
```

* `st.cache_resource` tells Streamlit to load the files only once and keep them in memory, so every later click is fast.
* The `if` check asks whether both pkl files are really there. If either one is missing, the function gives back two empty values instead of crashing, and the app can then show a friendly error.
* If both files exist, they are loaded and given back to the rest of the app.

### Part 5: the example button helper

```python
def fill_message(text):
    st.session_state["message"] = text
```

Streamlit runs the whole file again every time you click something. The session state is the small memory that survives those reruns. This function is set as the callback of the two example buttons: when you click one, the text is stored under the key `message`, and the text box, which uses the same key, shows that text.

### Part 6: the page itself and the missing model check

```python
st.title("Spam Detection")
st.write("Type a message below and click the button to check it.")

model, vectorizer = load_model()

if model is None:
    st.error(
        "Model files not found. Please run the training script first:\n\n"
        "`python train.py`"
    )
    st.stop()
```

* The first two lines draw the title and a one line instruction.
* Then the model and the vectorizer are loaded once.
* If the loading failed, the app shows a clear red error telling the user to run the training script first, and `st.stop()` ends the page right there. This is far friendlier than a long technical error.

### Part 7: the text box and the two example buttons

```python
st.text_area("Your message", key="message", height=120)

left, right = st.columns(2)
left.button(
    "Example 1 (likely spam)",
    on_click=fill_message,
    args=(EXAMPLE_SPAM,),
    use_container_width=True,
)
right.button(
    "Example 2 (likely not spam)",
    on_click=fill_message,
    args=(EXAMPLE_NOT_SPAM,),
    use_container_width=True,
)
```

* The text box is the only input of the app. The key `message` is what connects it to the session state used by the example buttons.
* The two buttons sit next to each other in two columns. Each one is linked to the helper from Part 5, and `args` is simply the message it should place in the box. Clicking a button fills the box, and you can still edit the text afterwards.

### Part 8: the main button, the prediction, and the colours

```python
if st.button("Check Message", type="primary"):
    message = st.session_state.get("message", "").strip()

    if not message:
        st.warning("Please type a message first.")
    else:
        vectors = vectorizer.transform([clean_text(message)])
        prediction = int(model.predict(vectors)[0])
        confidence = float(model.predict_proba(vectors)[0][prediction]) * 100

        if prediction == 1:
            st.error("Spam - confidence {:.1f}%".format(confidence))
        else:
            st.success("Not Spam - confidence {:.1f}%".format(confidence))
```

* Everything inside this block runs only after the Check Message button is clicked.
* The typed message is read from the session state and trimmed.
* If the box is empty, a small yellow warning asks the user to type something first.
* Otherwise three things happen in order: the message is cleaned with the very same steps used in training, the vectorizer turns it into numbers, and the model gives its guess together with the probability of that guess.
* A guess of 1 means spam, and the result is shown with `st.error`, which Streamlit paints red. A guess of 0 means a normal message, shown with `st.success`, which is painted green.
* The confidence is the probability of the answer that was chosen, turned into a percentage. So 99.9 percent means the model is very sure, while 55 percent would mean it is unsure.

---

## spam_detection.ipynb

**What it is for.** This is the beginner notebook. It teaches the same journey as `train.py`, but slowly, in 42 cells: 17 text cells and 25 code cells. Every text cell explains what the next code cell does and why, before you run it. The notebook never saves anything to the disk.

**How it is laid out.** The notebook runs in fifteen steps, with an introduction at the top and a summary at the end.

1. Step 1 brings in pandas and scikit-learn, and explains what each library is for.
2. Step 2 loads the file with the pandas function `read_csv`, mentions the latin-1 encoding, and shows the first five rows.
3. Step 3 explores the data: the shape of the table, the count of ham and spam, and a few real examples of each kind.
4. Step 4 cleans the data: two columns kept, the names changed to label and message, empty rows and copies removed, and ham and spam turned into 0 and 1.
5. Step 5 cleans the text, and prints one message before and after so you can see the symbols disappearing.
6. Step 6 splits the data into 80 percent training and 20 percent testing, and explains with a student and exam comparison why that matters.
7. Step 7 turns the text into numbers with TF-IDF, and explains in short why the important words stand out.
8. Step 8 trains the multinomial Naive Bayes model with `alpha=0.1`, and explains the model in simple words.
9. Step 9 tests the model and prints the accuracy, which comes out at 0.9855, together with the classification report.
10. Step 10 reads those numbers in plain words, using the real figures: about 1017 messages right out of 1032, spam precision of 0.97, and spam recall of 0.91.
11. Step 11 builds the confusion matrix as a small pandas table with clear row and column names, and explains the four boxes.
12. Step 12 reads that table: 900 and 117 correct answers, 4 false alarms, and 11 spam messages that slipped through.
13. Step 13 defines the function `check_message`, which cleans a message, turns it into numbers, and prints Spam or Not Spam with a confidence percentage.
14. Step 14 tries five messages: two spam, two normal, and one about a free ticket.
15. Step 15 has a variable named `my_message` that you can edit and run again, and the final text cell summarises what was learned and suggests next steps.

The notebook uses only pandas and scikit-learn, and the accuracy it prints matches the script exactly, which is a good sign that both follow the same recipe.

## requirements.txt

**What it is for.** This is the shopping list of the libraries the project needs, with the exact versions that were used.

```
scikit-learn==1.7.2
pandas==2.3.3
streamlit==1.64.0
joblib==1.6.0
```

* `scikit-learn` provides the TF-IDF tool, the Naive Bayes model, and the measuring tools.
* `pandas` reads the CSV file and works with the tables.
* `streamlit` draws the web page in `app.py`.
* `joblib` saves and loads the two pkl files.

The two equals signs mean that exact version, which is a way of saying that the project was built and tested with these versions. On this computer all four are already inside the conda environment named spamdetect. You can see the full list with the command `python -m pip list`.

## README.md

**What it is for.** The short instruction card. It says what the project is, which dataset it uses, how to train the model, and how to start the app.

**What is inside.** A short description of the project, a line naming the dataset file and its two columns, a code block with `python train.py`, a note that this saves the two pkl files, a code block with `streamlit run app.py`, a short line about how to use the page, and a reminder that the training must be done first. It is the file to open when you only want the two commands and nothing else.

---

## How to run everything

You need one terminal opened in the project folder, the folder that holds `train.py`, `app.py`, and `spam_detection.ipynb`. On this computer the four libraries are already installed inside the conda environment named spamdetect, so activate it first if it is not active yet.

**Step 1. Train the model.** Run this once before the first use, and again only if you change the training code:

```bash
python train.py
```

It prints the row counts, the accuracy of 0.9855, and the classification report, and then it writes `spam_model.pkl` and `tfidf_vectorizer.pkl` into the project folder.

**Step 2. Start the web app.** Run this every time you want to use the page:

```bash
streamlit run app.py
```

Streamlit prints a local address in the terminal and opens the page in your browser. Type a message, or click one of the two example buttons, then click Check Message.

**Step 3. Open the notebook.** In VS Code, open `spam_detection.ipynb`, then pick the kernel in the top right corner. Choose the kernel named spamdetect, which is the environment that has pandas and scikit-learn. Then click Run All, or run the cells one by one with Shift and Enter.

If your plain `python` command cannot find scikit-learn or streamlit, it is pointing at a different environment. In that case use the python of the spamdetect environment, or activate that environment before running the commands above.


