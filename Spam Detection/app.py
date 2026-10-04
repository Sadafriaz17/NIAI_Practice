# app.py
# Simple Streamlit frontend: type a message, click the button,
# and see if the message is Spam or Not Spam with a confidence score.

import os
import re

import joblib
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(HERE, "tfidf_vectorizer.pkl")

# Two ready-made messages the user can try with one click.
EXAMPLE_SPAM = (
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
    "Text FA to 87121 to receive entry question"
)
EXAMPLE_NOT_SPAM = (
    "Ok lar... Joking wif u oni... Are you coming home for dinner tonight?"
)


def clean_text(text):
    """Same cleaning used while training, so the input matches the model."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    """Load the saved model and vectorizer. Returns (None, None) if not found."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def fill_message(text):
    """Callback used by the example buttons to put text in the box."""
    st.session_state["message"] = text


st.title("Spam Detection")
st.write("Type a message below and click the button to check it.")

model, vectorizer = load_model()

if model is None:
    # Clear error if the training script was never run.
    st.error(
        "Model files not found. Please run the training script first:\n\n"
        "`python train.py`"
    )
    st.stop()

# The text box for the user's message.
st.text_area("Your message", key="message", height=120)

# Two example buttons (clicking one fills the text box).
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

# The main action button.
if st.button("Check Message", type="primary"):
    message = st.session_state.get("message", "").strip()

    if not message:
        st.warning("Please type a message first.")
    else:
        # Convert the message the same way as training, then predict.
        vectors = vectorizer.transform([clean_text(message)])
        prediction = int(model.predict(vectors)[0])
        confidence = float(model.predict_proba(vectors)[0][prediction]) * 100

        if prediction == 1:
            st.error("Spam - confidence {:.1f}%".format(confidence))
        else:
            st.success("Not Spam - confidence {:.1f}%".format(confidence))
