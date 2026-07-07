# RNN practical (many to one)
# SMS Spam Detection using simple RNN (Many-to-one)

# import libraries

import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# configuration

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Train model

def train_model():

    print("Training Dataset...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]

    df.columns = ["label", "text"]

    print(df.head())

    print(df["label"].value_counts())

    # convert label to numbers

    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # clean SMS

    df["text"] = df["text"].apply(clean_text)

    # Tokenization

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    x = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post"
    )

    y = df["label"]

    print("x shape:", x.shape)

    print("y shape:", y.shape)

    # save tokenizer

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # train test split

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Build RNN model

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(128)
    )

    # hidden layer

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    # output layer

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=32
    )

    # save model

    model.save(MODEL)

    # evaluate

    loss, accuracy = model.evaluate(x_test, y_test)

    print("\nAccuracy:", accuracy)

    predictions = (
        model.predict(x_test) > 0.5
    ).astype(int)

    print(classification_report(
        y_test,
        predictions
    ))

    print(confusion_matrix(
        y_test,
        predictions
    ))


# ----------------- NEW -----------------

@st.cache_resource
def load_saved_model():
    return load_model(MODEL)


@st.cache_resource
def load_saved_tokenizer():
    with open(TOKENIZER, "rb") as f:
        return pickle.load(f)

# ---------------------------------------


# predictions

def predict_sms(message):

    model = load_saved_model()
    tokenizer = load_saved_tokenizer()

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if probability > 0.5:
        return "spam", probability
    else:
        return "ham", 1 - probability


# Train only if model doesn't exist

if not os.path.exists(MODEL):

    st.warning("Model not found. Training...")

    with st.spinner("Training model. Please wait..."):
        train_model()

    st.success("Training completed!")


# streamlit UI

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

st.title("SMS Spam Detection using RNN")

st.write("Many to One RNN")

message = st.text_area("Enter SMS Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        prediction, probability = predict_sms(message)

        if prediction == "spam":
            st.error("Prediction: SPAM 🚨")
        else:
            st.success("Prediction: HAM ✅")

        st.write(
            "Confidence:",
            round(probability * 100, 2),
            "%"
        )