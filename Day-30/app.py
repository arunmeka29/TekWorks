# RNN practical (many to one)
# SMS Spam Detection using simple RNN (Many-to-one)

import pickle
import re
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="SMS Spam Detection",
    layout="centered"
)


BASE_DIR = Path(__file__).resolve().parent
MODEL = BASE_DIR / "spam_model.keras"
TOKENIZER = BASE_DIR / "tokenizer.pkl"
DATASET = BASE_DIR / "spam.csv"

MAX_WORDS = 5000
MAX_LEN = 50


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def model_files_exist():
    return MODEL.exists() and TOKENIZER.exists()


def train_model():
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET, encoding="latin-1")
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "text"]

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["text"] = df["text"].apply(clean_text)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])
    x = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    y = df["label"]

    with TOKENIZER.open("wb") as file:
        pickle.dump(tokenizer, file)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Sequential(
        [
            Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
            SimpleRNN(128),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=32,
        verbose=0
    )

    model.save(str(MODEL))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = (model.predict(x_test, verbose=0) > 0.5).astype(int)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


@st.cache_resource(show_spinner=False)
def load_saved_model():
    from tensorflow.keras.models import load_model

    return load_model(str(MODEL))


@st.cache_resource(show_spinner=False)
def load_saved_tokenizer():
    with TOKENIZER.open("rb") as file:
        return pickle.load(file)


def predict_sms(message):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = load_saved_model()
    tokenizer = load_saved_tokenizer()

    cleaned_message = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned_message])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    probability = float(model.predict(sequence, verbose=0)[0][0])

    if probability > 0.5:
        return "spam", probability

    return "ham", 1 - probability


st.title("SMS Spam Detection using RNN")
st.write("Many to One RNN")
st.caption("The page loads first now. The first prediction can still take a few seconds while the model loads.")

message = st.text_area("Enter SMS Message")
model_ready = model_files_exist()

if not model_ready:
    st.warning("Model files are missing. Train the model once to start predictions.")

    if st.button("Train Model"):
        with st.spinner("Training model. Please wait..."):
            train_model()
            load_saved_model.clear()
            load_saved_tokenizer.clear()
        model_ready = True
        st.success("Training completed! You can predict messages now.")

if st.button("Predict", disabled=not model_ready):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        with st.spinner("Loading model and analyzing your message..."):
            prediction, probability = predict_sms(message)

        if prediction == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: HAM")

        st.write("Confidence:", round(probability * 100, 2), "%")
