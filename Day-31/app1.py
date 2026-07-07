import os
import pickle
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings(
    "ignore",
    message=r".*reset_default_graph.*",
)

import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "diabetes.csv"
MODEL = BASE_DIR / "one_to_one_rnn.keras"
SCALER_FILE = BASE_DIR / "one_to_one_scaler.pkl"
META_FILE = BASE_DIR / "one_to_one_meta.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 25
BATCH_SIZE = 16


def import_tensorflow():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Input, SimpleRNN
    from tensorflow.keras.models import Sequential, load_model

    return tf, EarlyStopping, Dense, Input, SimpleRNN, Sequential, load_model


def load_dataset():
    return pd.read_csv(DATASET)


def prepare_data(df):
    if "Outcome" not in df.columns:
        raise ValueError("Dataset must contain an 'Outcome' target column.")

    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # One-to-one setup: one time step -> one output
    x_train_rnn = x_train_scaled.reshape(x_train_scaled.shape[0], 1, x_train_scaled.shape[1])
    x_test_rnn = x_test_scaled.reshape(x_test_scaled.shape[0], 1, x_test_scaled.shape[1])

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        x_train_rnn,
        x_test_rnn,
        scaler,
    )


def build_model(feature_count):
    _, _, Dense, Input, SimpleRNN, Sequential, _ = import_tensorflow()

    model = Sequential(
        [
            Input(shape=(1, feature_count)),
            SimpleRNN(32),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model():
    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET.name} was not found.")

    _, EarlyStopping, _, _, _, _, _ = import_tensorflow()

    df = load_dataset()
    x_train, x_test, y_train, y_test, x_train_rnn, x_test_rnn, scaler = prepare_data(df)

    model = build_model(x_train.shape[1])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train_rnn,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    model.save(MODEL)

    with open(SCALER_FILE, "wb") as file:
        pickle.dump(scaler, file)

    with open(META_FILE, "wb") as file:
        pickle.dump(
            {
                "feature_names": x_train.columns.tolist(),
                "defaults": df.drop(columns=["Outcome"]).median().to_dict(),
            },
            file,
        )

    predictions = (model.predict(x_test_rnn, verbose=0) > 0.5).astype(int).ravel()
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    load_saved_model.clear()
    load_saved_scaler.clear()
    load_saved_meta.clear()

    return {
        "accuracy": accuracy,
        "report": report,
        "epochs_ran": len(history.history["loss"]),
    }


@st.cache_resource
def load_saved_model():
    _, _, _, _, _, _, load_model = import_tensorflow()
    return load_model(MODEL)


@st.cache_resource
def load_saved_scaler():
    with open(SCALER_FILE, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_saved_meta():
    with open(META_FILE, "rb") as file:
        return pickle.load(file)


def predict_single(values):
    model = load_saved_model()
    scaler = load_saved_scaler()
    meta = load_saved_meta()

    frame = pd.DataFrame([values], columns=meta["feature_names"])
    scaled = scaler.transform(frame)
    sequence = scaled.reshape(1, 1, scaled.shape[1])

    probability = float(model.predict(sequence, verbose=0)[0][0])
    label = "Diabetic" if probability >= 0.5 else "Not Diabetic"

    return label, probability


st.set_page_config(
    page_title="One-to-One RNN",
    page_icon="R",
    layout="centered",
)

st.title("One-to-One RNN Classification")
st.write("Simple RNN using one time step and one output")
st.caption("Dataset: Kaggle Pima Indians Diabetes Database")

training_stats = None

if not DATASET.exists():
    st.warning(
        "Dataset missing. Download the Pima Indians Diabetes dataset from Kaggle "
        "and keep the file as diabetes.csv in the same folder as app1.py."
    )
    st.stop()

if not MODEL.exists() or not SCALER_FILE.exists() or not META_FILE.exists():
    with st.spinner("Training one-to-one RNN model..."):
        training_stats = train_model()

    st.success("Training completed!")

meta = load_saved_meta() if META_FILE.exists() else None

if st.button("Retrain Model"):
    with st.spinner("Retraining model..."):
        training_stats = train_model()

    st.success("Model retrained successfully!")

if training_stats is not None:
    st.info(
        f"Test accuracy: {training_stats['accuracy']:.2%} | "
        f"Epochs used: {training_stats['epochs_ran']}"
    )
    st.code(training_stats["report"], language="text")

st.subheader("Try a Prediction")

input_values = {}

for feature_name in meta["feature_names"]:
    default_value = float(meta["defaults"][feature_name])
    input_values[feature_name] = st.number_input(
        feature_name,
        value=default_value,
        format="%.2f",
    )

if st.button("Predict"):
    with st.spinner("Running prediction..."):
        label, probability = predict_single(input_values)

    if label == "Diabetic":
        st.error(f"Prediction: {label}")
        st.write(f"Confidence: {probability:.2%}")
    else:
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {(1 - probability):.2%}")
