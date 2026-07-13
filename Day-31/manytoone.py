
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Heart Rate Forecasting", layout="wide")
st.title("❤️ Heart Rate Forecasting using One-to-Many RNN")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "heart_rate.csv"
MODEL_FILE = BASE_DIR / "heart_rnn.keras"

if not os.path.exists(CSV_FILE):
    st.error("Place heart_rate.csv in the same folder as app.py")
    st.stop()

df = pd.read_csv(str(CSV_FILE))
series = df["T1"].astype(float).values.reshape(-1,1)



scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

SEQ_LEN = 30
OUT_STEPS = 10

X, y = [], []
for i in range(len(scaled)-SEQ_LEN-OUT_STEPS):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN:i+SEQ_LEN+OUT_STEPS].flatten())

X = np.array(X)
y = np.array(y)

split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
    st.success("Loaded saved model.")
else:
    model = Sequential([
        SimpleRNN(64, activation="tanh", input_shape=(SEQ_LEN,1)),
        Dense(64, activation="relu"),
        Dense(OUT_STEPS)
    ])
    model.compile(optimizer="adam", loss="mse")
    with st.spinner("Training model..."):
        hist = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=30,
            batch_size=32,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
    model.save(MODEL_FILE)
    st.success("Training completed and model saved.")

    fig = plt.figure(figsize=(7,4))
    plt.plot(hist.history["loss"], label="Train")
    plt.plot(hist.history["val_loss"], label="Validation")
    plt.legend()
    plt.title("Training Loss")
    st.pyplot(fig)

pred = model.predict(X_test, verbose=0)

actual = scaler.inverse_transform(y_test[:,0].reshape(-1,1)).flatten()
predicted = scaler.inverse_transform(pred[:,0].reshape(-1,1)).flatten()

fig2 = plt.figure(figsize=(10,4))
plt.plot(actual[:100], label="Actual")
plt.plot(predicted[:100], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted (First Forecast Value)")
st.pyplot(fig2)

st.subheader("Predict Next 10 Heart Rate Values")

last_seq = scaled[-SEQ_LEN:].reshape(1,SEQ_LEN,1)

if st.button("Predict"):
    future = model.predict(last_seq, verbose=0).flatten().reshape(-1,1)
    future = scaler.inverse_transform(future).flatten()
    st.write("### Next 10 Predicted Heart Rates")
    st.table(pd.DataFrame({
        "Step": np.arange(1,11),
        "Predicted BPM": np.round(future,2)
    }))
