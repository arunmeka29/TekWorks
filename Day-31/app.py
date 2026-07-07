import os
import pickle
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings(
    "ignore",
    message=r".*reset_default_graph.*",
)

import numpy as np
import streamlit as st


MODEL = "char_rnn_many_to_many.keras"
VOCAB = "char_vocab.pkl"
DATASET = "tiny-shakespeare.txt"

SEQ_LEN = 80
STEP_SIZE = 3
EMBED_DIM = 64
RNN_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 8
TRAIN_SPLIT = 0.9


def import_tensorflow():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Embedding, Input, SimpleRNN
    from tensorflow.keras.models import Sequential, load_model

    return tf, EarlyStopping, Dense, Embedding, Input, SimpleRNN, Sequential, load_model


def load_text():
    with open(DATASET, "r", encoding="utf-8") as file:
        return file.read()


def build_vocabulary(text):
    characters = sorted(set(text))
    char_to_idx = {char: index for index, char in enumerate(characters)}
    idx_to_char = characters
    return char_to_idx, idx_to_char


def encode_text(text, char_to_idx):
    return np.array([char_to_idx[char] for char in text], dtype=np.int32)


def split_input_target(chunk):
    return chunk[:, :-1], chunk[:, 1:]


def make_sequence_dataset(encoded_text, shuffle):
    if len(encoded_text) <= SEQ_LEN + 1:
        raise ValueError("Dataset is too short for the configured sequence length.")

    tf, _, _, _, _, _, _, _ = import_tensorflow()

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=encoded_text,
        targets=None,
        sequence_length=SEQ_LEN + 1,
        sequence_stride=STEP_SIZE,
        sampling_rate=1,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )

    dataset = dataset.map(
        split_input_target,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(vocab_size):
    _, _, Dense, Embedding, Input, SimpleRNN, Sequential, _ = import_tensorflow()

    model = Sequential(
        [
            Input(shape=(SEQ_LEN,)),
            Embedding(input_dim=vocab_size, output_dim=EMBED_DIM),
            SimpleRNN(RNN_UNITS, return_sequences=True),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(f"{DATASET} was not found.")

    _, EarlyStopping, _, _, _, _, _, _ = import_tensorflow()

    print("Training many-to-many character RNN...")

    text = load_text()
    char_to_idx, idx_to_char = build_vocabulary(text)
    encoded_text = encode_text(text, char_to_idx)

    split_index = int(len(encoded_text) * TRAIN_SPLIT)
    train_encoded = encoded_text[:split_index]
    val_encoded = encoded_text[split_index:]

    train_dataset = make_sequence_dataset(train_encoded, shuffle=True)
    val_dataset = make_sequence_dataset(val_encoded, shuffle=False)

    with open(VOCAB, "wb") as file:
        pickle.dump(
            {
                "char_to_idx": char_to_idx,
                "idx_to_char": idx_to_char,
                "vocab_size": len(idx_to_char),
            },
            file,
        )

    model = build_model(len(idx_to_char))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL)

    validation_loss, validation_accuracy = model.evaluate(
        val_dataset,
        verbose=0,
    )

    load_saved_model.clear()
    load_saved_vocab.clear()

    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")

    return {
        "history": history.history,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "vocab_size": len(idx_to_char),
        "dataset_size": len(encoded_text),
    }


@st.cache_resource
def load_saved_model():
    _, _, _, _, _, _, _, load_model = import_tensorflow()
    return load_model(MODEL)


@st.cache_resource
def load_saved_vocab():
    with open(VOCAB, "rb") as file:
        return pickle.load(file)


def sample_next_index(probabilities, temperature):
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.maximum(probabilities, 1e-8)

    scaled = np.log(probabilities) / max(temperature, 0.1)
    scaled = np.exp(scaled - np.max(scaled))
    scaled = scaled / np.sum(scaled)

    return int(np.random.choice(len(scaled), p=scaled))


def generate_text(seed_text, length, temperature):
    model = load_saved_model()
    vocab = load_saved_vocab()

    char_to_idx = vocab["char_to_idx"]
    idx_to_char = vocab["idx_to_char"]
    pad_index = char_to_idx.get(" ", 0)

    if not seed_text:
        seed_text = "ROMEO:\n"

    generated_indices = [char_to_idx.get(char, pad_index) for char in seed_text]

    for _ in range(length):
        window = generated_indices[-SEQ_LEN:]
        padded = np.full((SEQ_LEN,), pad_index, dtype=np.int32)
        padded[-len(window):] = window

        predictions = model(
            padded[np.newaxis, :],
            training=False,
        ).numpy()
        next_index = sample_next_index(predictions[0, -1], temperature)
        generated_indices.append(next_index)

    return "".join(idx_to_char[index] for index in generated_indices)


st.set_page_config(
    page_title="Shakespeare Character Generator",
    page_icon="T",
    layout="centered",
)

st.title("Character-Level Text Generation using RNN")
st.write("Many to Many RNN")
st.caption("The model learns a next-character sequence at every time step.")
st.info("The page now loads first. TensorFlow is loaded only when you generate text or retrain the model.")

if not os.path.exists(DATASET):
    st.error(f"{DATASET} was not found in the current folder.")
    st.stop()

training_stats = None

if not os.path.exists(MODEL) or not os.path.exists(VOCAB):
    st.warning("Saved model not found. Training a new many-to-many RNN...")

    with st.spinner("Training model. Please wait..."):
        training_stats = train_model()

    st.success("Training completed!")

st.subheader("Generate Text")

seed_text = st.text_area(
    "Enter seed text",
    value="ROMEO:\n",
    height=180,
)

generation_length = st.slider(
    "Characters to generate",
    min_value=50,
    max_value=500,
    value=150,
    step=25,
)

temperature = st.slider(
    "Temperature",
    min_value=0.2,
    max_value=1.5,
    value=0.8,
    step=0.1,
)

if st.button("Generate"):
    with st.spinner("Loading TensorFlow and generating text..."):
        generated_text = generate_text(
            seed_text=seed_text,
            length=generation_length,
            temperature=temperature,
        )

    st.text_area(
        "Generated output",
        value=generated_text,
        height=320,
    )

if st.button("Retrain Model"):
    with st.spinner("Retraining model. Please wait..."):
        training_stats = train_model()

    st.success("Model retrained successfully!")

if training_stats is not None:
    st.info(
        "Vocabulary size: "
        f"{training_stats['vocab_size']} | "
        "Characters used: "
        f"{training_stats['dataset_size']} | "
        "Validation accuracy: "
        f"{training_stats['validation_accuracy']:.2%}"
    )
    
