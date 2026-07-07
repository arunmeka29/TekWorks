

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load Dataset
data = load_iris()
x = data.data
y = data.target

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train Model
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier.fit(x_train, y_train)

# Accuracy
y_pred = svm_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Iris Flower Prediction using SVC")

st.write(f"Model Accuracy: {accuracy:.2f}")

# User Inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict"):
    input_data = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    prediction = svm_classifier.predict(input_data)

    flower_names = data.target_names
    result = flower_names[prediction[0]]

    st.success(f"Predicted Flower: {result}")