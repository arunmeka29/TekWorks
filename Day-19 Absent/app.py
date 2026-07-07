import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import sklearn with graceful handling for environments where sklearn
# may not be installed or resolvable by the editor/runner.
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingModuleSource]
    from sklearn.model_selection import train_test_split  # type: ignore[reportMissingModuleSource]
    from sklearn.metrics import (  # type: ignore[reportMissingModuleSource]
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
except Exception:
    st.error(
        "scikit-learn could not be imported. Please ensure scikit-learn is installed. "
        "You can install it with: pip install scikit-learn"
    )
    st.stop()

st.title("Telco Customer Churn Prediction")

st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload WA_Fn-UseC_-Telco-Customer-Churn.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df[['tenure']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Model Parameters")

    m = model.coef_[0][0]
    c = model.intercept_[0]

    st.write("m value:", m)
    st.write("c value:", c)

    st.subheader("Manual Prediction")

    x = st.number_input("Enter tenure value", value=5)

    z = (m * x) + c

    probability = 1 / (1 + np.exp(-z))

    st.write("z value:", z)
    st.write("Probability:", probability)

    if probability > 0.5:
        st.success("Prediction: Customer Will Churn")
    else:
        st.info("Prediction: Customer Will Not Churn")

    st.subheader("Sigmoid Curve")

    x_values = np.linspace(-10, 10, 100)
    y_values = 1 / (1 + np.exp(-x_values))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_values)
    ax.set_xlabel("z values")
    ax.set_ylabel("Sigmoid Output")
    ax.set_title("Sigmoid Function")
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    st.write(cm)

    st.subheader("Evaluation Metrics")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)