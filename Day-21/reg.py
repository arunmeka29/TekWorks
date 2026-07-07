import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title
st.title("Decision Tree Regression using Iris Dataset")

# Load Iris dataset
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Show dataset
st.subheader("Iris Dataset")
st.dataframe(data.head())

# Input features
X = data[['sepal length (cm)', 'sepal width (cm)']]

# Target variable
y = data['petal length (cm)']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sidebar option
max_depth = st.sidebar.slider("Select Max Depth", 1, 10, 3)

# Create model
model = DecisionTreeRegressor(max_depth=max_depth)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Error calculation
mse = mean_squared_error(y_test, y_pred)

# Display error
st.subheader("Mean Squared Error")
st.write(mse)

# Show predictions
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.subheader("Predicted Results")
st.dataframe(results)

# Plot graph
fig, ax = plt.subplots(figsize=(8,5))

ax.plot(y_test.values, label="Actual")
ax.plot(y_pred, label="Predicted")

ax.set_xlabel("Test Samples")
ax.set_ylabel("Petal Length")
ax.set_title("Actual vs Predicted Values")

ax.legend()

st.pyplot(fig)