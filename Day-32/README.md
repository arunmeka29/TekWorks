# Airline Passenger Forecasting Dashboard

This project is an interactive, deep-learning-powered forecasting application that predicts future airline passenger volumes using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTMs).

---

## Project Description

The application processes historical international airline passenger datasets to predict passenger numbers for upcoming months. It includes:
1. **Interactive Dashboard**: A Streamlit-based web dashboard offering real-time model evaluation, future forecasts, and exploratory data analysis.
2. **Flexible Model Architectures**: Supports training and using LSTM, GRU, and SimpleRNN architectures.
3. **Data Pipelines**: Streamlined pipelines for loading, cleaning, min-max scaling, train-test splitting, and rolling-window sequence generation (defaulting to 12 months input window size).
4. **Performance Metrics**: Reports evaluation metrics (Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error) to track model convergence and forecasting accuracy.
5. **Rich Visualizations**: Interactive graphs illustrating actual vs. predicted values, training vs. validation loss curves, and future passenger trends.

---

## Tech Stack

The following technology stack is utilized throughout the project:

### Frontend & UI
* **Streamlit**: Renders the dynamic user interface, sidebar controls, and handles application deployment.
* **HTML/CSS**: Custom CSS injection is used inside Streamlit to deliver a premium, dark-themed, glassmorphic UI layout.

### Machine Learning & Data Science
* **TensorFlow / Keras**: Used to define, build, train, compile (with Adam optimizer and MSE loss), and save/load the deep learning neural network.
* **Scikit-Learn**: Employs `MinMaxScaler` for scaling passenger features between [0, 1] and uses `mean_absolute_error` and `mean_squared_error` for accuracy evaluation.
* **Joblib**: Used for scaling-parameter persistence (`scaler.pkl`).

### Data Processing & Operations
* **Pandas**: Manages loading CSV data, time-series datetime conversions, and indexing.
* **NumPy**: Standardizes array formats and handles matrix transformations/sliding-window sequence creation.

### Data Visualization
* **Plotly**: Renders interactive, high-performance charts (Plotly Express and Graph Objects) within the web application.
* **Matplotlib**: Generates and saves static visual reports (`outputs/loss_curve.png`, `outputs/prediction.png`, and `outputs/forecast.png`).
