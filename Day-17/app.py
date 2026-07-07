import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏦 Loan Prediction App")

# ======================
# Inputs (Simple UI)
# ======================

income = st.number_input("Income")
credit_score = st.number_input("Credit Score")

# Default values (hidden)
age = 30
loan_amount = 50000
num_transactions = 10
annual_spend = 20000
day = 1
month = 1
year = 2024

city = "Hyderabad"
employment_type = "Salaried"
loan_type = "Personal"

# ======================
# Create Data
# ======================

input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'loan_amount': [loan_amount],
    'credit_score': [credit_score],
    'num_transactions': [num_transactions],
    'annual_spend': [annual_spend],
    'day': [day],
    'month': [month],
    'year': [year],
    'city': [city],
    'employment_type': [employment_type],
    'loan_type': [loan_type]
})

# Convert to dummies
input_data = pd.get_dummies(input_data)

# ======================
# 🔥 MAGIC LINE (AUTO FIX)
# ======================

# Get exact columns used during training
model_cols = model.feature_names_in_

# Match columns automatically
input_data = input_data.reindex(columns=model_cols, fill_value=0)

# ======================
# Prediction
# ======================

if st.button("Predict"):
    pred = model.predict(input_data)

    if pred[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")