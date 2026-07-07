import streamlit as st
import requests

st.title("Hierarchical Clustering App")

height = st.number_input("Enter Height")
weight = st.number_input("Enter Weight")

if st.button("Predict Cluster"):

    url = "http://127.0.0.1:8000/predict"

    params = {
        "height": height,
        "weight": weight
    }

    response = requests.post(url, params=params)

    result = response.json()

    st.success(f"Cluster: {result['cluster']}")