import streamlit as st
import pandas as pd
st.title("Census Data Analysis")
a=st.file_uploader("upload the DataSet")
if a:
    df=pd.read_csv(a)
    st.dataframe(df)
    st.write(df.head())