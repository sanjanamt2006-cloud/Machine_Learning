import streamlit as st
import pandas as pd

st.title("Dataset Columns Check")

ex1 = pd.read_csv("data/ex1data1[1].csv")
ex2 = pd.read_csv("data/ex2data1[1].csv")

st.subheader("ex1 Columns")
st.write(ex1.columns.tolist())
st.dataframe(ex1.head())

st.subheader("ex2 Columns")
st.write(ex2.columns.tolist())
st.dataframe(ex2.head())
