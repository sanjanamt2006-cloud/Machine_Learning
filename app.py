import streamlit as st
import pandas as pd

st.set_page_config(page_title="Machine Learning Dashboard")

st.title("Dataset Test")

try:
    ex1 = pd.read_csv("data/ex1data1[1].csv")

    st.success("Dataset Loaded Successfully!")

    st.write("Shape:", ex1.shape)

    st.dataframe(ex1.head())

except Exception as e:
    st.error(f"Error: {e}")
