import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Machine Learning Dashboard",
    layout="wide"
)

st.title("🤖 Andrew Ng Machine Learning Dashboard")

try:

    ex1 = pd.read_csv("data/ex1data1[1].csv")
    ex2 = pd.read_csv("data/ex2data1[1].csv")
    X = pd.read_csv("data/ex3data1-x[1].csv")
    y = pd.read_csv("data/ex3data1-y[1].csv")
    theta1 = pd.read_csv("data/ex3data1-theta1[1].csv")
    theta2 = pd.read_csv("data/ex3data1-theta2[1].csv")

    st.success("All datasets loaded successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric("Linear Regression", ex1.shape[0])
    col2.metric("Logistic Regression", ex2.shape[0])
    col3.metric("Digit Samples", X.shape[0])

    st.subheader("Dataset Shapes")

    st.write("ex1data1:", ex1.shape)
    st.write("ex2data1:", ex2.shape)
    st.write("Digit Features:", X.shape)
    st.write("Digit Labels:", y.shape)
    st.write("Theta1:", theta1.shape)
    st.write("Theta2:", theta2.shape)

except Exception as e:
    st.error(f"Error: {e}")
