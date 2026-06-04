import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import random

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Machine Learning Dashboard", layout="wide")

st.title("🤖 Andrew Ng Machine Learning Dashboard")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():

    ex1 = pd.read_csv("data/ex1data1[1].csv")
    ex2 = pd.read_csv("data/ex2data1[1].csv")

    X = pd.read_csv("data/ex3data1-x[1].csv")
    y = pd.read_csv("data/ex3data1-y[1].csv")

    theta1 = pd.read_csv("data/ex3data1-theta1[1].csv")
    theta2 = pd.read_csv("data/ex3data1-theta2[1].csv")

    return ex1, ex2, X, y, theta1, theta2


ex1, ex2, X, y, theta1, theta2 = load_data()

# ----------------------------
# SIDEBAR MENU
# ----------------------------
menu = st.sidebar.radio(
    "Select Section",
    ["Overview", "Linear Regression", "Logistic Regression", "Digit Recognition", "Neural Network"]
)

# ----------------------------
# OVERVIEW
# ----------------------------
if menu == "Overview":

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Linear Regression Rows", ex1.shape[0])
    col2.metric("Logistic Regression Rows", ex2.shape[0])
    col3.metric("Digit Samples", X.shape[0])

    st.write("### ex1 data")
    st.dataframe(ex1.head())

    st.write("### ex2 data")
    st.dataframe(ex2.head())

# ----------------------------
# LINEAR REGRESSION
# ----------------------------
elif menu == "Linear Regression":

    st.subheader("Population vs Profit")

    fig = px.scatter(ex1, x="population", y="profit", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# LOGISTIC REGRESSION
# ----------------------------
elif menu == "Logistic Regression":

    st.subheader("Admission based on scores")

    fig = px.scatter(
        ex2,
        x="e1score",
        y="e2score",
        color=ex2["admission"].astype(str)
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# DIGIT RECOGNITION
# ----------------------------
elif menu == "Digit Recognition":

    st.subheader("Random Digit Viewer")

    if st.button("Show Digit"):

        idx = random.randint(0, len(X)-1)

        img = X.iloc[idx].values.reshape(20, 20)

        st.image(img, caption=f"Label: {y.iloc[idx,0]}", width=200)

    st.subheader("Digit Distribution")

    st.bar_chart(y.value_counts())

    st.subheader("PCA Visualization")

    sample = X.sample(1000)
    pca = PCA(n_components=2)
    result = pca.fit_transform(sample)

    df = pd.DataFrame(result, columns=["PC1", "PC2"])

    st.scatter_chart(df)

# ----------------------------
# NEURAL NETWORK
# ----------------------------
elif menu == "Neural Network":

    st.subheader("Theta Visualizations")

    st.write("Theta1 shape:", theta1.shape)
    st.write("Theta2 shape:", theta2.shape)

    st.line_chart(theta1)
    st.line_chart(theta2)
