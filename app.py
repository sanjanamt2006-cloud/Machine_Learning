import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import random

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Andrew Ng ML Dashboard",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

[data-testid="metric-container"] {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():
    ex1 = pd.read_csv(
        "data/ex1data1.csv",
        header=None,
        names=["Population", "Profit"]
    )

    ex2 = pd.read_csv(
        "data/ex2data1.csv",
        header=None,
        names=["Exam1", "Exam2", "Admitted"]
    )

    X = pd.read_csv(
        "data/ex3data1-x.csv",
        header=None
    )

    y = pd.read_csv(
        "data/ex3data1-y.csv",
        header=None,
        names=["Digit"]
    )

    theta1 = pd.read_csv(
        "data/ex3data1-theta1.csv",
        header=None
    )

    theta2 = pd.read_csv(
        "data/ex3data1-theta2.csv",
        header=None
    )

    return ex1, ex2, X, y, theta1, theta2


try:
    ex1, ex2, X, y, theta1, theta2 = load_data()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🤖 Navigation")

page = st.sidebar.radio(
    "Choose Analysis",
    [
        "Overview",
        "Linear Regression",
        "Logistic Regression",
        "Digit Recognition",
        "Neural Network"
    ]
)

# -----------------------------------
# OVERVIEW
# -----------------------------------

if page == "Overview":

    st.title("🤖 Andrew Ng Machine Learning Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Datasets", "6")
    col2.metric("Digit Samples", X.shape[0])
    col3.metric("Features", X.shape[1])

    st.markdown("---")

    summary = pd.DataFrame({
        "Dataset": [
            "Linear Regression",
            "Logistic Regression",
            "Digit Features",
            "Digit Labels",
            "Theta1",
            "Theta2"
        ],
        "Rows": [
            ex1.shape[0],
            ex2.shape[0],
            X.shape[0],
            y.shape[0],
            theta1.shape[0],
            theta2.shape[0]
        ],
        "Columns": [
            ex1.shape[1],
            ex2.shape[1],
            X.shape[1],
            y.shape[1],
            theta1.shape[1],
            theta2.shape[1]
        ]
    })

    st.subheader("Dataset Summary")
    st.dataframe(summary, use_container_width=True)

# -----------------------------------
# LINEAR REGRESSION
# -----------------------------------

elif page == "Linear Regression":

    st.title("📈 Linear Regression Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(ex1.head())

    fig = px.scatter(
        ex1,
        x="Population",
        y="Profit",
        title="Population vs Profit"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(
        ex1,
        x="Profit",
        title="Profit Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    corr = ex1.corr(numeric_only=True)

    fig3 = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------
# LOGISTIC REGRESSION
# -----------------------------------

elif page == "Logistic Regression":

    st.title("🎯 Logistic Regression Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(ex2.head())

    fig = px.scatter(
        ex2,
        x="Exam1",
        y="Exam2",
        color=ex2["Admitted"].astype(str),
        title="Admission Classification"
    )

    st.plotly_chart(fig, use_container_width=True)

    admission_counts = ex2["Admitted"].value_counts()

    fig2 = px.pie(
        values=admission_counts.values,
        names=admission_counts.index,
        title="Admission Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------
# DIGIT RECOGNITION
# -----------------------------------

elif page == "Digit Recognition":

    st.title("🔢 Digit Recognition")

    col1, col2, col3 = st.columns(3)

    col1.metric("Images", X.shape[0])
    col2.metric("Features", X.shape[1])
    col3.metric("Classes", len(y["Digit"].unique()))

    st.markdown("---")

    if st.button("Show Random Digit"):

        idx = random.randint(0, X.shape[0] - 1)

        image = X.iloc[idx].values.reshape(20, 20)

        st.image(
            image,
            caption=f"Digit Label: {y.iloc[idx, 0]}",
            width=250
        )

    digit_counts = y["Digit"].value_counts().sort_index()

    fig = px.bar(
        x=digit_counts.index,
        y=digit_counts.values,
        labels={"x": "Digit", "y": "Count"},
        title="Digit Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Visualization")

    sample_size = min(1000, len(X))

    sample = X.sample(sample_size, random_state=42)

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(sample)

    pca_df = pd.DataFrame(
        pca_result,
        columns=["PC1", "PC2"]
    )

    fig2 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA Projection"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------
# NEURAL NETWORK
# -----------------------------------

elif page == "Neural Network":

    st.title("🧠 Neural Network Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Input Layer", 400)
    col2.metric("Hidden Layer", 25)
    col3.metric("Output Layer", 10)

    st.markdown("---")

    st.subheader("Theta1 Heatmap")

    fig1 = px.imshow(
        theta1,
        aspect="auto"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Theta2 Heatmap")

    fig2 = px.imshow(
        theta2,
        aspect="auto"
    )

    st.plotly_chart(fig2, use_container_width=True)

    weights = np.concatenate([
        theta1.values.flatten(),
        theta2.values.flatten()
    ])

    fig3 = px.histogram(
        weights,
        nbins=50,
        title="Weight Distribution"
    )

    st.plotly_chart(fig3, use_container_width=True)
