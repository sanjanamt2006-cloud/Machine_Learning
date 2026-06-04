```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import random

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Andrew Ng ML Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>
.main{
background-color:#0E1117;
}

h1,h2,h3{
color:#00CC96;
}

[data-testid="metric-container"]{
background-color:#262730;
padding:15px;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# LOAD DATA
# --------------------------------

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


ex1, ex2, X, y, theta1, theta2 = load_data()

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("🤖 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Linear Regression",
        "Logistic Regression",
        "Digit Recognition",
        "Neural Network"
    ]
)

# --------------------------------
# OVERVIEW
# --------------------------------

if page == "Overview":

    st.title("🤖 Andrew Ng Machine Learning Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric("Datasets", 6)
    c2.metric("Digit Images", X.shape[0])
    c3.metric("Features", X.shape[1])

    st.markdown("---")

    st.subheader("Dataset Information")

    summary = pd.DataFrame({
        "Dataset":[
            "Linear Regression",
            "Logistic Regression",
            "Digit Features",
            "Digit Labels",
            "Theta1",
            "Theta2"
        ],
        "Rows":[
            ex1.shape[0],
            ex2.shape[0],
            X.shape[0],
            y.shape[0],
            theta1.shape[0],
            theta2.shape[0]
        ],
        "Columns":[
            ex1.shape[1],
            ex2.shape[1],
            X.shape[1],
            y.shape[1],
            theta1.shape[1],
            theta2.shape[1]
        ]
    })

    st.dataframe(summary, use_container_width=True)

# --------------------------------
# LINEAR REGRESSION
# --------------------------------

elif page == "Linear Regression":

    st.title("📈 Linear Regression Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(ex1.head())

    st.subheader("Population vs Profit")

    fig = px.scatter(
        ex1,
        x="Population",
        y="Profit",
        trendline="ols"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Profit Distribution")

    fig2 = px.histogram(
        ex1,
        x="Profit"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation")

    corr = ex1.corr()

    fig3 = px.imshow(
        corr,
        text_auto=True
    )

    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------
# LOGISTIC REGRESSION
# --------------------------------

elif page == "Logistic Regression":

    st.title("🎯 Logistic Regression Analysis")

    st.dataframe(ex2.head())

    fig = px.scatter(
        ex2,
        x="Exam1",
        y="Exam2",
        color=ex2["Admitted"].astype(str)
    )

    st.plotly_chart(fig, use_container_width=True)

    admission = ex2["Admitted"].value_counts()

    fig2 = px.pie(
        values=admission.values,
        names=admission.index
    )

    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------
# DIGIT RECOGNITION
# --------------------------------

elif page == "Digit Recognition":

    st.title("🔢 Digit Recognition Dashboard")

    c1,c2,c3 = st.columns(3)

    c1.metric("Images", X.shape[0])
    c2.metric("Features", X.shape[1])
    c3.metric("Classes", len(y["Digit"].unique()))

    st.markdown("---")

    if st.button("Show Random Digit"):

        idx = random.randint(
            0,
            len(X)-1
        )

        image = X.iloc[idx].values.reshape(20,20)

        st.image(
            image,
            width=250,
            caption=f"Digit : {y.iloc[idx,0]}"
        )

    st.subheader("Digit Distribution")

    digit_counts = y["Digit"].value_counts().sort_index()

    fig = px.bar(
        x=digit_counts.index,
        y=digit_counts.values
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Visualization")

    pca = PCA(n_components=2)

    sample = X.sample(1000)

    pca_result = pca.fit_transform(sample)

    pca_df = pd.DataFrame({
        "PC1":pca_result[:,0],
        "PC2":pca_result[:,1]
    })

    fig2 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2"
    )

    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------
# NEURAL NETWORK
# --------------------------------

elif page == "Neural Network":

    st.title("🧠 Neural Network Analysis")

    st.metric("Input Layer",400)
    st.metric("Hidden Layer",25)
    st.metric("Output Layer",10)

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

    st.subheader("Weight Distribution")

    fig3 = px.histogram(
        weights,
        nbins=50
    )

    st.plotly_chart(fig3, use_container_width=True)
```

