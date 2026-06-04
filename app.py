ex1 = pd.read_csv(
    "data/ex1data1[1].csv",
    header=None,
    names=["Population", "Profit"]
)

ex2 = pd.read_csv(
    "data/ex2data1[1].csv",
    header=None,
    names=["Exam1", "Exam2", "Admitted"]
)

X = pd.read_csv(
    "data/ex3data1-x[1].csv",
    header=None
)

y = pd.read_csv(
    "data/ex3data1-y[1].csv",
    header=None,
    names=["Digit"]
)

theta1 = pd.read_csv(
    "data/ex3data1-theta1[1].csv",
    header=None
)

theta2 = pd.read_csv(
    "data/ex3data1-theta2[1].csv",
    header=None
)
