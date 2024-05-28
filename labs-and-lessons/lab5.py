# Exercise 1
def ex1():
    import pandas as pd
    import numpy as np
    import os

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    FOLDER_PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "employee_turnover.csv"
    df = pd.read_csv(FOLDER_PATH + FILE)
    print(df)

    # Separate into x and y values.
    predictorVariables = list(df.keys())
    predictorVariables.remove("turnover")
    print(predictorVariables)

    # Create X and y values.
    X = df[predictorVariables]
    y = df["turnover"]


# Exercise 2
def ex2():
    import pandas as pd
    import numpy as np
    import os

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    FOLDER_PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "employee_turnover.csv"
    df = pd.read_csv(FOLDER_PATH + FILE)
    # print(df)

    # Create X and y values.
    X = df[["experience", "age", "industry", "way"]]
    y = df["turnover"]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    # Build logistic regression model and make predictions.
    logisticModel = LogisticRegression(
        fit_intercept=True, solver="liblinear", random_state=0
    )
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)
    print(f"\nLogistic Predictions:\n{y_pred}")


# Exercise 6
def ex6():
    import pandas as pd
    import numpy as np
    import os

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    FOLDER_PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "employee_turnover.csv"
    df = pd.read_csv(FOLDER_PATH + FILE)
    # print(df)

    # Create X and y values.
    X = df[["experience", "age", "industry", "way"]]
    y = df["turnover"]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    # Build logistic regression model and make predictions.
    logisticModel = LogisticRegression(
        fit_intercept=True, solver="liblinear", random_state=0
    )
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)
    print(f"\nLogistic Predictions:\n{y_pred}")

    # Show confusion matrix and accuracy scores.
    from sklearn import metrics

    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
    print("\nAccuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(cm)


# Exercise 8
def ex8():
    import pandas as pd
    import numpy as np
    import os

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    FOLDER_PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "employee_turnover.csv"
    df = pd.read_csv(FOLDER_PATH + FILE)
    # print(df)

    # Create X and y values.
    X = df[["experience", "age", "industry", "way"]]
    y = df["turnover"]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    # Build logistic regression model and make predictions.
    logisticModel = LogisticRegression(
        fit_intercept=True, solver="liblinear", random_state=0
    )
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)
    print(f"\nLogistic Predictions:\n{y_pred}")

    # Show confusion matrix and accuracy scores.
    from sklearn import metrics

    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
    print("\nAccuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(cm)

    TN = cm[0][0]  # = 3 True Negative  (Col 0, Row 0)
    FN = cm[0][1]  # = 0 False Negative (Col 0, Row 1)
    FP = cm[1][0]  # = 2 False Positive (Col 1, Row 0)
    TP = cm[1][1]  # = 5 True Positive  (Col 1, Row 1)

    print("\nTrue Negative:  " + str(TN))
    print("False Negative: " + str(FN))
    print("False Positive: " + str(FP))
    print("True Positive:  " + str(TP))

    precision = TP / (FP + TP)
    print("\nPrecision:  " + str(round(precision, 3)))

    recall = TP / (TP + FN)
    print("Recall:     " + str(round(recall, 3)))

    F1 = 2 * ((precision * recall) / (precision + recall))
    print("F1:         " + str(round(F1, 3)))


# Exercise 11
def ex11():
    # Load libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "glass.csv"
    df = pd.read_csv(PATH + FILE)

    # Show DataFrame contents.
    print(df.head())

    # Get X values and remove target column.
    # Make copy to avoid over-writing.
    X = df.copy()
    del X["Type"]

    # Get y values
    y = df["Type"]

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Build logistic regression model and make predictions.
    logisticModel = LogisticRegression(
        fit_intercept=True, solver="liblinear", random_state=0
    )
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)
    print(f"\nLogistic Predictions:\n{y_pred}")

    # Show confusion matrix and accuracy scores.
    from sklearn import metrics

    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
    print("\nAccuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(cm)

    # Calculate precision, recall, and F1-score for each class.
    report = metrics.classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)


def main():
    ex1()
    ex2()
    ex6()
    ex8()
    ex11()


if __name__ == "__main__":
    main()
