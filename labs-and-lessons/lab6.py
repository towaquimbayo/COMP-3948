# Exercise 1
def ex1():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE
    from sklearn import metrics
    import statsmodels.api as sm
    import numpy as np
    import os

    # Read the data:
    df = pd.read_csv(f"{os.environ['DATASET_DIRECTORY']}USA_Housing.csv")

    # Separate the target and independent variable
    X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
    del X["Price"]  # Delete target variable.
    del X["Address"]  # Delete address column as it is not a numerical value.

    # Target variable
    y = df["Price"]

    # Create the object of the model
    model = LinearRegression()

    # Specify the number of  features to select
    rfe = RFE(model, n_features_to_select=5)

    # fit the model
    rfe = rfe.fit(X, y)
    # Please uncomment the following lines to see the result
    print("\n\nFEATUERS SELECTED\n\n")
    print(rfe.support_)

    columns = list(X.keys())
    for i in range(0, len(columns)):
        if rfe.support_[i]:
            print(columns[i])

    # Now we can build our model with RFE selected features.
    X = df[
        [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
        ]
    ]

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())

    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )


# Exercise 2
def ex2():
    import pandas as pd
    from sklearn.feature_selection import f_regression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import statsmodels.api as sm
    import numpy as np
    import os

    # Read the data:
    df = pd.read_csv(f"{os.environ['DATASET_DIRECTORY']}USA_Housing.csv")

    # Separate the target and independent variable
    X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
    del X["Price"]  # Delete target variable.
    del X["Address"]  # Delete address column as it is not a numerical value.

    # Target variable
    y = df["Price"]

    #  f_regression returns F statistic for each feature.
    ffs = f_regression(X, y)

    featuresDf = pd.DataFrame()
    for i in range(0, len(X.columns)):
        featuresDf = featuresDf._append(
            {"feature": X.columns[i], "ffs": ffs[0][i]}, ignore_index=True
        )
    featuresDf = featuresDf.sort_values(by=["ffs"])
    print(featuresDf)

    # Now we can build our model with forward selected features.
    X = df[
        [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Area Population",
        ]
    ]

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())

    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )


# Exercise 4
def ex4():
    from crucio import SCUT
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        accuracy_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "healthcare-dataset-stroke-data.csv"
    data = pd.read_csv(PATH + FILE)

    def showYPlots(y_train, y_test, title):
        print("\n ***" + title)
        plt.subplots(1, 2)

        plt.subplot(1, 2, 1)
        plt.hist(y_train)
        plt.title("Train Y: " + title)

        plt.subplot(1, 2, 2)
        plt.hist(y_test)
        plt.title("Test Y: " + title)
        plt.show()

    def evaluate_model(X_test, y_test, y_train, model, title):
        showYPlots(y_train, y_test, title)

        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        print(cm)
        precision = precision_score(y_test, preds, average="binary")
        print("Precision: " + str(precision))

        recall = recall_score(y_test, preds, average="binary")
        print("Recall:    " + str(recall))

        accuracy = accuracy_score(y_test, preds)
        print("Accuracy:    " + str(accuracy))

        f1_score = 2 * ((precision * recall) / (precision + recall))
        print("F1 Score:    " + str(f1_score))

    # Inspect data.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(data.head())
    print(data.describe())

    # Impute missing bmi values with average BMI value.
    averageBMI = np.mean(data["bmi"])
    data["bmi"] = data["bmi"].replace(np.nan, averageBMI)
    print(data.describe())

    def getTrainAndTestData(data):
        X = data[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]]
        y = data["stroke"]

        # Split the data into train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = getTrainAndTestData(data)

    scalerX = StandardScaler()
    X_train_scaled = scalerX.fit_transform(X_train)  # Fit and transform.
    X_test_scaled = scalerX.transform(X_test)  # Transform only.

    # Build logistic regressor and evaluate model.
    clf = LogisticRegression(solver="newton-cg", max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    evaluate_model(X_test_scaled, y_test, y_train, clf, "Before SCUT")

    # Boost representation of minority in training data with SCUT.
    dfTrain = X_train.copy()
    dfTrain["stroke"] = y_train
    scut = SCUT()
    df_scut = scut.balance(dfTrain, "stroke")

    # Adjust y_train and X_train with better represented minority.
    y_train = df_scut["stroke"]
    X_train = df_scut
    del df_scut["stroke"]

    X_train_scaled = scalerX.fit_transform(X_train)  # Fit and transform.

    showYPlots(y_train, y_test, "After SCUT")

    # Perform logistic regression with SCUT-treated train data and
    # evaluate with untouched test data.
    clf = LogisticRegression(solver="newton-cg", max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    evaluate_model(X_test_scaled, y_test, y_train, clf, "After SCUT")


# Exercise 5
def ex5():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        accuracy_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "framingham_v2.csv"
    data = pd.read_csv(PATH + FILE)

    def showYPlots(y_train, y_test, title):
        print("\n ***" + title)
        plt.subplots(1, 2)

        plt.subplot(1, 2, 1)
        plt.hist(y_train)
        plt.title("Train Y: " + title)

        plt.subplot(1, 2, 2)
        plt.hist(y_test)
        plt.title("Test Y: " + title)
        plt.show()

    def evaluate_model(X_test, y_test, y_train, model, title):
        showYPlots(y_train, y_test, title)

        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        print(cm)
        precision = precision_score(y_test, preds, average="binary")
        print("Precision: " + str(precision))

        recall = recall_score(y_test, preds, average="binary")
        print("Recall:    " + str(recall))

        accuracy = accuracy_score(y_test, preds)
        print("Accuracy:    " + str(accuracy))

        f1_score = 2 * ((precision * recall) / (precision + recall))
        print("F1 Score:    " + str(f1_score))

    # Inspect data.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(data.head())
    print(data.describe())

    def getTrainAndTestData(data):
        X = data[
            [
                "male",
                "age",
                "education",
                "currentSmoker",
                "cigsPerDay",
                "BPMeds",
                "prevalentStroke",
                "prevalentHyp",
                "diabetes",
                "totChol",
                "sysBP",
                "diaBP",
                "BMI",
                "heartRate",
                "glucose",
            ]
        ]
        y = data["TenYearCHD"]

        # Split the data into train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = getTrainAndTestData(data)

    # Build logistic regressor and evaluate model.
    clf = LogisticRegression(solver="newton-cg", max_iter=1000)
    clf.fit(X_train, y_train)
    evaluate_model(X_test, y_test, y_train, clf, "Before SCUT")

    from crucio import SCUT

    # Boost representation of minority in training data with SCUT.
    dfTrain = X_train.copy()
    dfTrain["TenYearCHD"] = y_train
    scut = SCUT()
    df_scut = scut.balance(dfTrain, "TenYearCHD")

    # Adjust y_train and X_train with better represented minority.
    y_train = df_scut["TenYearCHD"]
    X_train = df_scut
    del df_scut["TenYearCHD"]

    showYPlots(y_train, y_test, "After SCUT")

    # Perform logistic regression with SCUT-treated train data and
    # evaluate with untouched test data.
    clf = LogisticRegression(solver="newton-cg", max_iter=1000)
    clf.fit(X_train, y_train)
    evaluate_model(X_test, y_test, y_train, clf, "After SCUT")


# Exercise 6
def ex6():
    from crucio import SCUT
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        accuracy_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "framingham_v2.csv"
    data = pd.read_csv(PATH + FILE)

    def showYPlots(y_train, y_test, title):
        print("\n ***" + title)
        plt.subplots(1, 2)

        plt.subplot(1, 2, 1)
        plt.hist(y_train)
        plt.title("Train Y: " + title)

        plt.subplot(1, 2, 2)
        plt.hist(y_test)
        plt.title("Test Y: " + title)
        plt.show()

    def evaluate_model(X_test, y_test, y_train, model, title):
        showYPlots(y_train, y_test, title)

        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        print(cm)
        precision = precision_score(y_test, preds, average="binary")
        print("Precision: " + str(precision))

        recall = recall_score(y_test, preds, average="binary")
        print("Recall:    " + str(recall))

        accuracy = accuracy_score(y_test, preds)
        print("Accuracy:    " + str(accuracy))

        f1_score = 2 * ((precision * recall) / (precision + recall))
        print("F1 Score:    " + str(f1_score))

    # Inspect data.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(data.head())
    print(data.describe())

    def getTrainAndTestData(data):
        X = data[
            [
                "male",
                "age",
                "education",
                "currentSmoker",
                "cigsPerDay",
                "BPMeds",
                "prevalentStroke",
                "prevalentHyp",
                "diabetes",
                "totChol",
                "sysBP",
                "diaBP",
                "BMI",
                "heartRate",
                "glucose",
            ]
        ]
        y = data["TenYearCHD"]

        # Split the data into train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = getTrainAndTestData(data)

    scalerX = RobustScaler()
    X_train_scaled = scalerX.fit_transform(X_train)  # Fit and transform.
    X_test_scaled = scalerX.transform(X_test)  # Transform only.

    # Build logistic regressor and evaluate model.
    clf = LogisticRegression(solver="newton-cg", max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    evaluate_model(X_test_scaled, y_test, y_train, clf, "Before SCUT")

    # Boost representation of minority in training data with SCUT.
    dfTrain = X_train.copy()
    dfTrain["TenYearCHD"] = y_train
    scut = SCUT()
    df_scut = scut.balance(dfTrain, "TenYearCHD")

    # Adjust y_train and X_train with better represented minority.
    y_train = df_scut["TenYearCHD"]
    X_train = df_scut
    del df_scut["TenYearCHD"]

    X_train_scaled = scalerX.fit_transform(X_train)  # Fit and transform.

    showYPlots(y_train, y_test, "After SCUT")

    # Perform logistic regression with SCUT-treated train data and
    # evaluate with untouched test data.
    clf = LogisticRegression(solver="newton-cg", max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    evaluate_model(X_test_scaled, y_test, y_train, clf, "After SCUT")


def main():
    # ex1()
    # ex2()
    # ex4()
    # ex5()
    ex6()


if __name__ == "__main__":
    main()
