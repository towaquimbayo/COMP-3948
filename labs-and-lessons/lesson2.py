# Exercise 1
def ex1():
    import matplotlib.pyplot as plt

    # Plot scatter of actual values.
    dataX = [0.19, 0.28, 0.35, 0.37, 0.4, 0.18]
    dataY = [0.13, 0.12, 0.35, 0.3, 0.37, 0.1]

    plt.scatter(dataX, dataY, color="green", label="Sample Data")

    # Plot prediction line.
    dataX2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dataY2 = [
        -0.13014345,
        -0.0086258962,
        0.1128916576,
        0.2344092114,
        0.3559267652,
        0.477444319,
    ]
    plt.plot(dataX2, dataY2, color="blue", label="y=1.215175538*x - 0.13014345")

    # Show average
    x3 = [0, 0.5]
    y3 = [0.228333333, 0.228333333]
    plt.plot(x3, y3, "--", color="Black", label="Average Bacteria Level")

    # Add a legend, axis labels, and title.
    plt.legend()
    plt.xlabel("Data X")
    plt.ylabel("Data Y")
    plt.title("Data Exercise 1")

    plt.show()


# Exercise 14
def ex14():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Create DataFrame.
    dataSet = {
        "days": [0.2, 0.32, 0.38, 0.41, 0.43],
        "growth": [0.1, 0.15, 0.4, 0.6, 0.44],
    }
    df = pd.DataFrame(dataSet, columns=["days", "growth"])

    # Store x and y values.
    X = df["days"]
    target = df["growth"]

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, train_size=0.8, random_state=0
    )
    print(f"X_train:\n--------\n{X_train.to_string(index=False)}")
    print(f"\ny_train:\n--------\n{y_train.to_string(index=False)}")
    print(f"\nX_test:\n-------\n{X_test.to_string(index=False)}")
    print(f"\ny_test:\n-------\n{y_test.to_string(index=False)}")


# Exercise 15
def ex15():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from statsmodels.formula.api import ols
    from sklearn import metrics
    import math

    def performSimpleRegression():
        # Initialize collection of X & Y pairs like those used in example 5.
        data = [[0.2, 0.1], [0.32, 0.15], [0.38, 0.4], [0.41, 0.6], [0.43, 0.44]]

        # Create data frame.
        dfSample = pd.DataFrame(data, columns=["X", "target"])

        # Create training set with 80% of data and test set with 20% of data.
        X_train, X_test, y_train, y_test = train_test_split(
            dfSample["X"], dfSample["target"], train_size=0.8
        )

        # Create DataFrame with test data.
        dataTrain = {"X": X_train, "target": y_train}
        dfTrain = pd.DataFrame(dataTrain, columns=["X", "target"])

        # Generate model to predict target using X.
        model = ols("target ~ X", data=dfTrain).fit()
        y_prediction = model.predict(X_test)

        # Present X_test, y_test, y_predict and error sum of squares.
        data = {"X_test": X_test, "y_test": y_test, "y_prediction": y_prediction}
        dfResult = pd.DataFrame(data, columns=["X_test", "y_test", "y_prediction"])
        dfResult["y_test - y_pred"] = dfResult["y_test"] - dfResult["y_prediction"]
        dfResult["(y_test - y_pred)^2"] = (
            dfResult["y_test"] - dfResult["y_prediction"]
        ) ** 2

        # Present X_test, y_test, y_predict and error sum of squares.
        print(dfResult)

        # Manually calculate the deviation between actual and predicted values.
        rmse = math.sqrt(dfResult["(y_test - y_pred)^2"].sum() / len(dfResult))
        print(
            "RMSE is average deviation between actual and predicted values: "
            + str(rmse)
        )

        # Show faster way to calculate deviation between actual and predicted values.
        rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
        print("The automated root mean square error calculation is: " + str(rmse2))

    performSimpleRegression()


def main():
    ex1()
    ex14()
    ex15()


if __name__ == "__main__":
    main()
