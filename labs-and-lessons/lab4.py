# Exercise 1
def ex1():
    from statsmodels.formula.api import ols
    from sklearn import metrics
    import pandas as pd
    import numpy as np

    x = [0.14, 0.32, 0.35, 0.35, 0.32, 0.21, 0.32, 0.31, 0.47]
    y = [0.07, 0.13, 0.33, 0.54, 0.35, 0.15, 0.17, 0.23, 0.43]
    df = pd.DataFrame(data={"target": y, "X": x})

    # Generate model to predict target using X.
    model = ols("target ~ X", data=df).fit()
    print(model.summary())
    predictions = model.predict(df["X"])
    RMSE = np.sqrt(metrics.mean_squared_error(predictions, y))
    print(RMSE)


# Exercise 2
def ex2():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(PATH + CSV_DATA)
    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)
    print(dataset.head())
    print(dataset.describe())

    X = dataset[
        [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Area Population",
        ]
    ]

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)
    y = dataset[["Price"]]

    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=5, shuffle=True)

    from sklearn.metrics import mean_squared_error
    import numpy as np

    rmses = []
    for train_index, test_index in kfold.split(X):
        # use index lists to isolate rows for train and test sets.
        # Get rows filtered by index and all columns.
        # X.loc[row number array, all columns]
        X_train = X.loc[X.index.intersection(train_index), :]
        X_test = X.loc[X.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_test = y.loc[y.index.intersection(test_index), :]

        # build model
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(predictions, y_test)
        rmse = np.sqrt(mse)
        print("RMSE: " + str(rmse))
        rmses.append(rmse)

    avgRMSE = np.mean(rmses)
    print("Average rmse: " + str(avgRMSE))


def main():
    ex1()
    ex2()


if __name__ == "__main__":
    main()
