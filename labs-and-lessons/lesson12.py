# Exercise 1
def ex1():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv(f"{os.environ['DATASET_DIRECTORY']}CarRanking_train.csv")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df)

    def getSummary(df, attribute):
        averagesSeries = df.groupby([attribute])["Rank"].mean()
        attributeDict = {}
        min = averagesSeries.min()
        max = averagesSeries.max()
        attributeDict["attribute"] = attribute
        attributeDict["min"] = min
        attributeDict["max"] = max
        attributeDict["range"] = max - min

        averagesDf = averagesSeries.to_frame()
        levels = list(averagesDf.index)

        levelPartWorths = []
        for i in range(0, len(levels)):
            averagePartWorth = averagesSeries[i]
            levelName = levels[i]
            levelPartWorths.append({levelName: averagePartWorth})
        attributeDict["partWorths"] = levelPartWorths
        return attributeDict

    def getImportances(attributeSummaries):
        ranges = []
        for i in range(0, len(attributeSummaries)):
            ranges.append(attributeSummaries[i]["range"])
        rangeSum = sum(ranges)

        for i in range(0, len(attributeSummaries)):
            importance = attributeSummaries[i]["range"] / rangeSum
            attributeSummaries[i]["importance"] = importance
        return attributeSummaries

    attributeNames = ["Safety", "Fuel", "Accessories"]
    attributeSummaries = []
    for i in range(0, len(attributeNames)):
        attributeInfo = getSummary(df, attributeNames[i])
        attributeSummaries.append(attributeInfo)

    attributeSummaries = getImportances(attributeSummaries)
    print(attributeSummaries)

    def plotImportances(attributeSummaries):
        X = []
        y = []
        for i in range(0, len(attributeSummaries)):
            X.append(attributeSummaries[i]["attribute"])
            y.append(attributeSummaries[i]["importance"])

        plt.bar(X, y)
        plt.title("Importances")
        plt.xticks(rotation=75)
        plt.show()

    def plotLevels(attributeSummaries):
        X = []
        y = []
        for i in range(0, len(attributeSummaries)):
            attribute = attributeSummaries[i]["attribute"]
            partWorths = attributeSummaries[i]["partWorths"]
            for j in range(0, len(partWorths)):
                obj = partWorths[j]
                key = list(obj.keys())[0]
                val = list(obj.values())[0]
                label = attribute + "_" + key
                X.append(label)
                y.append(val)

        plt.bar(X, y)
        plt.title("Part-worths")
        plt.xticks(rotation=75)
        plt.show()

    plotImportances(attributeSummaries)
    plotLevels(attributeSummaries)

    def getUtility(attributeSummaries, attribute, level):
        for attributeSummary in attributeSummaries:
            if attributeSummary["attribute"] == attribute:
                partWorths = attributeSummary["partWorths"]
                for partWorth in partWorths:
                    key = list(partWorth.keys())[0]
                    if key == level:
                        importance = attributeSummary["importance"]
                        val = list(partWorth.values())[0]
                        return importance * val

    dfTest = pd.read_csv(f"{os.environ['DATASET_DIRECTORY']}CarRanking_test.csv")
    utilities = []
    for i in range(0, len(dfTest)):
        utilitySum = 0
        for j in range(0, len(attributeNames)):
            attribute = attributeNames[j]
            level = dfTest.iloc[i][attribute]
            utility = getUtility(attributeSummaries, attribute, level)
            utilitySum += utility
        utilities.append(utilitySum)
    dfTest["Utility"] = utilities
    print(dfTest)


# Exercise 2
def ex2():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "salaries.csv"

    # Note this has a comma separator.
    df = pd.read_csv(PATH + CSV_DATA)

    # Extract target and salary.
    x = df[["Level"]]
    y = df[["Salary"]]

    # Generate the matrix of x values which includes:
    # x^0 (which equals 1), x^1, X^2, X^3 and X^4
    polyFeatures = PolynomialFeatures(degree=5)
    x_transformed = polyFeatures.fit_transform(x)

    # Perform linear regression with the transformed data.
    linReg = LinearRegression()
    linReg.fit(x_transformed, y)

    # Estimate dependent variable for polynomial equations.
    predictions = linReg.predict(x_transformed)

    # Visualize result for Polynomial Regression.
    plt.scatter(x, y, color="blue")
    plt.plot(x, predictions, color="red")
    plt.title("Salary Prediction")
    plt.xlabel("Position Levels")
    plt.ylabel("Salary")
    plt.show()

    print(linReg.intercept_)
    print(linReg.coef_)


# Exercise 3
def ex3():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "salaries.csv"

    # Note this has a comma separator.
    df = pd.read_csv(PATH + CSV_DATA)

    # Extract target and salary.
    x = df[["Level"]]
    y = df[["Salary"]]

    # Generate the matrix of x values which includes:
    # x^0 (which equals 1), x^1, X^2, X^3 and X^4
    polyFeatures = PolynomialFeatures(degree=4)
    x_transformed = polyFeatures.fit_transform(x)
    print(x)
    print(x_transformed)

    # Perform linear regression with the transformed data.
    linReg = LinearRegression()
    linReg.fit(x_transformed, y)

    # Estimate dependent variable for polynomial equations.
    predictions = linReg.predict(x_transformed)

    # Visualize result for Polynomial Regression.
    plt.scatter(x, y, color="blue")
    plt.plot(x, predictions, color="red")
    plt.title("Salary Prediction")
    plt.xlabel("Position Levels")
    plt.ylabel("Salary")
    plt.show()

    print(linReg.intercept_)
    print(linReg.coef_)


# Exercise 4
def ex4():
    from sklearn.preprocessing import PolynomialFeatures
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "salaries.csv"

    # Note this has a comma separator.
    df = pd.read_csv(PATH + CSV_DATA)

    # Extract target and salary.
    x = df[["Level"]]
    y = df[["Salary"]]

    # Generate the matrix of x values which includes:
    # x^0 (which equals 1), x^1, X^2, X^3 and X^4
    polyFeatures = PolynomialFeatures(degree=3)
    x_transformed = polyFeatures.fit_transform(x)
    print(x_transformed)

    linReg = LinearRegression()
    linReg.fit(x_transformed, y)

    # Estimate dependent variable for polynomial equations.
    predictions = linReg.predict(x_transformed)

    # Visualize result for Polynomial Regression.
    plt.scatter(x, y, color="blue")
    plt.plot(x, predictions, color="red")
    plt.title("Salary Prediction")
    plt.xlabel("Position Levels")
    plt.ylabel("Salary")
    plt.show()

    print(linReg.intercept_)
    print(linReg.coef_)


# Exercise 5
def ex5():
    # Fitting Polynomial Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures

    poly_reg = PolynomialFeatures(degree=4)

    # This line of code generates
    # ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
    # for all x values.
    X_poly = poly_reg.fit_transform([[4]])
    print(X_poly)


# Exercise 6
def ex6():
    def scoreData(xarray, poly_reg):
        # ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
        # The polynomial tranform is an array the variable raised to
        # different powers.
        print("\nFeature Names: ")
        print(poly_reg.get_feature_names_out())

        # Show the polynomial features.
        x = poly_reg.fit_transform(xarray)[0]

        # Intercept from linear regression.
        b = 135486.37366816

        # Coefficients from linear regression.
        b1 = -143427.12888394
        b2 = 68157.93948411
        b3 = -11608.84599445
        b4 = 709.02332493

        # Multiply linear regression coefficients by polynomial features.
        predict = b * x[0] + b1 * x[1] + b2 * x[2] + b3 * x[3] + b4 * x[4]
        print("\n***Prediction: " + str(predict))

    # Fitting Polynomial Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures

    poly_reg = PolynomialFeatures(degree=4)
    poly_reg.fit_transform([[3]])
    scoreData([[3]], poly_reg)


# Exercise 8
def ex8():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "usCityData.csv"
    df = pd.read_csv(PATH + CSV_DATA)
    X = df[["lstat"]]
    y = df[["medv"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True
    )

    # Polynomial Regression-nth order
    plt.scatter(X_test, y_test, s=10, alpha=0.3)

    for degree in [1, 2, 3, 4, 5, 6, 7]:
        # This line of code generates
        # ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
        # for all x values.
        poly_reg = PolynomialFeatures(degree=degree)
        transformed_x = poly_reg.fit_transform(X_train)

        # Generate intercept and coefficient.
        linReg = LinearRegression()
        linReg.fit(transformed_x, y_train)

        # Generate predictions and evaluate R^2.
        transformed_x = poly_reg.fit_transform(X_test)
        predictions = linReg.predict(transformed_x)
        rmse = np.sqrt(np.sum((y_test - predictions) ** 2) / len(predictions))

        # Print the predictions and RMSE
        print(f"\nDegree: {degree}, Predictions:\n{predictions}")
        print(f"\nDegree: {degree}, RMSE: {rmse}")

        plt.plot(
            X_test,
            predictions,
            label="degree %d" % degree
            + "; $R^2$: %.2f" % linReg.score(transformed_x, y_test),
        )

    # Show the plots for all polynomial equations.
    plt.legend(loc="upper right")
    plt.xlabel("LSTAT ")
    plt.ylabel("MEDV")
    plt.title("Fit for Polynomial Models")
    plt.show()


# Exercise 9
def ex9():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "usCityData.csv"
    df = pd.read_csv(PATH + CSV_DATA)
    X = df[["lstat"]]
    y = df[["medv"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True
    )
    # Draw scatter plot of the test set.
    plt.scatter(X_test, y_test)

    # Train the model.
    degree = 5
    poly_reg = PolynomialFeatures(degree=degree)
    transformed_x = poly_reg.fit_transform(X_train)
    linReg = LinearRegression()

    # Generate linear regression model with transformed x.
    linReg.fit(transformed_x, y_train)

    # Sort the test data.
    sortedTestDf = X_test[["lstat"]]
    sortedTestDf["medv"] = y_test[["medv"]]
    sortedTestDf = sortedTestDf.sort_values(["lstat"], ascending=[True])

    # Generate predictions with sorted test data.
    transformed_x_test = poly_reg.fit_transform(sortedTestDf[["lstat"]])
    predictions = linReg.predict(transformed_x_test)
    print(predictions)

    # Visualize the prediction line against the scatter plot.
    plt.plot(
        sortedTestDf[["lstat"]],
        predictions,
        label="degree %d" % degree
        + "; $R^2$: %.2f" % linReg.score(transformed_x_test, sortedTestDf["medv"]),
    )

    plt.legend(loc="upper right")
    plt.xlabel("Test LSTAT Data")
    plt.ylabel("Predicted Price")
    plt.title("Variance Explained with Varying Polynomial")
    plt.show()

    def scoreData(xarray, poly_reg, linReg):
        # ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
        # The polynomial tranform is an array the variable raised to
        # different powers.
        print("\nFeature Names: ")
        print(poly_reg.get_feature_names_out())

        # Show the polynomial features.
        x = poly_reg.fit_transform(xarray)[0]

        # Intercept from linear regression.
        b = linReg.intercept_

        # Coefficients from linear regression.
        coefficients = linReg.coef_.tolist()[0]

        # Multiply linear regression coefficients by polynomial features.
        predict = (
            b * x[0]
            + coefficients[1] * x[1]
            + coefficients[2] * x[2]
            + coefficients[3] * x[3]
            + coefficients[4] * x[4]
        )
        # print("\n***R2: " + str(linReg.score(transformed_x_test, sortedTestDf["medv"])))
        print("\n***Prediction: " + str(predict))

    scoreData(transformed_x_test, poly_reg, linReg)


# Exercise 10
def ex10():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "usCityData.csv"
    df = pd.read_csv(PATH + CSV_DATA)
    X = df[["lstat"]]
    y = df[["medv"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True
    )
    # Draw scatter plot of the test set.
    plt.scatter(X_test, y_test)

    # Train the model.
    degree = 5
    poly_reg = PolynomialFeatures(degree=degree)
    transformed_x = poly_reg.fit_transform(X_train)
    linReg = LinearRegression()

    # Generate linear regression model with transformed x.
    linReg.fit(transformed_x, y_train)

    # Sort the test data.
    sortedTestDf = X_test[["lstat"]]
    sortedTestDf["medv"] = y_test[["medv"]]
    sortedTestDf = sortedTestDf.sort_values(["lstat"], ascending=[True])

    # Generate predictions with sorted test data.
    transformed_x_test = poly_reg.fit_transform(sortedTestDf[["lstat"]])
    predictions = linReg.predict(transformed_x_test)
    print(predictions)

    # Visualize the prediction line against the scatter plot.
    plt.plot(
        sortedTestDf[["lstat"]],
        predictions,
        label="degree %d" % degree
        + "; $R^2$: %.2f" % linReg.score(transformed_x_test, sortedTestDf["medv"]),
    )

    plt.legend(loc="upper right")
    plt.xlabel("Test LSTAT Data")
    plt.ylabel("Predicted Price")
    plt.title("Variance Explained with Varying Polynomial")
    plt.show()

    def scoreData(xarray, poly_reg, linReg):
        # ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
        # The polynomial tranform is an array the variable raised to
        # different powers.
        print("\nFeature Names: ")
        print(poly_reg.get_feature_names_out())

        # Show the polynomial features.
        x = poly_reg.fit_transform(xarray)[0]

        # Intercept from linear regression.
        b = linReg.intercept_

        # Coefficients from linear regression.
        coefficients = linReg.coef_.tolist()[0]

        # Multiply linear regression coefficients by polynomial features.
        predict = (
            b * x[0]
            + coefficients[1] * x[1]
            + coefficients[2] * x[2]
            + coefficients[3] * x[3]
            + coefficients[4] * x[4]
        )
        # print("\n***R2: " + str(linReg.score(transformed_x_test, sortedTestDf["medv"])))
        print("\n***Prediction: " + str(predict))

    scoreData([[2.88]], poly_reg, linReg)


# Exercise 11
def ex11():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "usCityData.csv"
    df = pd.read_csv(PATH + CSV_DATA)
    X = df[["lstat"]]
    y = df[["medv"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True
    )
    # Draw scatter plot of the test set.
    plt.scatter(X_test, y_test)

    # Train the model.
    degree = 5
    poly_reg = PolynomialFeatures(degree=degree)
    transformed_x = poly_reg.fit_transform(X_train)
    linReg = LinearRegression()

    # Generate linear regression model with transformed x.
    linReg.fit(transformed_x, y_train)

    # Sort the test data.
    sortedTestDf = X_test[["lstat"]]
    sortedTestDf["medv"] = y_test[["medv"]]
    sortedTestDf = sortedTestDf.sort_values(["lstat"], ascending=[True])

    # Generate predictions with sorted test data.
    transformed_x_test = poly_reg.fit_transform(sortedTestDf[["lstat"]])
    predictions = linReg.predict(transformed_x_test)
    rmse = np.sqrt(np.sum((y_test - predictions) ** 2) / len(predictions))
    print(f"RMSE: {rmse}")
    print(predictions)

    # Visualize the prediction line against the scatter plot.
    plt.plot(
        sortedTestDf[["lstat"]],
        predictions,
        label="degree %d" % degree
        + "; $R^2$: %.2f" % linReg.score(transformed_x_test, sortedTestDf["medv"]),
    )

    plt.legend(loc="upper right")
    plt.xlabel("Test LSTAT Data")
    plt.ylabel("Predicted Price")
    plt.title("Variance Explained with Varying Polynomial")
    plt.show()


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    # ex5()
    # ex6()
    # ex8()
    # ex9()
    # ex10()
    ex11()


if __name__ == "__main__":
    main()
