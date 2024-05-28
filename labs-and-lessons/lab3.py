# Exercise 1
def ex1():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
            "Price",
            "Address",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(dataset.head(3))


# Exercise 2
def ex2():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
            "Price",
            "Address",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(dataset.describe())


# Exercise 3
def ex3():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
            "Price",
            "Address",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Compute the correlation matrix
    corr = dataset.iloc[:, :-1].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()


# Exercise 4, 5, and 6
def ex4():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import statsmodels.api as sm
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
            "Price",
            "Address",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)
    print(dataset.head())
    print(dataset.describe())
    X = dataset[
        [
            # "Avg. Area Income", # removed for Model C
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            # "Avg. Area Number of Bedrooms", # removed for Model B
            # "Area Population", # removed for Model C
        ]
    ].values

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)
    y = dataset["Price"].values
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


# Exercise 8
def ex8():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
            "Price",
            "Address",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

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
    y = dataset["Price"]
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

    def plotPredictionVsActual(plt, title, y_test, predictions):
        plt.scatter(y_test, predictions)
        plt.legend()
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted (Y) vs. Actual (X): " + title)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")

    def plotResidualsVsActual(plt, title, y_test, predictions):
        residuals = y_test - predictions
        print(y_test, predictions, len(y_test), len(predictions))
        plt.scatter(y_test, residuals, label="Residuals vs Actual")
        plt.xlabel("Actual")
        plt.ylabel("Residual")
        plt.title("Error Residuals (Y) vs. Actual (X): " + title)
        plt.plot([y_test.min(), y_test.max()], [0, 0], "k--")

    def plotResidualHistogram(plt, title, y_test, predictions, bins):
        residuals = y_test - predictions
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.hist(residuals, label="Residuals vs Actual", bins=bins)
        plt.title("Error Residual Frequency: " + title)

    plt.plot()

    def drawValidationPlots(title, bins, y_test, predictions):
        # Define number of rows and columns for graph display.
        plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        plt.subplot(1, 3, 1)  # Specify total rows, columns and image #`
        plotPredictionVsActual(plt, title, y_test, predictions)
        plt.subplot(1, 3, 2)  # Specify total rows, columns and image #
        plotResidualsVsActual(plt, title, y_test, predictions)
        plt.subplot(1, 3, 3)  # Specify total rows, columns and image #
        plotResidualHistogram(plt, title, y_test, predictions, bins)
        plt.show()

    BINS = 8
    TITLE = "Price"
    drawValidationPlots(TITLE, BINS, y_test, predictions)


# Exercise 10
def ex10():
    def get_price(
        actual_price,
        avg_area_income,
        avg_area_house_age,
        avg_area_number_of_rooms,
        area_population,
    ):
        price = (
            -2647000
            + 21.6681 * avg_area_income
            + 165800 * avg_area_house_age
            + 121600 * avg_area_number_of_rooms
            + 15.2785 * area_population
        )
        print("Price actual: " + str(actual_price) + "   predicted: " + str(price))

    get_price(
        894251.06863578, 6.12007262e04, 5.29969400e00, 6.23461464e00, 4.27896922e04
    )
    get_price(
        932979.36062132, 6.33808147e04, 5.34466404e00, 6.00157433e00, 4.02173336e04
    )
    get_price(
        920747.91128789, 7.12082693e04, 5.30032605e00, 6.07798886e00, 2.56963617e04
    )


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    ex8()
    # ex10()


if __name__ == "__main__":
    main()
