# Exercise 2
def ex2():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}bodyfat.txt"
    df = pd.read_csv(path, sep="\t")
    plt.rcParams.update({"font.size": 22})

    # This line allows us to set the figure size supposedly in inches.
    # When rendered in the IDE the output often does not translate to inches.
    print(df.head())
    plt.subplots(nrows=1, ncols=4, figsize=(24, 7))

    plt.subplot(1, 4, 1)  # Specifies total rows, columns and image #
    # where images are drawn clockwise.
    plt.xticks([], ())
    boxplot = df.boxplot(column=["Pct.BF"])

    plt.subplot(1, 4, 2)  # Specifies total rows, columns and image #
    # where images are drawn clockwise.
    boxplot = df.boxplot(column=["Age"])

    plt.subplot(1, 4, 3)  # Specifies total rows, columns and image #
    # where images are drawn clockwise.
    boxplot = df.boxplot(column=["Weight"])

    plt.subplot(1, 4, 4)  # Specifies total rows, columns and image #
    # where images are drawn clockwise.
    boxplot = df.boxplot(column=["Height"])

    plt.show()


# Exercise 3
def ex3():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_csv(path, sep="\t")

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    dfSub = df[["MomAge", "MomMarital", "numlive", "dobmm", "gestation"]]

    from scipy import stats
    import numpy as np

    z = np.abs(stats.zscore(dfSub))
    print(z)

    THRESHOLD = 2.33
    print(np.where(z > THRESHOLD))

    array = np.where(z > THRESHOLD)
    for i in range(0, len(array[0])):
        print(dfSub.loc[array[0][i]][[array[1][i]]])


# Exercise 4
def ex4():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(PATH + CSV_DATA)

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(dataset.head())

    # ------------------------------------------------------------------
    # Show statistics, boxplot, extreme values and returns DataFrame
    # row indexes where outliers exist.
    # ------------------------------------------------------------------
    def viewAndGetOutliers(df, colName, threshold, plt):
        # Show basic statistics.
        dfSub = df[[colName]]
        print("*** Statistics for " + colName)
        print(dfSub.describe())

        # Show boxplot.
        dfSub.boxplot(column=[colName])
        plt.title(colName)
        plt.show()

        # Note this is absolute 'abs' so it gets both high and low values.
        z = np.abs(stats.zscore(dfSub))
        rowColumnArray = np.where(z > threshold)
        rowIndices = rowColumnArray[0]

        # Show outlier rows.
        print("\nOutlier row indexes for " + colName + ":")
        print(rowIndices)
        print("")

        # Show filtered and sorted DataFrame with outliers.
        dfSub = df.iloc[rowIndices]
        dfSorted = dfSub.sort_values([colName], ascending=[True])
        print("\nDataFrame rows containing outliers for " + colName + ":")
        print(dfSorted)
        return rowIndices

    THRESHOLD_Z = 3
    priceOutlierRows = viewAndGetOutliers(dataset, "Avg. Area Income", THRESHOLD_Z, plt)


# Exercise 5
def ex5():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_csv(path, sep="\t")

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())

    # ------------------------------------------------------------------
    # Show statistics, boxplot, extreme values and returns DataFrame
    # row indexes where outliers exist.
    # ------------------------------------------------------------------
    def viewAndGetOutliers(df, colName, threshold, plt):
        # Show basic statistics.
        dfSub = df[[colName]]
        print("*** Statistics for " + colName)
        print(dfSub.describe())

        # Show boxplot.
        dfSub.boxplot(column=[colName])
        plt.title(colName)
        plt.show()

        # Note this is absolute 'abs' so it gets both high and low values.
        z = np.abs(stats.zscore(dfSub))
        rowColumnArray = np.where(z > threshold)
        rowIndices = rowColumnArray[0]

        # Show outlier rows.
        print("\nOutlier row indexes for " + colName + ":")
        print(rowIndices)
        print("")

        # Show filtered and sorted DataFrame with outliers.
        dfSub = df.iloc[rowIndices]
        dfSorted = dfSub.sort_values([colName], ascending=[True])
        print("\nDataFrame rows containing outliers for " + colName + ":")
        print(dfSorted)
        return rowIndices

    THRESHOLD_Z = 2.33
    priceOutlierRows = viewAndGetOutliers(df, "weight", THRESHOLD_Z, plt)


# Exercise 6
def ex6():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_csv(path, sep="\t")

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())

    # ------------------------------------------------------------------
    # Show statistics, boxplot, extreme values and returns DataFrame
    # row indexes where outliers exist outside an upper and lower percentile.
    # ------------------------------------------------------------------
    def viewAndGetOutliersByPercentile(df, colName, lowerP, upperP, plt):
        # Show basic statistics.
        dfSub = df[[colName]]
        print("*** Statistics for " + colName)
        print(dfSub.describe())

        # Show boxplot.
        dfSub.boxplot(column=[colName])
        plt.title(colName)
        plt.show()

        # Get upper and lower perctiles and filter with them.
        up = df[colName].quantile(upperP)
        lp = df[colName].quantile(lowerP)
        outlierDf = df[(df[colName] < lp) | (df[colName] > up)]

        # Show filtered and sorted DataFrame with outliers.
        dfSorted = outlierDf.sort_values([colName], ascending=[True])
        print("\nDataFrame rows containing outliers for " + colName + ":")
        print(dfSorted)

        return lp, up  # return lower and upper percentiles

    LOWER_PERCENTILE = 0.02
    UPPER_PERCENTILE = 0.98
    lp, up = viewAndGetOutliersByPercentile(
        df, "gestation", LOWER_PERCENTILE, UPPER_PERCENTILE, plt
    )


# Exercise 7
def ex7():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "wnba.csv"
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=("PLAYER", "GP", "PTS"),
    )

    # Trim prices to fixed boundaries.
    dfAdjusted = df["GP"].clip(0, 36)
    df["GPAdjusted"] = dfAdjusted
    dfAdjusted = df["PTS"].clip(0, 860)
    df["PTSAdjusted"] = dfAdjusted
    print(df.head(30))


# Exercise 8
def ex8():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "wnba.csv"
    dataset = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=("PLAYER", "GP", "PTS"),
    )

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Dataframe length: " + str(len(dataset)))
    df_filtered = dataset[
        (dataset["GP"] >= 0)
        & (dataset["GP"] <= 36)
        & (dataset["PTS"] >= 0)
        & (dataset["PTS"] <= 860)
    ]
    print("\nDataframe length after filtering: " + str(len(df_filtered)))
    print("\nFiltered Dataframe:")
    print(df_filtered)


def main():
    ex2()
    ex3()
    ex4()
    ex5()
    ex6()
    ex7()
    ex8()


if __name__ == "__main__":
    main()
