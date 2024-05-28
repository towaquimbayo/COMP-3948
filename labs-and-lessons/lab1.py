# Exercise 1
def ex1():
    fullName = "Bob Jones"
    spacePosition = fullName.find(" ")
    lastName = fullName[spacePosition + 1 : len(fullName)]
    print(lastName)


# Exercise 2
def ex2():
    text = "She sells seashells by the sea shore."
    positions = []
    for i in range(0, len(text) - 1):
        if text[i : i + 3] == "sea":
            positions.append(i)
    print(positions)


# Exercise 3
def ex3():
    text = "A lazy dog jumped over a log."
    newText = text.replace("lazy ", "")
    print(newText)


# Exercise 4
def ex4():
    sentenceArray = ["A", "lazy", "dog", "jumped", "over", "a", "log."]
    delimiter = ","
    newSentence = delimiter.join(sentenceArray)
    print(newSentence)


# Exercise 5
def ex5():
    import pandas as pd

    dataSet = {
        "First Name": ["Jonny", "Holly", "Nira"],
        "Grade": [85, 95, 91],
        "Last Name": ["Staub", "Conway", "Arora"],
    }
    df = pd.DataFrame(dataSet, columns=["First Name", "Grade", "Last Name"])
    print(df)


# Exercise 6
def ex6():
    import pandas as pd

    dataSet = {
        "Market": ["S&P 500", "Dow", "Nikkei"],
        "Last": [2932.05, 26485.01, 21087.16],
    }
    df = pd.DataFrame(dataSet, columns=["Market", "Last"])

    change = [-21.51, -98.41, -453.83]
    df["Change"] = change

    print("\n*** Original DataFrame ***")
    print(df)

    percentageChange = [-0.73, -0.37, -2.11]
    df["Percentage Change"] = percentageChange

    print("\n*** Adjusted DataFrame ***")
    print(df)


# Exercise 7
def ex7():
    import pandas as pd

    dataSet = {
        "Market": ["S&P 500", "Dow", "Nikkei"],
        "Last": [2932.05, 26485.01, 21087.16],
    }

    df = pd.DataFrame(dataSet, columns=["Market", "Last"])

    print("\n*** Original DataFrame ***")
    print(df)
    print("\n")

    for i in range(len(df)):
        print(df.loc[i]["Last"])


# Exercise 9
def ex9():
    import pandas as pd

    # Create data set.
    dataSet = {
        "Market": ["S&P 500", "Dow", "Nikkei"],
        "Last": [2932.05, 26485.01, 21087.16],
    }

    # Create dataframe with data set and named columns.
    df1 = pd.DataFrame(dataSet, columns=["Market", "Last"])

    # Show original DataFrame.
    print("\n*** Original DataFrame ***")
    print(df1)

    dataSet2 = {"Market": ["Hang Seng", "DAX"], "Last": [26918.58, 11872.44]}
    df2 = pd.DataFrame(dataSet2, columns=["Market", "Last"])

    df1 = df1._append(df2)
    print("\n*** Adjusted DataFrame ***")
    print(df1)

    dataSet3 = {"Market": ["FTSE100"], "Last": [7407.06]}
    df3 = pd.DataFrame(dataSet3, columns=["Market", "Last"])

    df1 = df1._append(df3)
    print("\n*** Adjusted DataFrame Again ***")
    print(df1)


# Exercise 10
def ex10():
    import pandas as pd

    dataSet = {
        "Market": ["S&P 500", "Dow"],
        "Last": [2932.05, 26485.01],
    }

    # The dictionary is an object made of name value pairs.
    stockDictionary = {"Market": "Nikkei", "Last": 21087.16}

    # Create dataframe with data set and named columns.
    df = pd.DataFrame(dataSet, columns=["Market", "Last"])

    # Show original DataFrame.
    print("\n*** Original DataFrame ***")

    df = df._append(stockDictionary, ignore_index=True)
    print(df)

    # Show modified DataFrame.
    print("\n*** Modified DataFrame ***")

    stockDictionary = {"Market": "DAX", "Last": 11872.44}
    df = df._append(stockDictionary, ignore_index=True)
    print(df)


# Exercise 11
def ex11():
    import os
    import pandas as pd

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}bodyfat.txt"
    df = pd.read_table(
        path,
        skiprows=1,
        delim_whitespace=True,
        names=(
            "Density",
            "Pct.BF",
            "Age",
            "Weight",
            "Height",
            "Neck",
            "Chest",
            "Abdomen",
            "Waist",
            "Hip",
            "Thigh",
            "Knee",
            "Ankle",
            "Bicep",
            "Forearm",
            "Wrist",
        ),
    )

    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)

    print("\n FIRST 2 ROWS")
    print(df.head(2))
    print("\n LAST 2 ROWS")
    print(df.tail(2))
    # Show data types for each column.
    print("\n DATA TYPES")  # Prints title with space before.
    print(df.dtypes)
    # Show statistical summaries for numeric columns.
    print("\nSTATISTIC SUMMARIES for NUMERIC COLUMNS")
    print(df.describe().round(2))


# Exercise 12
def ex12():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}bodyfat.txt"
    df = pd.read_table(
        path,
        skiprows=1,
        delim_whitespace=True,
        names=(
            "Density",
            "Pct.BF",
            "Age",
            "Weight",
            "Height",
            "Neck",
            "Chest",
            "Abdomen",
            "Waist",
            "Hip",
            "Thigh",
            "Knee",
            "Ankle",
            "Bicep",
            "Forearm",
            "Wrist",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)

    df2 = df[["Height", "Waist", "Weight", "Pct.BF"]]
    print(df2.head())


# Exercise 13
def ex13():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}bodyfat.txt"
    df = pd.read_table(
        path,
        skiprows=1,
        delim_whitespace=True,
        names=(
            "Density",
            "Pct.BF",
            "Age",
            "Weight",
            "Height",
            "Neck",
            "Chest",
            "Abdomen",
            "Waist",
            "Hip",
            "Thigh",
            "Knee",
            "Ankle",
            "Bicep",
            "Forearm",
            "Wrist",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)

    df2 = df[["Height", "Waist", "Weight", "Pct.BF"]]
    df2 = df2.rename({"Pct.BF": "Percent Body Fat"}, axis=1)
    print(df2.head())


# Exercise 14
def ex14():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_csv(
        path,
        skiprows=1,
        sep="\t",
        names=(
            "MomAge",
            "DadAge",
            "MomEduc",
            "MomMarital",
            "numlive",
            "dobmm",
            "gestation",
            "sex",
            "weight",
            "prenatalstart",
            "orig.id",
            "preemie",
        ),
    )

    # Rename the columns so they are more reader-friendly.
    df = df.rename(
        {
            "MomAge": "Mom Age",
            "DadAge": "Dad Age",
            "MomEduc": "Mom Edu",
            "weight": "Weight",
        },
        axis=1,
    )  # new method
    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)

    print("TOP FREQUENCY FIRST")
    print(df["Mom Edu"].value_counts())


# Exercise 15
def ex15():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_csv(
        path,
        skiprows=1,
        sep="\t",
        names=(
            "MomAge",
            "DadAge",
            "MomEduc",
            "MomMarital",
            "numlive",
            "dobmm",
            "gestation",
            "sex",
            "weight",
            "prenatalstart",
            "orig.id",
            "preemie",
        ),
    )

    # Rename the columns so they are more reader-friendly.
    df = df.rename(
        {
            "MomAge": "Mom Age",
            "DadAge": "Dad Age",
            "MomEduc": "Mom Edu",
            "weight": "Weight",
        },
        axis=1,
    )  # new method
    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)

    # Sort by ascending gestation and then by ascending weight.
    dfSorted = df.sort_values(["gestation", "Weight"], ascending=[True, True])
    print(dfSorted)


# Exercise 16
def ex16():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_csv(
        path,
        skiprows=1,
        sep="\t",
        names=(
            "MomAge",
            "DadAge",
            "MomEduc",
            "MomMarital",
            "numlive",
            "dobmm",
            "gestation",
            "sex",
            "weight",
            "prenatalstart",
            "orig.id",
            "preemie",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)

    print("Count:\t" + str(df["MomAge"].count()))
    print("Min:\t" + str(df["MomAge"].min()))
    print("Max:\t" + str(df["MomAge"].max()))
    print("Mean:\t" + str(df["MomAge"].mean()))
    print("Median:\t" + str(df["MomAge"].median()))
    print("Standard Deviation:\t" + str(df["MomAge"].std()))


# Exercise 17
def ex17():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "phone_data.csv"

    # Note this has a comma separator.
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,
        encoding="ISO-8859-1",
        sep=",",
        names=("index", "date", "duration", "item", "month", "network", "network_type"),
    )
    # Get count of items per month.
    dfStats = (
        df.groupby("network")["index"]
        .count()
        .reset_index()
        .rename(columns={"index": "# Calls"})
    )

    # Get duration mean for network groups and convert to DataFrame.
    dfDurationMean = (
        df.groupby("network")["duration"]
        .mean()
        .reset_index()
        .rename(columns={"duration": "Duration Mean"})
    )

    # Get duration max for network groups and convert to DataFrame.
    dfDurationMax = (
        df.groupby("network")["duration"]
        .max()
        .reset_index()
        .rename(columns={"duration": "Duration Max"})
    )

    # Get duration min for network groups and convert to DataFrame.
    dfDurationMin = (
        df.groupby("network")["duration"]
        .min()
        .reset_index()
        .rename(columns={"duration": "Duration Min"})
    )

    # Get duration std for network groups and convert to DataFrame.
    dfDurationStd = (
        df.groupby("network")["duration"]
        .std()
        .reset_index()
        .rename(columns={"duration": "Duration Std"})
    )

    # Append duration mean to stats matrix.
    dfStats["Duration Mean"] = dfDurationMean["Duration Mean"]

    # Append duration max to stats matrix.
    dfStats["Duration Max"] = dfDurationMax["Duration Max"]

    # Append duration min to stats matrix.
    dfStats["Duration Min"] = dfDurationMin["Duration Min"]

    # Append duration std to stats matrix.
    dfStats["Duration Std"] = dfDurationStd["Duration Std"]
    print(dfStats)


# Exercise 19
def ex19():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_csv(
        path,
        skiprows=1,
        sep="\t",
        names=(
            "MomAge",
            "DadAge",
            "MomEduc",
            "MomMarital",
            "numlive",
            "dobmm",
            "gestation",
            "sex",
            "weight",
            "prenatalstart",
            "orig.id",
            "preemie",
        ),
    )

    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)

    # Get count of items per sex.
    dfStats = df.groupby("sex")["sex"].count().reset_index(name="# of sex")

    # Get weight mean for sex groups and convert to DataFrame.
    dfWeightMean = (
        df.groupby("sex")["weight"]
        .mean()
        .reset_index()
        .rename(columns={"weight": "Weight Mean"})
    )

    # Get weight max for sex groups and convert to DataFrame.
    dfWeightMax = (
        df.groupby("sex")["weight"]
        .max()
        .reset_index()
        .rename(columns={"weight": "Weight Max"})
    )

    # Get weight min for sex groups and convert to DataFrame.
    dfWeightMin = (
        df.groupby("sex")["weight"]
        .min()
        .reset_index()
        .rename(columns={"weight": "Weight Min"})
    )

    # Append weight mean to stats matrix.
    dfStats["Weight Mean"] = dfWeightMean["Weight Mean"]

    # Append weight max to stats matrix.
    dfStats["Weight Max"] = dfWeightMax["Weight Max"]

    # Append weight min to stats matrix.
    dfStats["Weight Min"] = dfWeightMin["Weight Min"]

    print(dfStats)


# Exercise 20
def ex20():
    import pandas as pd

    # Create data set.
    dataSet = {"Fahrenheit": [85, 95, 91]}

    # Create dataframe with data set and named columns.
    # Column names must match the dataSet properties.
    df = pd.DataFrame(dataSet, columns=["Fahrenheit"])

    df["Celsius"] = (df["Fahrenheit"] - 32) * 5 / 9
    df["Kelvin"] = df["Celsius"] + 273.15
    # Show DataFrame
    print(df)


# Exercise 21
def ex21():
    import pandas as pd
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "phone_data.csv"

    # Note this has a comma separator.
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,
        encoding="ISO-8859-1",
        sep=",",
        names=("index", "date", "duration", "item", "month", "network", "network_type"),
    )

    df2 = (
        df.groupby(["network_type", "item"])["index"]
        .count()
        .reset_index()
        .rename(columns={"network_type": "network type", "index": "total count"})
    )
    print(df2)


def main():
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
    ex6()
    ex7()
    ex9()
    ex10()
    ex11()
    ex12()
    ex13()
    ex14()
    ex15()
    ex16()
    ex17()
    ex19()
    ex20()
    ex21()


if __name__ == "__main__":
    main()
