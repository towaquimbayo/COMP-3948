# Exercise 1
def ex1():
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Import Data
    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "carPrice.csv"
    df = pd.read_csv(PATH + FILE)

    # Enable the display of all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    tempDf = df[["transmission", "fuel type2"]]
    dummyDf = pd.get_dummies(tempDf, columns=["transmission", "fuel type2"]).astype(int)
    df = pd.concat(([df, dummyDf]), axis=1)

    # ---------------------------------------------
    # Generate quick views of data.
    def viewQuickStats():
        print("\n*** Show contents of the file.")
        print(df.head())

        print("\n*** Show the description for all columns.")
        print(df.info())

        print("\n*** Describe numeric values.")
        print(df.describe())

        print("\n*** Showing frequencies.")

        # Show frequencies.
        print(df["model"].value_counts())
        print("")
        print(df["transmission"].value_counts())
        print("")
        print(df["fuel type"].value_counts())
        print("")
        print(df["engine size"].value_counts())
        print("")
        print(df["fuel type2"].value_counts())
        print("")
        print(df["year"].value_counts())
        print("")

    # ---------------------------------------------
    # Fix the price column.
    for i in range(0, len(df)):
        priceStr = str(df.iloc[i]["price"])
        priceStr = priceStr.replace("£", "")
        priceStr = priceStr.replace("-", "")
        priceStr = priceStr.replace(",", "")
        df.at[i, "price"] = priceStr

    # Convert column to number.
    df["price"] = pd.to_numeric(df["price"])

    # ---------------------------------------------
    # Fix the price column.
    averageYear = df["year"].mean()
    for i in range(0, len(df)):
        year = df.iloc[i]["year"]

        if np.isnan(year):
            df.at[i, "year"] = averageYear

    # ---------------------------------------------
    # Fix the engine size2 column.
    for i in range(0, len(df)):
        try:
            engineSize2 = df.loc[i]["engine size2"]
            if pd.isna(engineSize2):
                df.at[i, "engine size2"] = "0"

        except Exception as e:
            error = str(e)
            print(error)

    df["engine size2"] = pd.to_numeric(df["engine size2"])
    df["mileage2"].value_counts()
    viewQuickStats()

    # ---------------------------------------------
    # Fix the mileage column.
    for i in range(0, len(df)):
        mileageStr = str(df.iloc[i]["mileage"])
        mileageStr = mileageStr.replace(",", "")
        df.at[i, "mileage"] = mileageStr
        try:
            if not mileageStr.isnumeric():
                df.at[i, "mileage"] = "0"
        except Exception as e:
            error = str(e)
            print(error)

    df["mileage"] = pd.to_numeric(df["mileage"])
    viewQuickStats()

    # Compute the correlation matrix
    # corr = df.corr()
    # plot the heatmap
    # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    # plt.show()

    from sklearn.linear_model import LinearRegression

    X = df[["engine size2", "year"]]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import statsmodels.api as sm

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)

    y = df["price"]

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
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Import Data
    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "carPrice.csv"
    df = pd.read_csv(PATH + FILE)

    # Enable the display of all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    tempDf = df[["transmission", "fuel type2"]]
    dummyDf = pd.get_dummies(tempDf, columns=["transmission", "fuel type2"]).astype(int)
    df = pd.concat(([df, dummyDf]), axis=1)

    # ---------------------------------------------
    # Generate quick views of data.
    def viewQuickStats():
        print("\n*** Show contents of the file.")
        print(df.head())

        print("\n*** Show the description for all columns.")
        print(df.info())

        print("\n*** Describe numeric values.")
        print(df.describe())

        print("\n*** Showing frequencies.")

        # Show frequencies.
        print(df["model"].value_counts())
        print("")
        print(df["transmission"].value_counts())
        print("")
        print(df["fuel type"].value_counts())
        print("")
        print(df["engine size"].value_counts())
        print("")
        print(df["fuel type2"].value_counts())
        print("")
        print(df["year"].value_counts())
        print("")

    # ---------------------------------------------
    # Fix the price column.
    for i in range(0, len(df)):
        priceStr = str(df.iloc[i]["price"])
        priceStr = priceStr.replace("£", "")
        priceStr = priceStr.replace("-", "")
        priceStr = priceStr.replace(",", "")
        df.at[i, "price"] = priceStr

    # Convert column to number.
    df["price"] = pd.to_numeric(df["price"])

    # ---------------------------------------------
    # Fix the price column.
    averageYear = df["year"].mean()
    for i in range(0, len(df)):
        year = df.iloc[i]["year"]

        if np.isnan(year):
            df.at[i, "year"] = averageYear

    # ---------------------------------------------
    # Fix the engine size2 column.
    for i in range(0, len(df)):
        try:
            engineSize2 = df.loc[i]["engine size2"]
            if pd.isna(engineSize2):
                df.at[i, "engine size2"] = "0"

        except Exception as e:
            error = str(e)
            print(error)

    df["engine size2"] = pd.to_numeric(df["engine size2"])
    df["mileage2"].value_counts()
    viewQuickStats()

    # ---------------------------------------------
    # Fix the mileage column.
    for i in range(0, len(df)):
        mileageStr = str(df.iloc[i]["mileage"])
        mileageStr = mileageStr.replace(",", "")
        df.at[i, "mileage"] = mileageStr
        try:
            if not mileageStr.isnumeric():
                df.at[i, "mileage"] = "0"
        except Exception as e:
            error = str(e)
            print(error)

    df["mileage"] = pd.to_numeric(df["mileage"])
    viewQuickStats()

    from sklearn.linear_model import LinearRegression

    X = df[
        [
            "engine size2",
            "year",
            "transmission_Automatic",
            "transmission_Manual",
            "transmission_Other",
            "transmission_Semi-Auto",
            "fuel type2_Diesel",
            "fuel type2_Hybrid",
            "fuel type2_Other",
            "fuel type2_Petrol",
        ]
    ]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import statsmodels.api as sm

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)

    y = df["price"]

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


# Exercise 3
def ex3():
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Import Data
    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "carPrice.csv"
    df = pd.read_csv(PATH + FILE)

    # Enable the display of all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    tempDf = df[["transmission", "fuel type2"]]
    dummyDf = pd.get_dummies(tempDf, columns=["transmission", "fuel type2"]).astype(int)
    df = pd.concat(([df, dummyDf]), axis=1)

    df["yearBin"] = pd.cut(
        x=df["year"],
        bins=[
            0,
            2013,
            2014,
            2015,
            2016,
            2017,
            2018,
            2019,
            2020,
        ],  # anything less than 2013
    )
    tempDf = df[["yearBin"]]  # Isolate columns
    dummyDf = pd.get_dummies(tempDf, columns=["yearBin"]).astype(int)
    df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original

    # ---------------------------------------------
    # Generate quick views of data.
    def viewQuickStats():
        print("\n*** Show contents of the file.")
        print(df.head())

        print("\n*** Show the description for all columns.")
        print(df.info())

        print("\n*** Describe numeric values.")
        print(df.describe())

        print("\n*** Showing frequencies.")

        # Show frequencies.
        print(df["model"].value_counts())
        print("")
        print(df["transmission"].value_counts())
        print("")
        print(df["fuel type"].value_counts())
        print("")
        print(df["engine size"].value_counts())
        print("")
        print(df["fuel type2"].value_counts())
        print("")
        print(df["year"].value_counts())
        print("")

    # ---------------------------------------------
    # Fix the price column.
    for i in range(0, len(df)):
        priceStr = str(df.iloc[i]["price"])
        priceStr = priceStr.replace("£", "")
        priceStr = priceStr.replace("-", "")
        priceStr = priceStr.replace(",", "")
        df.at[i, "price"] = priceStr

    # Convert column to number.
    df["price"] = pd.to_numeric(df["price"])

    # ---------------------------------------------
    # Fix the price column.
    averageYear = df["year"].mean()
    for i in range(0, len(df)):
        year = df.iloc[i]["year"]

        if np.isnan(year):
            df.at[i, "year"] = averageYear

    # ---------------------------------------------
    # Fix the engine size2 column.
    for i in range(0, len(df)):
        try:
            engineSize2 = df.loc[i]["engine size2"]
            if pd.isna(engineSize2):
                df.at[i, "engine size2"] = "0"

        except Exception as e:
            error = str(e)
            print(error)

    df["engine size2"] = pd.to_numeric(df["engine size2"])
    df["mileage2"].value_counts()
    viewQuickStats()

    # ---------------------------------------------
    # Fix the mileage column.
    for i in range(0, len(df)):
        mileageStr = str(df.iloc[i]["mileage"])
        mileageStr = mileageStr.replace(",", "")
        df.at[i, "mileage"] = mileageStr
        try:
            if not mileageStr.isnumeric():
                df.at[i, "mileage"] = "0"
        except Exception as e:
            error = str(e)
            print(error)

    df["mileage"] = pd.to_numeric(df["mileage"])
    viewQuickStats()

    from sklearn.linear_model import LinearRegression

    X = df[
        [
            "engine size2",
            "year",
            "transmission_Automatic",
            "transmission_Manual",
            "transmission_Other",
            "transmission_Semi-Auto",
            # "fuel type2_Diesel",  # insignificant
            "fuel type2_Hybrid",
            # "fuel type2_Other",  # insignificant
            "fuel type2_Petrol",
            "yearBin_(0, 2013]",
            "yearBin_(2013, 2014]",
            "yearBin_(2014, 2015]",
            "yearBin_(2015, 2016]",
            "yearBin_(2016, 2017]",
            "yearBin_(2017, 2018]",
            "yearBin_(2018, 2019]",
            "yearBin_(2019, 2020]",
        ]
    ]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import statsmodels.api as sm

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)

    y = df["price"]

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
    def get_car_price(
        actual_price,
        year,
        transmission_automatic,
        transmission_manual,
        transmission_other,
        transmission_semi_auto,
        fuel_type2_petrol,
        year_bin_0_2013,
        year_bin_2013_2014,
        year_bin_2014_2015,
        year_bin_2016_2017,
        year_bin_2017_2018,
        year_bin_2018_2019,
        year_bin_2019_2020,
    ):
        price = (
            -1184000
            + 743.0012 * year
            - 295400 * transmission_automatic
            - 300300 * transmission_manual
            - 292300 * transmission_other
            - 295800 * transmission_semi_auto
            + 2620.1692 * fuel_type2_petrol
            - 6754.5360 * year_bin_0_2013
            - 2955.9264 * year_bin_2013_2014
            - 2221.9470 * year_bin_2014_2015
            + 1734.1694 * year_bin_2016_2017
            + 3111.7976 * year_bin_2017_2018
            + 7423.2256 * year_bin_2018_2019
            + 13960 * year_bin_2019_2020
        )
        print("Car Price actual: " + str(actual_price) + "   predicted: " + str(price))

    get_car_price(3200, 2006.0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0)
    get_car_price(33800, 2020.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    get_car_price(16499, 2015.0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    get_car_price(23849, 2019.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0)
    get_car_price(30299, 2019.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0)


def main():
    ex1()
    ex2()
    ex3()
    ex4()


if __name__ == "__main__":
    main()
