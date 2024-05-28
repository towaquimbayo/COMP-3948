# Exercise 1
def ex1():
    import numpy as np
    import matplotlib.pyplot as plt

    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    colorado = [
        5029196,
        5029316,
        5048281,
        5121771,
        5193721,
        5270482,
        5351218,
        5452107,
        5540921,
        5615902,
        5695564,
    ]
    connecticut = [
        3574097,
        3574147,
        3579125,
        3588023,
        3594395,
        3594915,
        3594783,
        3587509,
        3578674,
        3573880,
        3572665,
    ]
    delaware = [
        897934,
        897934,
        899595,
        907316,
        915188,
        923638,
        932596,
        941413,
        949216,
        957078,
        967171,
    ]

    # red dashes, blue squares and green solid line
    plt.plot(years, colorado, "--", color="red", label="Colorado")
    plt.plot(years, connecticut, "s", color="blue", label="Connecticut")
    plt.plot(years, delaware, color="green", label="Delaware")

    # legend
    # https://matplotlib.org/users/legend_guide.html
    plt.ylim(ymin=0)  # Set's y axis start to 0.
    plt.legend(loc=0)
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("Population by State")

    plt.show()


# Exercise 2
def ex2():
    import matplotlib.pyplot as plt

    # Plot scatter of pounds and inches.
    pounds = [120, 110, 160]
    inches = [50, 48, 68]
    plt.scatter(pounds, inches, color="orange", label="Students Region A")

    # Add a legend, axis labels, and title.
    plt.legend()
    plt.xlabel("Weight (Pounds)")
    plt.ylabel("Height (Inches)")
    plt.title("Height vs. Weight for Students Region A")

    plt.show()


# Exercise 3
def ex3():
    import matplotlib.pyplot as plt

    # Plot scatter of pounds and inches.
    poundsA = [120, 110, 160]
    inchesA = [50, 48, 68]
    poundsB = [121, 108, 150, 121, 121, 146]
    inchesB = [49, 45, 85, 46, 50, 85]
    plt.scatter(poundsA, inchesA, color="orange", label="Students Region A")
    plt.scatter(poundsB, inchesB, color="green", label="Students Region B")

    # Add a legend, axis labels, and title.
    plt.legend()
    plt.xlabel("Weight (Pounds)")
    plt.ylabel("Height (Inches)")
    plt.title("Height vs. Weight for Students Region A and B")

    plt.show()


# Exercise 4
def ex4():
    import matplotlib.pyplot as plt
    import numpy as np

    NUM_MEANS = 4
    NUM_GROUPS = 3
    bc_means = [20, 35, 30, 35, 27]
    alberta_means = [25, 32, 34, 20, 25]
    saskatchewan_means = [18, 28, 32, 24, 31]

    # This generates indices from 0 to 4 in a format that is accepted for
    # plotting bar charts.
    ind = np.arange(NUM_MEANS)
    print(ind)
    width = 0.25
    plt.bar(ind, bc_means[:4], width, label="BC")
    plt.bar(ind + width, alberta_means[:4], width, label="AB")
    plt.bar(ind + width * 2, saskatchewan_means[:4], width, label="SA")

    plt.ylabel("Revenue")
    plt.title("Quarterly Revenue by Province")

    plt.xticks(ind + width, ("Q1", "Q2", "Q3", "Q4"))
    plt.legend(loc="best")
    plt.show()


# Exercise 5
def ex5():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}bodyfat.txt"
    df = pd.read_csv(
        path,
        skiprows=1,
        sep="\t",
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
            "Ankle",
            "Knee",
            "Bicep",
            "Forearm",
            "Wrist",
        ),
    )

    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)
    # This line allows us to set the figure size supposedly in inches.
    # When rendered in the IDE the output often does not translate to inches.
    plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

    plt.subplot(2, 2, 1)
    plt.hist(df["Pct.BF"], bins=10)
    plt.xlabel("Pct BF")

    plt.subplot(2, 2, 2)
    plt.hist(df["Age"], bins=10)
    plt.xlabel("Age")

    plt.subplot(2, 2, 3)
    plt.hist(df["Weight"], bins=10)
    plt.xlabel("Weight")

    # 1 is married. 2 is unmarried.
    plt.subplot(2, 2, 4)
    plt.hist(df["Height"], bins=10)
    plt.xlabel("Height")

    plt.show()


# Exercise 6
def ex6():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas.plotting import scatter_matrix
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

    sub_df = df[["MomAge", "gestation", "weight"]]
    scatter_matrix(sub_df, figsize=(6, 6))
    plt.show()


# Exercise 7
def ex7():
    import matplotlib.pylab as plt
    import pandas as pd
    import seaborn as sns
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
    sub_df = df[["MomAge", "gestation", "weight"]]
    # Compute the correlation matrix
    corr = sub_df.corr()

    # plot the heatmap
    sns.set(rc={"figure.figsize": (6, 4)})
    sns.heatmap(corr[["weight"]], linewidths=0.1, vmin=-1, vmax=1, cmap="YlGnBu")
    plt.show()


# Exercise 8
def ex8():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT * FROM RetailInventory"
    results = showQueryResult(SQL)
    print(results)


# Exercise 9
def ex9():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT * FROM RetailInventory WHERE price >= 4"
    results = showQueryResult(SQL)
    print(results)


# Exercise 10
def ex10():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT * FROM RetailInventory ORDER BY productName"
    results = showQueryResult(SQL)
    print(results)


# Exercise 11
def ex11():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT productName, vendor, quantity, price FROM RetailInventory ORDER BY productName, quantity"
    results = showQueryResult(SQL)
    print(results)


# Exercise 12
def ex12():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT * FROM RetailInventory WHERE vendor NOT IN('Waterford Corp.', 'Cadbury')"
    results = showQueryResult(SQL)
    print(results)


# Exercise 13
def ex13():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = (
        "SELECT productName, price, price * 1.07 AS afterTaxPrice FROM RetailInventory"
    )
    results = showQueryResult(SQL)
    print(results)


# Exercise 14
def ex14():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT vendor, productName FROM RetailInventory WHERE vendor LIKE '%ry'"
    results = showQueryResult(SQL)
    print(results)


# Exercise 15
def ex15():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT productName FROM RetailInventory WHERE productName LIKE 'F%'"
    results = showQueryResult(SQL)
    print(results)


# Exercise 16
def ex16():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT MAX(price) AS 'Maximum Price' FROM RetailInventory"
    results = showQueryResult(SQL)
    print(results.iloc[0]["Maximum Price"])


# Exercise 17
def ex17():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT vendor, SUM(price * quantity) AS revenue FROM RetailInventory GROUP BY vendor"
    results = showQueryResult(SQL)
    print(results)


# Exercise 18
def ex18():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT productName, MIN(price) AS 'Minimum Price' FROM RetailInventory GROUP BY productName"
    results = showQueryResult(SQL)
    print(results)


# Exercise 19
def ex19():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT vendor, SUM(quantity) AS 'Total Quantity' FROM RetailInventory GROUP BY vendor HAVING vendor IN ('Cadbury', 'Waterford Corp.')"
    results = showQueryResult(SQL)
    print(results)


# Exercise 20
def ex20():
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # The data file path and file name need to be configured.
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "retailerDB.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Placed query in this function to enable code re-usability.
    def showQueryResult(sql):
        # This code creates an in-memory table called 'Inventory'.
        engine = create_engine("sqlite://", echo=False)
        connection = engine.connect()
        df.to_sql(
            name="RetailInventory", con=connection, if_exists="replace", index=False
        )

        # This code performs the query.
        queryResult = pd.read_sql(sql, connection)
        return queryResult

    # Read all rows from the table.
    SQL = "SELECT vendor, SUM(quantity * price) AS 'revenueValue' FROM RetailInventory GROUP BY vendor HAVING vendor IN ('Silverware Inc.', 'Waterford Corp.')"
    results = showQueryResult(SQL)
    print(results)


def main():
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
    ex6()
    ex7()
    ex8()
    ex9()
    ex10()
    ex11()
    ex12()
    ex13()
    ex14()
    ex15()
    ex16()
    ex17()
    ex18()
    ex19()
    ex20()


if __name__ == "__main__":
    main()
