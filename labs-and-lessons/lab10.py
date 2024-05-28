# Exercise 1
def ex1():
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    import statsmodels.api as sm
    import numpy as np

    # ------------------------------------------------
    # Shows plot of x vs. y.
    # ------------------------------------------------
    def showXandYplot(x, y, xtitle, title):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, color="blue")
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel("y")
        plt.show()

    # ------------------------------------------------
    # Shows plot of actual vs. predicted and RMSE.
    # ------------------------------------------------
    def showResidualPlotAndRMSE(x, y, predictions):
        xmax = max(x)
        xmin = min(x)
        residuals = y - predictions

        plt.figure(figsize=(8, 3))
        plt.title("x and y")
        plt.plot([xmin, xmax], [0, 0], "--", color="black")
        plt.title("Residuals")
        plt.scatter(x, residuals, color="red")
        plt.show()

        # Calculate RMSE
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        print("RMSE: " + str(rmse))

    # Section A: Define the raw data.
    # For sqrt(x)
    # x = [0, 20, 50, 100, 150, 200, 250, 300, 350, 400]
    # y = [0.0, 4.47, 7.07, 10.0, 12.24, 14.14, 15.81, 17.32, 18.70, 20.0]

    # For 1/x
    # x = [0.2, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5, 6]
    # y = [5.0, 2.0, 1.44, 1.12, 1.1, 0.53, 0.34, 0.25, 0.2, 0.17]

    # For 1/-x
    # x = [0.2, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5, 6]
    # y = [-5.0, -2.0, -1.43, -1.2, -1.0, -0.5, -0.34, -0.25, -0.2, -0.16]

    # For x^2
    # x = [
    #     -6,
    #     -5,
    #     -4,
    #     -3,
    #     -2,
    #     -1,
    #     -0.9,
    #     -0.7,
    #     -0.5,
    #     -0.2,
    #     0,
    #     0.2,
    #     0.5,
    #     0.7,
    #     0.9,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    # ]
    # y = [
    #     36,
    #     25,
    #     16,
    #     9,
    #     4,
    #     1,
    #     0.81,
    #     0.49,
    #     0.25,
    #     0.04,
    #     0,
    #     0.04,
    #     0.25,
    #     0.49,
    #     0.81,
    #     1,
    #     4,
    #     9,
    #     16,
    #     25,
    #     36,
    # ]

    # For log(x)
    # x = [
    #     0.01,
    #     0.2,
    #     0.5,
    #     0.7,
    #     0.9,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     10,
    #     11,
    #     12,
    #     13,
    #     14,
    #     15,
    #     16,
    #     17,
    #     18,
    #     19,
    #     20,
    # ]
    # y = [
    #     -4.61,
    #     -1.61,
    #     -0.693,
    #     -0.357,
    #     -0.1054,
    #     0.0,
    #     0.6931,
    #     1.099,
    #     1.3862,
    #     1.6094,
    #     1.792,
    #     2.303,
    #     2.398,
    #     2.48491,
    #     2.56,
    #     2.6390,
    #     2.71,
    #     2.772,
    #     2.83,
    #     2.89,
    #     2.94,
    #     2.99,
    # ]

    # For -log(x)
    x = [
        0.01,
        0.2,
        0.5,
        0.7,
        0.9,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]
    y = [
        4.60,
        1.60,
        0.69,
        0.35,
        0.11,
        0.0,
        -0.69,
        -1.1,
        -1.4,
        -1.61,
        -1.79,
        -2.3,
        -2.4,
        -2.49,
        -2.57,
        -2.64,
        -2.71,
        -2.77,
        -2.83,
        -2.89,
        -2.94,
        -2.996,
    ]

    print(y)

    showXandYplot(x, y, "x", "x and y")

    # Show raw x and y relationship
    dfX = pd.DataFrame({"x": x})
    dfY = pd.DataFrame({"y": y})
    dfX = sm.add_constant(dfX)

    # Show residuals from y(x)
    model = sm.OLS(y, dfX).fit()
    predictions = model.predict(dfX)
    print(model.summary())
    showResidualPlotAndRMSE(x, y, predictions)

    # Section B: Transform and plot graph with transformed x
    # By square root.
    # dfX["xt"] = np.sqrt(dfX["x"])
    # showXandYplot(dfX["xt"], y, "x", "y=sqrt(x)")

    # By 1/x
    # dfX["xt"] = 1 / dfX["x"]
    # showXandYplot(dfX["xt"], y, "x", "y=1/x")

    # By 1/-x
    # dfX["xt"] = 1 / -dfX["x"]
    # showXandYplot(dfX["xt"], y, "sqrt(x)", "1/-x vs. y")

    # By x^2
    # dfX["xt"] = dfX["x"] * dfX["x"]
    # showXandYplot(dfX["xt"], y, "x", "y=x^2")

    # By log(x)
    # dfX["xt"] = np.log(dfX["x"])
    # showXandYplot(dfX["xt"], y, "sqrt(x)", "log(x)")

    # By -log(x)
    dfX["xt"] = -np.log(dfX["x"])
    showXandYplot(dfX["xt"], y, "sqrt(x)", "-log(x)")

    # Build model with transformed x.
    model_t = sm.OLS(y, dfX[["const", "xt"]]).fit()
    predictions_t = model_t.predict(dfX[["const", "xt"]])
    print(model_t.summary())
    showResidualPlotAndRMSE(x, y, predictions_t)


# Exercise 2
def ex2():
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    import statsmodels.api as sm
    import numpy as np

    # ------------------------------------------------
    # Shows plot of x vs. y.
    # ------------------------------------------------
    def showXandYplot(x, y, xtitle, title):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, color="blue")
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel("y")
        plt.show()

    # ------------------------------------------------
    # Shows plot of actual vs. predicted and RMSE.
    # ------------------------------------------------
    def showResidualPlotAndRMSE(x, y, predictions):
        xmax = max(x)
        xmin = min(x)
        residuals = y - predictions

        plt.figure(figsize=(8, 3))
        plt.title("x and y")
        plt.plot([xmin, xmax], [0, 0], "--", color="black")
        plt.title("Residuals")
        plt.scatter(x, residuals, color="red")
        plt.show()

        # Calculate RMSE
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        print("RMSE: " + str(rmse))

    # Section A: Define the raw data.
    # For exp(x)
    # x = [
    #     0.01,
    #     0.2,
    #     0.5,
    #     0.7,
    #     0.9,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     10,
    #     11,
    #     12,
    #     13,
    #     14,
    #     15,
    #     16,
    #     17,
    #     18,
    #     19,
    #     20,
    # ]
    # y = [
    #     1.01,
    #     1.22,
    #     1.65,
    #     2.01,
    #     2.5,
    #     2.718,
    #     7.389,
    #     20.09,
    #     54.598,
    #     148.41,
    #     403.43,
    #     22026.46,
    #     59874.14,
    #     162754.79,
    #     442413.3,
    #     604.28,
    #     3269017.37,
    #     8886110.52,
    #     24154952.75,
    #     65659969.14,
    #     178482300.96,
    #     485165195.4,
    # ]

    # For exp(-x)
    x = [
        0.01,
        0.2,
        0.5,
        0.7,
        0.9,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]
    y = [
        0.99,
        0.819,
        0.6065,
        0.4966,
        0.40657,
        0.368,
        0.1353,
        0.0498,
        0.01831,
        0.00674,
        0.0025,
        4.5399e-05,
        1.670e-05,
        6.1e-06,
        2.260e-06,
        8.3153e-07,
        3.0590e-07,
        1.12e-07,
        4.14e-08,
        1.52e-08,
        5.60e-09,
        2.061e-09,
    ]

    print(y)

    showXandYplot(x, y, "x", "x and y")

    # Show raw x and y relationship
    dfX = pd.DataFrame({"x": x})
    dfY = pd.DataFrame({"y": y})
    dfX = sm.add_constant(dfX)

    # Show residuals from y(x)
    model = sm.OLS(y, dfX).fit()
    predictions = model.predict(dfX)
    print(model.summary())
    showResidualPlotAndRMSE(x, y, predictions)

    # Section B: Transform and plot graph with transformed x
    # By exp(x)
    # dfX["xt"] = np.exp(dfX["x"])
    # showXandYplot(dfX["xt"], y, "x", "y=exp(x)")

    # By exp(-x)
    dfX["xt"] = np.exp(-dfX["x"])
    showXandYplot(dfX["xt"], y, "x", "y=exp(-x)")

    # Build model with transformed x.
    model_t = sm.OLS(y, dfX[["const", "xt"]]).fit()
    predictions_t = model_t.predict(dfX[["const", "xt"]])
    print(model_t.summary())
    showResidualPlotAndRMSE(x, y, predictions_t)


# Exercise 3
def ex3():
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # ------------------------------------------------
    # Shows plot of x vs. y.
    # ------------------------------------------------
    def showXandYplot(x, y, xtitle, title):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, color="blue")
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel("y")
        plt.show()

    # ------------------------------------------------
    # Shows plot of actual vs. predicted and RMSE.
    # ------------------------------------------------
    def showResidualPlotAndRMSE(x, y, predictions):
        xmax = max(x)
        xmin = min(x)
        residuals = y - predictions

        plt.figure(figsize=(8, 3))
        plt.title("x and y")
        plt.plot([xmin, xmax], [0, 0], "--", color="black")
        plt.title("Residuals")
        plt.scatter(x, residuals, color="red")
        plt.show()

        # Calculate RMSE
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        print("RMSE: " + str(rmse))

    def grid_search(dfX, y, trans):
        trans_func = {
            "sqrt": lambda x: np.sqrt(x),
            "inv": lambda x: 1 / x,
            "neg_inv": lambda x: -1 / x,
            "sqr": lambda x: x * x,
            "log": lambda x: np.log(x),
            "neg_log": lambda x: -np.log(x),
            "exp": lambda x: np.exp(x),
            "neg_exp": lambda x: np.exp(-x),
        }
        dfTransformations = pd.DataFrame()
        for tran in trans:
            # Transform x
            dfX["xt"] = trans_func[tran](dfX["x"])
            model_t = sm.OLS(y, dfX[["const", "xt"]]).fit()
            predictions_t = model_t.predict(dfX[["const", "xt"]])
            # print(model_t.summary())
            # showResidualPlotAndRMSE(dfX["x"].values, y["ug/L"].values, predictions_t)
            # Calculate RMSE
            mse = mean_squared_error(y, predictions_t)
            rmse = np.sqrt(mse)
            dfTransformations = dfTransformations._append(
                {"tran": tran, "rmse": rmse}, ignore_index=True
            )
        dfTransformations = dfTransformations.sort_values(by=["rmse"])
        return dfTransformations

    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "abs.csv"
    df = pd.read_csv(PATH + FILE)

    x = df[["abs(450nm)"]]  # absorbance
    y = df[["ug/L"]]  # protein concentration
    showXandYplot(x, y, "absorbance x", "Protein Concentration(y) and Absorbance(x)")

    # Show raw x and y relationship
    x = sm.add_constant(x)

    # Show model.
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)
    print(model.summary())

    # Show RMSE.
    preddf = pd.DataFrame({"predictions": predictions})
    residuals = y["ug/L"] - preddf["predictions"]
    resSq = [i**2 for i in residuals]
    rmse = np.sqrt(np.sum(resSq) / len(resSq))
    print("RMSE: " + str(rmse))

    # Show the residual plot
    plt.scatter(x["abs(450nm)"], residuals)
    plt.show()

    dfX = pd.DataFrame({"x": df["abs(450nm)"].values})
    dfX = sm.add_constant(dfX)

    transformation_results = grid_search(
        dfX, y, ("sqrt", "neg_inv", "log", "exp", "neg_exp")
    )
    print(transformation_results["tran"].values[0])
    trans = transformation_results["tran"].values[0]
    if trans == "sqrt":
        dfX["xt"] = np.sqrt(dfX["x"])
    elif trans == "neg_inv":
        dfX["xt"] = 1 / -dfX["x"]
    elif trans == "log":
        dfX["xt"] = np.log(dfX["x"])
    elif trans == "exp":
        dfX["xt"] = np.exp(dfX["x"])
    elif trans == "neg_exp":
        dfX["xt"] = np.exp(-dfX["x"])

    model = sm.OLS(y, dfX[["const", "xt"]]).fit()
    predictions = model.predict(dfX[["const", "xt"]])
    print(model.summary())
    showResidualPlotAndRMSE(dfX["x"].values, y["ug/L"].values, predictions)


def main():
    # ex1()
    # ex2()
    ex3()


if __name__ == "__main__":
    main()
